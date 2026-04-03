from __future__ import annotations

from dataclasses import asdict, dataclass
import html
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import MOLNET_SPLITS, SaeExperimentConfig
from chem_sae.data import SmilesClassificationDataset
from chem_sae.eval.downstream import train_linear_probe
from chem_sae.modeling import RobertaForMaskedLM, build_mlm_model
from chem_sae.utils import append_csv_row, save_json
from chem_sae.vendor import JumpReLUAutoencoder, load_molnet_dataset


@dataclass(frozen=True)
class FeatureAuditTaskArtifacts:
    task: str
    label_column: str
    rankings: list[dict[str, Any]]
    selected_summaries: list[dict[str, Any]]
    feature_cards: list[dict[str, Any]]


def _valid_binary_arrays(
    activations: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim > 1:
        labels = labels[:, 0]
    if label_mask.ndim > 1:
        label_mask = label_mask[:, 0]
    mask = label_mask.astype(bool)
    return activations[mask], labels[mask]


def _safe_binary_auc(activations: np.ndarray, labels: np.ndarray) -> float:
    if activations.size == 0 or labels.size == 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, activations))


def _activation_frequency(activations: np.ndarray) -> float:
    if activations.size == 0:
        return float("nan")
    return float(np.mean(activations > 0))


def aggregate_probe_coefficients(coefficients: np.ndarray) -> list[dict[str, Any]]:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [num_seeds, num_features]")
    num_seeds, num_features = coefficients.shape
    if num_seeds < 1 or num_features < 1:
        return []

    abs_coefficients = np.abs(coefficients)
    ranks = np.zeros_like(abs_coefficients, dtype=np.float64)
    for seed_idx in range(num_seeds):
        order = np.argsort(-abs_coefficients[seed_idx], kind="stable")
        ranks[seed_idx, order] = np.arange(1, num_features + 1, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for feature_idx in range(num_features):
        feature_coefs = coefficients[:, feature_idx]
        pos_count = int(np.sum(feature_coefs > 0))
        neg_count = int(np.sum(feature_coefs < 0))
        sign_consistency = 0.0
        if num_seeds > 0:
            sign_consistency = max(pos_count, neg_count) / float(num_seeds)
        rows.append(
            {
                "feature_idx": int(feature_idx),
                "coef_mean": float(np.mean(feature_coefs)),
                "coef_std": float(np.std(feature_coefs)),
                "abs_coef_mean": float(np.mean(np.abs(feature_coefs))),
                "sign_consistency": float(sign_consistency),
                "rank_mean": float(np.mean(ranks[:, feature_idx])),
                "rank_std": float(np.std(ranks[:, feature_idx])),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["abs_coef_mean"]),
            -float(row["sign_consistency"]),
            float(row["rank_mean"]),
            int(row["feature_idx"]),
        )
    )
    return rows


def collect_top_activating_examples(
    *,
    smiles: list[str],
    activations: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    split_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if len(smiles) != int(activations.shape[0]):
        raise ValueError("smiles length must match activation rows")

    order = np.argsort(-activations, kind="stable")
    rows: list[dict[str, Any]] = []
    for rank_idx, row_idx in enumerate(order[:top_k], start=1):
        valid_label = bool(label_mask[row_idx])
        rows.append(
            {
                "rank": rank_idx,
                "split": split_name,
                "smiles": smiles[int(row_idx)],
                "activation": float(activations[row_idx]),
                "label": float(labels[row_idx]) if valid_label else None,
                "has_label": valid_label,
            }
        )
    return rows


def summarize_single_feature(
    *,
    task: str,
    feature_idx: int,
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    train_label_mask: np.ndarray,
    test_activations: np.ndarray,
    test_labels: np.ndarray,
    test_label_mask: np.ndarray,
    coefficient_stats: dict[str, Any],
) -> dict[str, Any]:
    valid_train_acts, valid_train_labels = _valid_binary_arrays(
        train_activations, train_labels, train_label_mask
    )
    valid_test_acts, valid_test_labels = _valid_binary_arrays(
        test_activations, test_labels, test_label_mask
    )

    combined_acts = np.concatenate([valid_train_acts, valid_test_acts], axis=0)
    combined_labels = np.concatenate([valid_train_labels, valid_test_labels], axis=0)

    positive_mask = combined_labels == 1
    negative_mask = combined_labels == 0

    positive_mean = float(np.mean(combined_acts[positive_mask])) if positive_mask.any() else float("nan")
    negative_mean = float(np.mean(combined_acts[negative_mask])) if negative_mask.any() else float("nan")

    row = {
        "task": task,
        "feature_idx": int(feature_idx),
        "single_feature_roc_auc": _safe_binary_auc(valid_test_acts, valid_test_labels),
        "positive_mean_activation": positive_mean,
        "negative_mean_activation": negative_mean,
        "activation_frequency": _activation_frequency(valid_train_acts),
    }
    row.update(coefficient_stats)
    return row


def load_jumprelu_from_checkpoint(
    checkpoint_path: Path,
    cfg: SaeExperimentConfig,
    device: torch.device,
) -> JumpReLUAutoencoder:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = dict(state["model"])
    n_inputs, n_latents = model_state["W_enc"].shape
    ae = JumpReLUAutoencoder(
        n_latents=n_latents,
        n_inputs=n_inputs,
        threshold_init=cfg.jumprelu_threshold,
        bandwidth=cfg.jumprelu_bandwidth,
        normalize=True,
        sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
        tanh_scale=cfg.jumprelu_tanh_scale,
        pre_act_loss_coefficient=cfg.jumprelu_pre_act_loss_coefficient,
    ).to(device)
    ae.process_state_dict_for_loading(model_state)
    ae.load_state_dict(model_state, strict=False)
    return ae


@torch.no_grad()
def compute_latent_features_with_smiles(
    model: RobertaForMaskedLM,
    ae: JumpReLUAutoencoder,
    dataloader: DataLoader,
    dataset: SmilesClassificationDataset,
    layer: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    ae.eval()

    feats = []
    labels = []
    label_masks = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["labels"].cpu()
        label_mask = batch["label_mask"].cpu()

        _, attn_outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attn_outputs=True,
            attn_output_layers={layer},
        )
        attn = attn_outputs[layer]
        flat = attn.reshape(-1, attn.shape[-1])
        latents, _ = ae.encode(flat)
        latents = latents.reshape(attn.shape[0], attn.shape[1], -1)
        feats.append(latents[:, 0, :].cpu())
        labels.append(label)
        label_masks.append(label_mask)

    return (
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
        torch.cat(label_masks, dim=0).numpy(),
        list(dataset.smiles),
    )


def fit_probe_coefficients(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_mask: np.ndarray,
    num_seeds: int,
    base_seed: int,
) -> np.ndarray:
    if y_train.ndim > 1:
        y_train = y_train[:, 0]
        y_train_mask = y_train_mask[:, 0]
    mask_train = y_train_mask.astype(bool)
    if mask_train.sum() == 0:
        raise ValueError("no valid labeled samples available for training")

    coefficients = []
    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        clf = train_linear_probe(X_train[mask_train], y_train[mask_train], random_state=seed)
        coefficients.append(clf.coef_[0].astype(np.float64))
    return np.stack(coefficients, axis=0)


def _render_feature_atlas_html(
    *,
    task_artifacts: list[FeatureAuditTaskArtifacts],
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    cards = []
    for task_artifact in task_artifacts:
        for summary in task_artifact.selected_summaries:
            cards.append(
                "<tr>"
                f"<td>{html.escape(task_artifact.task)}</td>"
                f"<td>{int(summary['feature_idx'])}</td>"
                f"<td>{float(summary['coef_mean']):.6f}</td>"
                f"<td>{float(summary['sign_consistency']):.3f}</td>"
                f"<td>{float(summary['single_feature_roc_auc']):.4f}</td>"
                f"<td>{float(summary['positive_mean_activation']):.4f}</td>"
                f"<td>{float(summary['negative_mean_activation']):.4f}</td>"
                "</tr>"
            )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Feature Atlas</title>
  <style>
    body {{ font-family: "IBM Plex Sans", sans-serif; margin: 24px; color: #111827; }}
    .meta {{ color: #4b5563; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Layer 0 Feature Atlas</h1>
  <div class="meta">checkpoint={html.escape(str(checkpoint_path))}</div>
  <table>
    <thead>
      <tr>
        <th>Task</th>
        <th>Feature</th>
        <th>Coef Mean</th>
        <th>Sign Consistency</th>
        <th>Single-Feature ROC-AUC</th>
        <th>Pos Mean Activation</th>
        <th>Neg Mean Activation</th>
      </tr>
    </thead>
    <tbody>
      {"".join(cards)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def audit_task_features(
    *,
    task: str,
    label_column: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_label_mask: np.ndarray,
    train_smiles: list[str],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_label_mask: np.ndarray,
    test_smiles: list[str],
    top_k: int,
    num_seeds: int,
    base_seed: int,
) -> FeatureAuditTaskArtifacts:
    coefficients = fit_probe_coefficients(
        X_train=train_features,
        y_train=train_labels,
        y_train_mask=train_label_mask,
        num_seeds=num_seeds,
        base_seed=base_seed,
    )
    rankings = aggregate_probe_coefficients(coefficients)

    positive_rankings = [row for row in rankings if float(row["coef_mean"]) > 0][:top_k]
    negative_rankings = [row for row in rankings if float(row["coef_mean"]) < 0][:top_k]
    selected = positive_rankings + negative_rankings

    selected_summaries = []
    feature_cards = []
    for rank_row in selected:
        feature_idx = int(rank_row["feature_idx"])
        summary = summarize_single_feature(
            task=task,
            feature_idx=feature_idx,
            train_activations=train_features[:, feature_idx],
            train_labels=train_labels,
            train_label_mask=train_label_mask,
            test_activations=test_features[:, feature_idx],
            test_labels=test_labels,
            test_label_mask=test_label_mask,
            coefficient_stats=rank_row,
        )
        selected_summaries.append(summary)
        feature_cards.append(
            {
                "task": task,
                "label_column": label_column,
                "summary": summary,
                "top_train_examples": collect_top_activating_examples(
                    smiles=train_smiles,
                    activations=train_features[:, feature_idx],
                    labels=train_labels[:, 0] if train_labels.ndim > 1 else train_labels,
                    label_mask=(
                        train_label_mask[:, 0]
                        if train_label_mask.ndim > 1
                        else train_label_mask
                    ),
                    split_name="train",
                    top_k=top_k,
                ),
                "top_test_examples": collect_top_activating_examples(
                    smiles=test_smiles,
                    activations=test_features[:, feature_idx],
                    labels=test_labels[:, 0] if test_labels.ndim > 1 else test_labels,
                    label_mask=(
                        test_label_mask[:, 0]
                        if test_label_mask.ndim > 1
                        else test_label_mask
                    ),
                    split_name="test",
                    top_k=top_k,
                ),
            }
        )

    return FeatureAuditTaskArtifacts(
        task=task,
        label_column=label_column,
        rankings=rankings,
        selected_summaries=selected_summaries,
        feature_cards=feature_cards,
    )


def run_feature_audit(
    cfg: SaeExperimentConfig,
    *,
    checkpoint_path: Path,
    layer: int,
    tasks: tuple[str, ...],
    top_k: int = 10,
) -> dict[str, Any]:
    cfg.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)
    ae = load_jumprelu_from_checkpoint(checkpoint_path, cfg, device)
    cfg.n_latents = int(ae.W_enc.shape[1])

    task_artifacts: list[FeatureAuditTaskArtifacts] = []
    rankings_path = cfg.run_context.reports_dir / "feature_rankings.csv"
    cards_dir = cfg.run_context.reports_dir / "feature_cards"
    summary_path = cfg.run_context.reports_dir / "feature_summary.json"
    atlas_path = cfg.run_context.reports_dir / "feature_atlas.html"

    for task in tasks:
        split_type = MOLNET_SPLITS.get(task, "scaffold")
        label_cols, (train_df, _valid_df, test_df), _ = load_molnet_dataset(
            task,
            split=split_type,
            df_format="chemprop",
            local_only=cfg.local_only,
        )
        train_dataset = SmilesClassificationDataset(
            train_df, tokenizer, list(label_cols), max_len=cfg.max_len
        )
        test_dataset = SmilesClassificationDataset(
            test_df, tokenizer, list(label_cols), max_len=cfg.max_len
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.downstream_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.downstream_batch_size, shuffle=False
        )
        train_features, train_labels, train_label_mask, train_smiles = (
            compute_latent_features_with_smiles(
                model, ae, train_loader, train_dataset, layer, device
            )
        )
        test_features, test_labels, test_label_mask, test_smiles = (
            compute_latent_features_with_smiles(
                model, ae, test_loader, test_dataset, layer, device
            )
        )
        task_result = audit_task_features(
            task=task,
            label_column=list(label_cols)[0],
            train_features=train_features,
            train_labels=train_labels,
            train_label_mask=train_label_mask,
            train_smiles=train_smiles,
            test_features=test_features,
            test_labels=test_labels,
            test_label_mask=test_label_mask,
            test_smiles=test_smiles,
            top_k=top_k,
            num_seeds=cfg.num_seeds,
            base_seed=cfg.seed,
        )
        task_artifacts.append(task_result)

        for row in task_result.rankings:
            append_csv_row(rankings_path, {"task": task, **row})

        for feature_card in task_result.feature_cards:
            card_path = cards_dir / task / f"feature_{int(feature_card['summary']['feature_idx']):04d}.json"
            save_json(card_path, feature_card)

    payload = {
        "run_id": cfg.run_id,
        "layer": layer,
        "checkpoint_path": str(checkpoint_path),
        "tasks": list(tasks),
        "top_k": top_k,
        "n_latents": cfg.n_latents,
        "task_artifacts": [asdict(task_artifact) for task_artifact in task_artifacts],
    }
    save_json(summary_path, payload)
    _render_feature_atlas_html(
        task_artifacts=task_artifacts,
        checkpoint_path=checkpoint_path,
        output_path=atlas_path,
    )
    return {
        "run_id": cfg.run_id,
        "rankings_path": str(rankings_path),
        "summary_path": str(summary_path),
        "atlas_path": str(atlas_path),
        "cards_dir": str(cards_dir),
    }
