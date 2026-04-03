from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import MOLNET_SPLITS, SaeExperimentConfig
from chem_sae.data import SmilesClassificationDataset
from chem_sae.eval.downstream import eval_roc_auc, train_linear_probe
from chem_sae.eval.feature_audit import (
    compute_latent_features_with_smiles,
    load_jumprelu_from_checkpoint,
)
from chem_sae.modeling import RobertaForMaskedLM, build_mlm_model
from chem_sae.utils import append_csv_row, save_json
from chem_sae.vendor import JumpReLUAutoencoder, load_molnet_dataset


def parse_feature_indices(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one feature index is required")
    return values


def sample_matched_control_features(
    *,
    num_features: int,
    group_size: int,
    seed: int,
    exclude: set[int],
) -> list[int]:
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    available = [idx for idx in range(num_features) if idx not in exclude]
    if len(available) < group_size:
        raise ValueError("not enough available features to sample control group")
    rng = random.Random(seed)
    return sorted(rng.sample(available, k=group_size))


def build_feature_intervention_result_row(
    *,
    run_id: str,
    task: str,
    layer: int,
    checkpoint_path: str,
    condition: str,
    feature_indices: list[int],
    mode: str,
    baseline_roc_auc: float,
    intervened_roc_auc: float,
    mean_logit_shift: float,
    mean_probability_shift: float,
    control_kind: str,
    baseline_roc_auc_std: float = 0.0,
    intervened_roc_auc_std: float = 0.0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "task": task,
        "layer": int(layer),
        "checkpoint_path": checkpoint_path,
        "condition": condition,
        "control_kind": control_kind,
        "feature_indices": ",".join(str(idx) for idx in feature_indices),
        "feature_count": len(feature_indices),
        "mode": mode,
        "baseline_roc_auc": float(baseline_roc_auc),
        "baseline_roc_auc_std": float(baseline_roc_auc_std),
        "intervened_roc_auc": float(intervened_roc_auc),
        "intervened_roc_auc_std": float(intervened_roc_auc_std),
        "roc_auc_delta": float(intervened_roc_auc - baseline_roc_auc),
        "mean_logit_shift": float(mean_logit_shift),
        "mean_probability_shift": float(mean_probability_shift),
    }


@torch.no_grad()
def extract_final_hidden_features_with_latent_intervention(
    model: RobertaForMaskedLM,
    dataloader: DataLoader,
    sae_dict: dict[int, JumpReLUAutoencoder],
    pattern: list[bool],
    device: torch.device,
    *,
    latent_intervention_dict: dict[int, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    for sae in sae_dict.values():
        sae.eval()

    feats = []
    labels = []
    label_masks = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["labels"].cpu()
        label_mask = batch["label_mask"].cpu()

        final_hidden = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sae_dict=sae_dict,
            intervention_pattern=pattern,
            latent_intervention_dict=latent_intervention_dict,
        )
        feats.append(final_hidden[:, 0, :].cpu())
        labels.append(label)
        label_masks.append(label_mask)

    return (
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
        torch.cat(label_masks, dim=0).numpy(),
    )


def _resolve_feature_values(
    *,
    latent_train_features: np.ndarray,
    feature_indices: list[int],
    mode: str,
    explicit_feature_values: float | list[float] | None,
) -> float | list[float] | None:
    if mode == "zero":
        return None
    if explicit_feature_values is not None:
        return explicit_feature_values

    selected = latent_train_features[:, feature_indices]
    if mode == "mean_clamp":
        return np.mean(selected, axis=0).tolist()
    if mode == "force_on":
        return np.quantile(selected, 0.95, axis=0).tolist()
    raise ValueError(f"unsupported feature intervention mode: {mode}")


def _extract_binary_labels(
    labels: np.ndarray,
    label_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim > 1:
        labels = labels[:, 0]
        label_mask = label_mask[:, 0]
    return labels, label_mask


def _aggregate_probe_with_intervention(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_mask: np.ndarray,
    X_test_baseline: np.ndarray,
    X_test_intervened: np.ndarray,
    y_test: np.ndarray,
    y_test_mask: np.ndarray,
    num_seeds: int,
    base_seed: int,
) -> tuple[float, float, float, float, float]:
    mask_train = y_train_mask.astype(bool)
    mask_test = y_test_mask.astype(bool)
    if mask_train.sum() == 0 or mask_test.sum() == 0:
        raise ValueError("no valid labeled samples available for intervention evaluation")

    baseline_aucs = []
    intervened_aucs = []
    logit_shifts = []
    probability_shifts = []
    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        clf = train_linear_probe(X_train[mask_train], y_train[mask_train], random_state=seed)
        baseline_probs = clf.predict_proba(X_test_baseline[mask_test])[:, 1]
        intervened_probs = clf.predict_proba(X_test_intervened[mask_test])[:, 1]
        baseline_aucs.append(eval_roc_auc(clf, X_test_baseline[mask_test], y_test[mask_test]))
        intervened_aucs.append(eval_roc_auc(clf, X_test_intervened[mask_test], y_test[mask_test]))
        baseline_logits = clf.decision_function(X_test_baseline[mask_test])
        intervened_logits = clf.decision_function(X_test_intervened[mask_test])
        logit_shifts.append(float(np.mean(intervened_logits - baseline_logits)))
        probability_shifts.append(float(np.mean(intervened_probs - baseline_probs)))

    return (
        float(np.mean(baseline_aucs)),
        float(np.std(baseline_aucs)),
        float(np.mean(intervened_aucs)),
        float(np.std(intervened_aucs)),
        float(np.mean(logit_shifts)),
        float(np.mean(probability_shifts)),
    )


def run_feature_intervention(
    cfg: SaeExperimentConfig,
    *,
    checkpoint_path: Path,
    layer: int,
    task: str,
    feature_indices: list[int],
    mode: str,
    control_kind: str = "none",
    control_seed: int | None = None,
    explicit_feature_values: float | list[float] | None = None,
) -> dict[str, Any]:
    cfg.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)
    ae = load_jumprelu_from_checkpoint(checkpoint_path, cfg, device)
    cfg.n_latents = int(ae.W_enc.shape[1])

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

    pattern = [False] * int(model.config.num_hidden_layers)
    pattern[layer] = True
    sae_dict = {layer: ae}

    baseline_train, y_train, y_train_mask = extract_final_hidden_features_with_latent_intervention(
        model,
        train_loader,
        sae_dict,
        pattern,
        device,
        latent_intervention_dict=None,
    )
    baseline_test, y_test, y_test_mask = extract_final_hidden_features_with_latent_intervention(
        model,
        test_loader,
        sae_dict,
        pattern,
        device,
        latent_intervention_dict=None,
    )
    y_train, y_train_mask = _extract_binary_labels(y_train, y_train_mask)
    y_test, y_test_mask = _extract_binary_labels(y_test, y_test_mask)

    latent_train_features, _latent_labels, _latent_masks, _train_smiles = (
        compute_latent_features_with_smiles(model, ae, train_loader, train_dataset, layer, device)
    )

    conditions: list[tuple[str, list[int], str]] = [("target", feature_indices, control_kind)]
    if control_kind == "matched_random":
        control_features = sample_matched_control_features(
            num_features=cfg.n_latents,
            group_size=len(feature_indices),
            seed=control_seed if control_seed is not None else cfg.seed,
            exclude=set(feature_indices),
        )
        conditions.append(("control", control_features, control_kind))

    results = []
    report_path = cfg.run_context.reports_dir / "feature_intervention_results.csv"
    for condition_name, selected_features, selected_control_kind in conditions:
        feature_values = _resolve_feature_values(
            latent_train_features=latent_train_features,
            feature_indices=selected_features,
            mode=mode,
            explicit_feature_values=explicit_feature_values,
        )
        intervention_spec = {
            layer: {
                "feature_indices": selected_features,
                "mode": mode,
                "feature_values": feature_values,
            }
        }
        intervened_test, _y_test_tmp, _mask_tmp = extract_final_hidden_features_with_latent_intervention(
            model,
            test_loader,
            sae_dict,
            pattern,
            device,
            latent_intervention_dict=intervention_spec,
        )
        (
            baseline_roc_auc,
            baseline_roc_auc_std,
            intervened_roc_auc,
            intervened_roc_auc_std,
            mean_logit_shift,
            mean_probability_shift,
        ) = _aggregate_probe_with_intervention(
            X_train=baseline_train,
            y_train=y_train,
            y_train_mask=y_train_mask,
            X_test_baseline=baseline_test,
            X_test_intervened=intervened_test,
            y_test=y_test,
            y_test_mask=y_test_mask,
            num_seeds=cfg.num_seeds,
            base_seed=cfg.seed,
        )
        row = build_feature_intervention_result_row(
            run_id=cfg.run_id or "unknown_run",
            task=task,
            layer=layer,
            checkpoint_path=str(checkpoint_path),
            condition=condition_name,
            feature_indices=selected_features,
            mode=mode,
            baseline_roc_auc=baseline_roc_auc,
            intervened_roc_auc=intervened_roc_auc,
            mean_logit_shift=mean_logit_shift,
            mean_probability_shift=mean_probability_shift,
            control_kind=selected_control_kind,
            baseline_roc_auc_std=baseline_roc_auc_std,
            intervened_roc_auc_std=intervened_roc_auc_std,
        )
        append_csv_row(report_path, row)
        results.append(row)

    summary_path = cfg.run_context.reports_dir / "feature_intervention_summary.json"
    summary_payload = {
        "run_id": cfg.run_id,
        "task": task,
        "layer": layer,
        "checkpoint_path": str(checkpoint_path),
        "mode": mode,
        "feature_indices": feature_indices,
        "control_kind": control_kind,
        "results": results,
    }
    save_json(summary_path, summary_payload)
    return {
        "run_id": cfg.run_id,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
