from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import MOLNET_SPLITS, SaeExperimentConfig
from chem_sae.data import SmilesClassificationDataset
from chem_sae.modeling import RobertaForMaskedLM
from chem_sae.utils import append_csv_row
from chem_sae.vendor import Autoencoder, load_molnet_dataset


def _resolve_sae_type(ae: Autoencoder, extra_fields: dict[str, Any] | None) -> str:
    arch = str(extra_fields.get("arch", "")).lower() if extra_fields else ""
    if arch == "jumprelu":
        return "JumpReLU"
    if arch in {"batchtopk", "topk"}:
        return "TopK"
    if ae.__class__.__name__.lower().startswith("jumprelu"):
        return "JumpReLU"
    return "TopK"


@torch.no_grad()
def compute_latent_features(
    model: RobertaForMaskedLM,
    ae: Autoencoder,
    dataloader: DataLoader,
    layer: int,
    device: torch.device,
):
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
        cls_latent = latents[:, 0, :]
        feats.append(cls_latent.cpu())
        labels.append(label)
        label_masks.append(label_mask)
    return (
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
        torch.cat(label_masks, dim=0).numpy(),
    )


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    clf = LogisticRegression(max_iter=1000, random_state=random_state, solver="saga")
    clf.fit(X_train, y_train)
    return clf


def eval_roc_auc(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    if len(clf.classes_) < 2:
        return float("nan")
    probs = clf.predict_proba(X)[:, 1]
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, probs)


@torch.no_grad()
def compute_original_features(
    model: RobertaForMaskedLM,
    dataloader: DataLoader,
    layer: int,
    device: torch.device,
):
    model.eval()
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
        cls_feat = attn[:, 0, :]
        feats.append(cls_feat.detach().cpu())
        labels.append(label)
        label_masks.append(label_mask)
    return (
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
        torch.cat(label_masks, dim=0).numpy(),
    )


def evaluate_baseline_frozen(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    layer: int,
    device: torch.device,
    log_path: Path | None = None,
    run_id: str | None = None,
    extra_fields: dict[str, Any] | None = None,
):
    for task in cfg.downstream_tasks:
        split_type = MOLNET_SPLITS.get(task, "scaffold")
        tasks, (train_df, _valid_df, test_df), _ = load_molnet_dataset(
            task,
            split=split_type,
            df_format="chemprop",
            local_only=cfg.local_only,
        )
        label_cols = list(tasks)

        train_dataset = SmilesClassificationDataset(
            train_df, tokenizer, label_cols, max_len=cfg.max_len
        )
        test_dataset = SmilesClassificationDataset(
            test_df, tokenizer, label_cols, max_len=cfg.max_len
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.downstream_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.downstream_batch_size, shuffle=False
        )

        X_train, y_train, y_train_mask = compute_original_features(
            model, train_loader, layer, device
        )
        X_test, y_test, y_test_mask = compute_original_features(
            model, test_loader, layer, device
        )

        if y_train.ndim > 1:
            y_train, y_train_mask = y_train[:, 0], y_train_mask[:, 0]
            y_test, y_test_mask = y_test[:, 0], y_test_mask[:, 0]

        mask_train = y_train_mask.astype(bool)
        mask_test = y_test_mask.astype(bool)

        if mask_train.sum() == 0 or mask_test.sum() == 0:
            print(f"[layer {layer}] {task} - No valid samples, skipping")
            continue

        roc_aucs = []
        for seed_idx in range(cfg.num_seeds):
            seed = cfg.seed + seed_idx
            np.random.seed(seed)
            random.seed(seed)

            clf = train_linear_probe(
                X_train[mask_train], y_train[mask_train], random_state=seed
            )
            roc = eval_roc_auc(clf, X_test[mask_test], y_test[mask_test])
            roc_aucs.append(roc)

        roc_mean = np.mean(roc_aucs)
        roc_std = np.std(roc_aucs)

        print(f"[BASELINE-FROZEN layer {layer}] {task} ROC-AUC={roc_mean:.4f}+-{roc_std:.4f}")

        row = {
            "run_id": run_id or cfg.run_id or f"{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": cfg.model_name,
            "layer": layer,
            "attn_source": "attn_output",
            "sae_apply_mode": "none",
            "sae_training_mode": "none",
            "backbone_frozen": True,
            "sae_type": "none",
            "n_latents": 0,
            "k": 0,
            "l1_weight": 0.0,
            "lr": 0.0,
            "batch_size": 0,
            "epochs": 0,
            "seed": cfg.seed,
            "num_seeds": cfg.num_seeds,
            "downstream_task": task,
            "downstream_method": "original_probe",
            "pooling": "cls",
            "probe_model": "linear",
            "roc_auc_mean": roc_mean,
            "roc_auc_std": roc_std,
            "split_type": split_type,
        }
        if extra_fields:
            row.update(extra_fields)
        append_csv_row(log_path or cfg.log_path, row)


def evaluate_downstream(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    ae: Autoencoder,
    tokenizer: RobertaTokenizerFast,
    layer: int,
    device: torch.device,
    log_path: Path | None = None,
    run_id: str | None = None,
    extra_fields: dict[str, Any] | None = None,
):
    sae_type = _resolve_sae_type(ae, extra_fields)

    for task in cfg.downstream_tasks:
        split_type = MOLNET_SPLITS.get(task, "scaffold")
        tasks, (train_df, _valid_df, test_df), _ = load_molnet_dataset(
            task,
            split=split_type,
            df_format="chemprop",
            local_only=cfg.local_only,
        )
        label_cols = list(tasks)

        train_dataset = SmilesClassificationDataset(
            train_df, tokenizer, label_cols, max_len=cfg.max_len
        )
        test_dataset = SmilesClassificationDataset(
            test_df, tokenizer, label_cols, max_len=cfg.max_len
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.downstream_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.downstream_batch_size, shuffle=False
        )

        X_train, y_train, y_train_mask = compute_latent_features(
            model, ae, train_loader, layer, device
        )
        X_test, y_test, y_test_mask = compute_latent_features(
            model, ae, test_loader, layer, device
        )

        if y_train.ndim > 1:
            y_train, y_train_mask = y_train[:, 0], y_train_mask[:, 0]
            y_test, y_test_mask = y_test[:, 0], y_test_mask[:, 0]

        mask_train = y_train_mask.astype(bool)
        mask_test = y_test_mask.astype(bool)

        if mask_train.sum() == 0 or mask_test.sum() == 0:
            print(f"[layer {layer}] {task} - No valid samples, skipping")
            continue

        roc_aucs = []
        for seed_idx in range(cfg.num_seeds):
            seed = cfg.seed + seed_idx
            np.random.seed(seed)
            random.seed(seed)

            clf = train_linear_probe(
                X_train[mask_train], y_train[mask_train], random_state=seed
            )
            roc = eval_roc_auc(clf, X_test[mask_test], y_test[mask_test])
            roc_aucs.append(roc)

        roc_mean = np.mean(roc_aucs)
        roc_std = np.std(roc_aucs)

        print(f"Using tasks {label_cols} from available tasks for {task}: {list(tasks)}")
        print(f"[layer {layer}] {task} ROC-AUC={roc_mean:.4f}+-{roc_std:.4f}")

        row = {
            "run_id": run_id or cfg.run_id or f"{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": cfg.model_name,
            "layer": layer,
            "attn_source": "attn_output",
            "sae_apply_mode": "model_mod",
            "sae_training_mode": "pretrain_only",
            "backbone_frozen": True,
            "sae_type": sae_type,
            "n_latents": cfg.n_latents,
            "k": cfg.topk,
            "l1_weight": cfg.l1_weight,
            "lr": cfg.sae_lr,
            "batch_size": cfg.sae_batch_size,
            "epochs": cfg.sae_epochs,
            "seed": cfg.seed,
            "num_seeds": cfg.num_seeds,
            "downstream_task": task,
            "downstream_method": "latent_probe",
            "pooling": "cls",
            "probe_model": "linear",
            "roc_auc_mean": roc_mean,
            "roc_auc_std": roc_std,
            "split_type": split_type,
        }
        if extra_fields:
            row.update(extra_fields)
        append_csv_row(log_path or cfg.log_path, row)
