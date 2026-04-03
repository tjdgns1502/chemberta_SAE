from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import MOLNET_SPLITS, SaeInterventionConfig
from chem_sae.data import SmilesClassificationDataset
from chem_sae.eval.downstream import eval_roc_auc, train_linear_probe
from chem_sae.modeling import RobertaForMaskedLM, build_mlm_model
from chem_sae.vendor import Autoencoder, load_molnet_dataset


@torch.no_grad()
def extract_final_hidden_features(
    model: RobertaForMaskedLM,
    dataloader: DataLoader,
    sae_dict: dict[int, Autoencoder],
    pattern: list[bool],
    device: torch.device,
):
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
        )

        cls_feat = final_hidden[:, 0, :]
        feats.append(cls_feat.cpu())
        labels.append(label)
        label_masks.append(label_mask)

    return (
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
        torch.cat(label_masks, dim=0).numpy(),
    )


def evaluate_intervention(
    cfg: SaeInterventionConfig,
    sae_dict: dict[int, Autoencoder],
    pattern: list[bool],
    pattern_id: int,
    device: torch.device,
) -> list[dict]:
    model, _ = build_mlm_model(cfg, device)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )

    results = []
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

        X_train, y_train, y_train_mask = extract_final_hidden_features(
            model, train_loader, sae_dict, pattern, device
        )
        X_test, y_test, y_test_mask = extract_final_hidden_features(
            model, test_loader, sae_dict, pattern, device
        )

        if y_train.ndim > 1:
            y_train, y_train_mask = y_train[:, 0], y_train_mask[:, 0]
            y_test, y_test_mask = y_test[:, 0], y_test_mask[:, 0]

        mask_train = y_train_mask.astype(bool)
        mask_test = y_test_mask.astype(bool)

        if mask_train.sum() == 0 or mask_test.sum() == 0:
            print(f"  {task}: No valid samples, skipping")
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

        print(f"  {task:20s}: ROC-AUC = {roc_mean:.4f} ± {roc_std:.4f}")

        results.append(
            {
                "pattern_id": pattern_id,
                "task": task,
                "split_type": split_type,
                "roc_auc_mean": roc_mean,
                "roc_auc_std": roc_std,
                "num_seeds": cfg.num_seeds,
            }
        )

    return results
