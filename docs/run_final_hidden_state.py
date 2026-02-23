"""Evaluate final hidden state (after all layers) for baseline comparison."""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "code", "bert-loves-chemistry"))
sys.path.insert(0, project_root)

from sae_experiment import (
    SaeExperimentConfig,
    build_mlm_model,
    set_seed,
    SmilesClassificationDataset,
    MOLNET_SPLITS,
    train_linear_probe,
    eval_roc_auc,
    append_csv_row,
)
import torch
import numpy as np
import random
import time
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
from chemberta.utils.molnet_dataloader import load_molnet_dataset
from pathlib import Path


def compute_final_hidden_features(model, dataloader, device):
    """Extract final hidden state (after all layers) using CLS token."""
    model.eval()
    feats = []
    labels = []
    label_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["labels"].cpu()
            label_mask = batch["label_mask"].cpu()

            # Get final hidden state (after all layers)
            final_hidden = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attn_outputs=False,
            )
            
            # Use CLS token (ChemBERTa standard)
            cls_feat = final_hidden[:, 0, :]
            feats.append(cls_feat.detach().cpu())
            labels.append(label)
            label_masks.append(label_mask)
    
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy(), torch.cat(label_masks, dim=0).numpy()


def evaluate_final_hidden_state(cfg, model, tokenizer, device):
    """Evaluate final hidden state for all downstream tasks."""
    for task in cfg.downstream_tasks:
        split_type = MOLNET_SPLITS.get(task, "scaffold")
        tasks, (train_df, valid_df, test_df), _ = load_molnet_dataset(
            task, split=split_type, df_format="chemprop"
        )
        label_cols = list(tasks)
        
        train_dataset = SmilesClassificationDataset(
            train_df, tokenizer, label_cols, max_len=cfg.max_len
        )
        test_dataset = SmilesClassificationDataset(
            test_df, tokenizer, label_cols, max_len=cfg.max_len
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.downstream_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.downstream_batch_size, shuffle=False)

        X_train, y_train, y_train_mask = compute_final_hidden_features(
            model, train_loader, device
        )
        X_test, y_test, y_test_mask = compute_final_hidden_features(
            model, test_loader, device
        )

        # Take first task only (single-label classification)
        if y_train.ndim > 1:
            y_train, y_train_mask = y_train[:, 0], y_train_mask[:, 0]
            y_test, y_test_mask = y_test[:, 0], y_test_mask[:, 0]

        # Filter out NaN labels
        mask_train = y_train_mask.astype(bool)
        mask_test = y_test_mask.astype(bool)
        
        if mask_train.sum() == 0 or mask_test.sum() == 0:
            print(f"[FINAL-HIDDEN] {task} - No valid samples, skipping")
            continue
        
        # Run with multiple seeds (different solver initialization)
        roc_aucs = []
        for seed_idx in range(cfg.num_seeds):
            seed = cfg.seed + seed_idx
            np.random.seed(seed)
            random.seed(seed)
            
            clf = train_linear_probe(X_train[mask_train], y_train[mask_train], random_state=seed)
            roc = eval_roc_auc(clf, X_test[mask_test], y_test[mask_test])
            roc_aucs.append(roc)
        
        roc_mean = np.mean(roc_aucs)
        roc_std = np.std(roc_aucs)
        
        print(f"[FINAL-HIDDEN] {task} ROC-AUC={roc_mean:.4f}±{roc_std:.4f}")

        row = {
            "run_id": f"{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": cfg.model_name,
            "layer": "final",
            "attn_source": "final_hidden",
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
            "downstream_method": "final_hidden_probe",
            "pooling": "cls",
            "probe_model": "linear",
            "roc_auc_mean": roc_mean,
            "roc_auc_std": roc_std,
            "split_type": split_type,
        }
        append_csv_row(cfg.runs_dir / "experiments.csv", row)


if __name__ == "__main__":
    cfg = SaeExperimentConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)
    
    print(f"\n{'='*60}")
    print(f"FINAL HIDDEN STATE EVALUATION")
    print(f"{'='*60}\n")
    
    evaluate_final_hidden_state(cfg, model, tokenizer, device)
    
    print("\n✅ Final hidden state evaluation complete!")
