from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, RobertaConfig, RobertaTokenizerFast


MOLNET_SPLITS = {
    "bace_classification": "scaffold",
    "bbbp": "scaffold",
    "clintox": "scaffold",
}


def _capstone_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here,) + tuple(here.parents):
        if (parent / "chemberta_repro_final").exists() and (parent / "sparse_autoencoder").exists():
            return parent
        if parent.name == "chemberta_repro_final":
            candidate = parent.parent
            if (candidate / "sparse_autoencoder").exists():
                return candidate
    return here.parents[2]


def _ensure_modules_on_path() -> None:
    repo_root = _capstone_root()
    sae_root = repo_root / "sparse_autoencoder"
    if sae_root.exists() and str(sae_root) not in os.sys.path:
        os.sys.path.insert(0, str(sae_root))
    chemberta_root = repo_root / "chemberta_repro_final" / "code" / "bert-loves-chemistry"
    if chemberta_root.exists() and str(chemberta_root) not in os.sys.path:
        os.sys.path.insert(0, str(chemberta_root))


_ensure_modules_on_path()

from sparse_autoencoder.loss import autoencoder_loss
from sparse_autoencoder.model import Autoencoder, TopK
from chemberta.utils.molnet_dataloader import load_molnet_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_activation(name: str):
    if name in ("gelu", "gelu_new", "gelu_fast"):
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(name)


# ============================================================================
# Model Components with SAE Intervention Support
# ============================================================================

class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + pos_embeds + type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        
        return context_layer, attention_probs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output, self_outputs[1]


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sae: Autoencoder | None = None,
        apply_intervention: bool = False,
        return_attn_output: bool = False,
    ):
        attn_output, _ = self.attention(hidden_states, attention_mask)
        
        # SAE Intervention: Replace attention output with reconstruction
        if apply_intervention and sae is not None:
            B, seq_len, d = attn_output.shape
            flat = attn_output.reshape(-1, d)
            _, latents, recons = sae(flat)
            attn_output = recons.reshape(B, seq_len, d)
        
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        
        if return_attn_output:
            return layer_output, attn_output
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sae_dict: dict[int, Autoencoder] | None = None,
        intervention_pattern: list[bool] | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: set[int] | None = None,
    ):
        attn_outputs = {} if return_attn_outputs else None
        target_layers = set(attn_output_layers or [])
        
        for i, layer_module in enumerate(self.layer):
            apply_intervention = intervention_pattern[i] if intervention_pattern else False
            sae = sae_dict.get(i) if sae_dict else None
            
            if return_attn_outputs and (i in target_layers):
                hidden_states, attn_output = layer_module(
                    hidden_states,
                    attention_mask,
                    sae=sae,
                    apply_intervention=apply_intervention,
                    return_attn_output=True,
                )
                attn_outputs[i] = attn_output
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask,
                    sae=sae,
                    apply_intervention=apply_intervention,
                )
        
        if return_attn_outputs:
            return hidden_states, attn_outputs
        return hidden_states


class RobertaModel(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        sae_dict: dict[int, Autoencoder] | None = None,
        intervention_pattern: list[bool] | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: set[int] | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        
        embeddings = self.embeddings(input_ids, token_type_ids)
        encoder_out = self.encoder(
            embeddings,
            attention_mask=extended_mask,
            sae_dict=sae_dict,
            intervention_pattern=intervention_pattern,
            return_attn_outputs=return_attn_outputs,
            attn_output_layers=attn_output_layers,
        )
        return encoder_out


class RobertaLMHead(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features: torch.Tensor):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class RobertaForMaskedLM(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.config = config

    def tie_weights(self):
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sae_dict: dict[int, Autoencoder] | None = None,
        intervention_pattern: list[bool] | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: set[int] | None = None,
    ):
        encoder_out = self.roberta(
            input_ids,
            attention_mask,
            sae_dict=sae_dict,
            intervention_pattern=intervention_pattern,
            return_attn_outputs=return_attn_outputs,
            attn_output_layers=attn_output_layers,
        )
        
        if return_attn_outputs:
            sequence_output, attn_outputs = encoder_out
        else:
            sequence_output = encoder_out
            attn_outputs = None
        
        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        if return_attn_outputs:
            return logits, loss, attn_outputs
        return logits, loss


# ============================================================================
# Utilities
# ============================================================================

def load_state_dict_from_hf(model_name: str, local_only: bool = True):
    snapshot_dir = snapshot_download(repo_id=model_name, local_files_only=local_only)
    safetensors_path = Path(snapshot_dir) / "model.safetensors"
    bin_path = Path(snapshot_dir) / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    return torch.load(str(bin_path), map_location="cpu")


def load_config_from_hf(model_name: str, local_only: bool = True):
    snapshot_dir = snapshot_download(repo_id=model_name, local_files_only=local_only)
    cfg_path = Path(snapshot_dir) / "config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    return RobertaConfig(**cfg_dict)


class MLMSmilesDataset(Dataset):
    def __init__(self, path: Path, tokenizer: RobertaTokenizerFast, max_len: int = 128):
        self.lines = [l.strip() for l in open(path) if l.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.lines[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


class SmilesClassificationDataset(Dataset):
    def __init__(self, df, tokenizer: RobertaTokenizerFast, label_cols: list[str], max_len: int = 128):
        self.smiles = df["smiles"].astype(str).tolist()
        labels = df[label_cols].to_numpy(dtype=np.float32)
        self.label_mask = ~np.isnan(labels)
        labels = np.nan_to_num(labels, nan=0.0)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "label_mask": torch.tensor(self.label_mask[idx], dtype=torch.float32),
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SaeInterventionConfig:
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    local_only: bool = True
    mlm_data_path: Path = (
        _capstone_root() / "chemberta_repro_final/code/bert-loves-chemistry/chemberta/data/100k_rndm_zinc_drugs_clean.txt"
    )
    max_len: int = 128
    mlm_batch_size: int = 8
    
    n_latents: int = 4096
    topk: int = 32
    sae_lr: float = 1e-4
    sae_batch_size: int = 2048
    sae_epochs: int = 20
    l1_weight: float = 0.0
    chunk_size: int = 20000
    val_fraction: float = 0.05
    early_stopping_patience: int = 5
    
    seed: int = 42
    num_seeds: int = 5
    layers: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    downstream_batch_size: int = 64
    downstream_tasks: tuple[str, ...] = ("bbbp", "bace_classification", "clintox")
    
    runs_dir: Path = _capstone_root() / "runs/sae_intervention"
    acts_dir: Path = _capstone_root() / "runs/sae_intervention/acts"
    ckpt_dir: Path = _capstone_root() / "runs/sae_intervention/checkpoints"
    log_path: Path = _capstone_root() / "runs/sae_intervention/intervention_results.csv"


# ============================================================================
# Core Functions
# ============================================================================

def prepare_mlm_loader(cfg: SaeInterventionConfig, tokenizer: RobertaTokenizerFast):
    dataset = MLMSmilesDataset(cfg.mlm_data_path, tokenizer, max_len=cfg.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    return DataLoader(dataset, batch_size=cfg.mlm_batch_size, shuffle=True, collate_fn=collator)


def build_mlm_model(cfg: SaeInterventionConfig, device: torch.device):
    config = load_config_from_hf(cfg.model_name, local_only=cfg.local_only)
    model = RobertaForMaskedLM(config)
    model.tie_weights()
    state_dict = load_state_dict_from_hf(cfg.model_name, local_only=cfg.local_only)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config


def save_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def generate_intervention_patterns():
    """Generate all 64 intervention patterns (2^6 combinations)"""
    patterns = list(itertools.product([False, True], repeat=6))
    return [list(p) for p in patterns]


def pattern_to_string(pattern: list[bool]) -> str:
    """Convert pattern [True, False, True, False, False, True] -> '101001'"""
    return ''.join(['1' if x else '0' for x in pattern])


def get_intervened_layers(pattern: list[bool]) -> str:
    """Get comma-separated list of intervened layer indices"""
    layers = [str(i) for i, intervene in enumerate(pattern) if intervene]
    return ','.join(layers) if layers else ''


@torch.no_grad()
def extract_activations_with_intervention(
    model: RobertaForMaskedLM,
    loader: DataLoader,
    target_layer: int,
    sae_dict: dict[int, Autoencoder],
    pattern: list[bool],
    device: torch.device,
):
    """Extract attention output activations for target_layer with intervention applied to previous layers"""
    model.eval()
    for sae in sae_dict.values():
        sae.eval()
    
    current_pattern = [pattern[i] if i < target_layer else False for i in range(6)]
    buffered = []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        _, attn_outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sae_dict=sae_dict,
            intervention_pattern=current_pattern,
            return_attn_outputs=True,
            attn_output_layers={target_layer},
        )
        
        attn = attn_outputs[target_layer]
        flat = attn[attention_mask.bool()].detach().cpu().to(torch.float16)
        buffered.append(flat)
    
    return torch.cat(buffered, dim=0)


def train_sae(
    activations: torch.Tensor,
    cfg: SaeInterventionConfig,
    device: torch.device,
) -> Autoencoder:
    """Train SAE on given activations"""
    d_model = activations.shape[-1]
    num_samples = activations.shape[0]
    
    train_size = int(num_samples * (1 - cfg.val_fraction))
    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_acts = activations[train_indices].to(torch.float32)
    val_acts = activations[val_indices].to(torch.float32)
    
    train_loader = DataLoader(train_acts, batch_size=cfg.sae_batch_size, shuffle=True)
    val_loader = DataLoader(val_acts, batch_size=cfg.sae_batch_size, shuffle=False)
    
    ae = Autoencoder(
        n_latents=cfg.n_latents,
        n_inputs=d_model,
        activation=TopK(cfg.topk),
        normalize=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(ae.parameters(), lr=cfg.sae_lr)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.sae_epochs):
        ae.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            _, latents, recons = ae(batch)
            loss = autoencoder_loss(recons, batch, latents, cfg.l1_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        
        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, latents, recons = ae(batch)
                loss = autoencoder_loss(recons, batch, latents, cfg.l1_weight)
                val_loss += loss.item()
        
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"      epoch {epoch+1}/{cfg.sae_epochs} train_loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f} (best={best_val_loss:.4f})")
        
        if patience_counter >= cfg.early_stopping_patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break
    
    return ae


def train_sae_with_intervention(
    cfg: SaeInterventionConfig,
    pattern: list[bool],
    pattern_id: int,
    device: torch.device,
) -> dict[int, Autoencoder]:
    """Train SAEs for all layers with given intervention pattern"""
    model, _ = build_mlm_model(cfg, device)
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.model_name, local_files_only=cfg.local_only)
    mlm_loader = prepare_mlm_loader(cfg, tokenizer)
    
    sae_dict = {}
    
    for layer in range(6):
        intervene_this_layer = pattern[layer]
        
        if intervene_this_layer:
            print(f"  Training SAE for layer {layer}...")
            activations = extract_activations_with_intervention(
                model, mlm_loader, layer, sae_dict, pattern, device
            )
            sae = train_sae(activations, cfg, device)
            sae_dict[layer] = sae
            
            save_path = cfg.ckpt_dir / f"pattern_{pattern_id}" / f"layer_{layer}.pt"
            save_checkpoint(save_path, {"model": sae.state_dict()})
        else:
            print(f"  Skipping SAE training for layer {layer} (no intervention)")
    
    return sae_dict


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    clf = LogisticRegression(max_iter=1000, random_state=random_state, solver='saga')
    clf.fit(X_train, y_train)
    return clf


def eval_roc_auc(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    probs = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, probs)


@torch.no_grad()
def extract_final_hidden_features(
    model: RobertaForMaskedLM,
    dataloader: DataLoader,
    sae_dict: dict[int, Autoencoder],
    pattern: list[bool],
    device: torch.device,
):
    """Extract final hidden state features with full intervention pattern applied"""
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
    """Evaluate downstream tasks with given intervention pattern"""
    model, _ = build_mlm_model(cfg, device)
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.model_name, local_only=cfg.local_only)
    
    results = []
    
    for task in cfg.downstream_tasks:
        split_type = MOLNET_SPLITS.get(task, "scaffold")
        tasks, (train_df, valid_df, test_df), _ = load_molnet_dataset(task, split=split_type, df_format="chemprop")
        label_cols = list(tasks)
        
        train_dataset = SmilesClassificationDataset(train_df, tokenizer, label_cols, max_len=cfg.max_len)
        test_dataset = SmilesClassificationDataset(test_df, tokenizer, label_cols, max_len=cfg.max_len)
        train_loader = DataLoader(train_dataset, batch_size=cfg.downstream_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.downstream_batch_size, shuffle=False)
        
        X_train, y_train, y_train_mask = extract_final_hidden_features(model, train_loader, sae_dict, pattern, device)
        X_test, y_test, y_test_mask = extract_final_hidden_features(model, test_loader, sae_dict, pattern, device)
        
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
            clf = train_linear_probe(X_train[mask_train], y_train[mask_train], random_state=seed)
            roc = eval_roc_auc(clf, X_test[mask_test], y_test[mask_test])
            roc_aucs.append(roc)
        
        roc_mean = np.mean(roc_aucs)
        roc_std = np.std(roc_aucs)
        
        print(f"  {task:20s}: ROC-AUC = {roc_mean:.4f} ± {roc_std:.4f}")
        
        results.append({
            "pattern_id": pattern_id,
            "pattern_binary": pattern_to_string(pattern),
            "num_intervened": sum(pattern),
            "intervened_layers": get_intervened_layers(pattern),
            "task": task,
            "split_type": split_type,
            "roc_auc_mean": roc_mean,
            "roc_auc_std": roc_std,
            "num_seeds": cfg.num_seeds,
        })
    
    return results


def run_intervention_experiment(pattern_ids: list[int], gpu_id: int):
    """Run intervention experiments for given pattern IDs on specific GPU"""
    device = torch.device(f"cuda:{gpu_id}")
    cfg = SaeInterventionConfig()
    patterns = generate_intervention_patterns()
    
    for pattern_id in pattern_ids:
        pattern = patterns[pattern_id]
        pattern_str = pattern_to_string(pattern)
        num_intervened = sum(pattern)
        intervened_layers = get_intervened_layers(pattern)
        
        print(f"\n{'='*70}")
        print(f"GPU {gpu_id}: Pattern {pattern_id}/63")
        print(f"Binary: {pattern_str}")
        print(f"Intervened layers: {num_intervened}/6 ({intervened_layers if intervened_layers else 'none'})")
        print(f"{'='*70}")
        
        set_seed(cfg.seed)
        
        print("Training SAEs...")
        sae_dict = train_sae_with_intervention(cfg, pattern, pattern_id, device)
        
        print("Evaluating downstream tasks...")
        results = evaluate_intervention(cfg, sae_dict, pattern, pattern_id, device)
        
        for row in results:
            row["gpu_id"] = gpu_id
            append_csv_row(cfg.log_path, row)
        
        print(f"Pattern {pattern_id} complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_ids", type=str, required=True, help="Comma-separated pattern IDs (e.g., '0,1,2')")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use (0-5)")
    args = parser.parse_args()
    
    pattern_ids = [int(x) for x in args.pattern_ids.split(',')]
    run_intervention_experiment(pattern_ids, args.gpu_id)
