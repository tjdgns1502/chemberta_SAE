from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaTokenizerFast,
)

# MoleculeNet benchmark standard split settings (from ChemBERTa paper)
MOLNET_SPLITS = {
    "bace_classification": "scaffold",
    "bbbp": "scaffold",
    "clintox": "scaffold",
}


def _capstone_root() -> Path:
    """Find capstone project root directory."""
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
    """Add required modules to Python path."""
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


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pad_token_id = config.pad_token_id

    def create_position_ids_from_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.pad_token_id).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + self.pad_token_id

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = self.create_position_ids_from_input_ids(input_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        return context_layer, attn_probs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
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
        self_outputs = self.self(hidden_states, attention_mask=attention_mask)
        attn_output = self.output(self_outputs[0], hidden_states)
        return attn_output, self_outputs[1]


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
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
        return_attn_output: bool = False,
    ):
        attn_output, _ = self.attention(hidden_states, attention_mask=attention_mask)
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        if return_attn_output:
            return layer_output, attn_output
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attn_outputs: bool = False,
        attn_output_layers: Iterable[int] | None = None,
    ):
        attn_outputs = {} if return_attn_outputs else None
        target_layers = set(attn_output_layers or [])

        for i, layer_module in enumerate(self.layer):
            if return_attn_outputs and (i in target_layers):
                hidden_states, attn_output = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_attn_output=True,
                )
                attn_outputs[i] = attn_output
            else:
                hidden_states = layer_module(hidden_states, attention_mask=attention_mask)

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
        return_attn_outputs: bool = False,
        attn_output_layers: Iterable[int] | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        embeddings = self.embeddings(input_ids, token_type_ids=token_type_ids)
        encoder_out = self.encoder(
            embeddings,
            attention_mask=extended_mask,
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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
        return_attn_outputs: bool = False,
        attn_output_layers: Iterable[int] | None = None,
    ):
        encoder_out = self.roberta(
            input_ids,
            attention_mask=attention_mask,
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


def load_state_dict_from_hf(model_name: str, local_only: bool = True):
    snapshot_dir = snapshot_download(repo_id=model_name, local_files_only=local_only)
    safetensors_path = Path(snapshot_dir) / "model.safetensors"
    bin_path = Path(snapshot_dir) / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    return torch.load(str(bin_path), map_location="cpu")


def load_config_from_hf(model_name: str, local_only: bool = True) -> RobertaConfig:
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

    def __len__(self) -> int:
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

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        item["label_mask"] = torch.tensor(self.label_mask[idx], dtype=torch.float32)
        return item


class ActivationChunkDataset(IterableDataset):
    def __init__(
        self,
        chunk_paths: list[Path],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.chunk_paths = chunk_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        paths = list(self.chunk_paths)
        if self.shuffle:
            rng.shuffle(paths)
        for path in paths:
            acts = torch.load(path, map_location="cpu")
            if self.shuffle:
                idx = torch.randperm(acts.shape[0])
                acts = acts[idx]
            for i in range(0, acts.shape[0], self.batch_size):
                yield acts[i : i + self.batch_size]


@dataclass
class SaeExperimentConfig:
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    local_only: bool = True
    mlm_data_path: Path = (
        _capstone_root()
        / "chemberta_repro_final/code/bert-loves-chemistry/chemberta/data/100k_rndm_zinc_drugs_clean.txt"
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
    early_stopping_patience: int = 5  # Stop if val loss doesn't improve for 5 epochs
    seed: int = 42
    num_seeds: int = 5  # ChemBERTa paper uses 5 seeds (0-4) for mean±std

    layers: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    downstream_batch_size: int = 64  # ChemBERTa paper uses 64

    runs_dir: Path = _capstone_root() / "runs/sae"
    acts_dir: Path = _capstone_root() / "runs/sae/acts"
    ckpt_dir: Path = _capstone_root() / "runs/sae/checkpoints"
    log_path: Path = _capstone_root() / "runs/sae/experiments.csv"

    downstream_tasks: tuple[str, ...] = ("bbbp", "bace_classification", "clintox")


def prepare_mlm_loader(cfg: SaeExperimentConfig, tokenizer: RobertaTokenizerFast):
    dataset = MLMSmilesDataset(cfg.mlm_data_path, tokenizer, max_len=cfg.max_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    return DataLoader(dataset, batch_size=cfg.mlm_batch_size, shuffle=True, collate_fn=collator)


def build_mlm_model(cfg: SaeExperimentConfig, device: torch.device):
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


def _write_chunk(layer_dir: Path, chunk_idx: int, tensor: torch.Tensor) -> Path:
    layer_dir.mkdir(parents=True, exist_ok=True)
    path = layer_dir / f"chunk_{chunk_idx:05d}.pt"
    torch.save(tensor, path)
    return path


@torch.no_grad()
def extract_attn_activations(
    cfg: SaeExperimentConfig, model: RobertaForMaskedLM, loader: DataLoader, device: torch.device
) -> None:
    model.eval()
    for layer in cfg.layers:
        layer_dir = cfg.acts_dir / f"layer_{layer}"
        chunk_idx = 0
        buffered = []
        buffered_tokens = 0
        total_tokens = 0
        chunk_paths = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, attn_outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attn_outputs=True,
                attn_output_layers={layer},
            )
            attn = attn_outputs[layer]
            flat = attn[attention_mask.bool()].detach().cpu().to(torch.float16)
            buffered.append(flat)
            buffered_tokens += flat.shape[0]
            total_tokens += flat.shape[0]

            if buffered_tokens >= cfg.chunk_size:
                chunk = torch.cat(buffered, dim=0)
                chunk_paths.append(_write_chunk(layer_dir, chunk_idx, chunk))
                chunk_idx += 1
                buffered = []
                buffered_tokens = 0

        if buffered:
            chunk = torch.cat(buffered, dim=0)
            chunk_paths.append(_write_chunk(layer_dir, chunk_idx, chunk))

        meta = {
            "layer": layer,
            "d_model": attn.shape[-1],
            "num_tokens": total_tokens,
            "num_chunks": len(chunk_paths),
            "chunk_size": cfg.chunk_size,
            "dtype": "float16",
            "model_name": cfg.model_name,
            "mlm_data_path": str(cfg.mlm_data_path),
        }
        save_meta(layer_dir / "meta.json", meta)


def _list_chunks(layer_dir: Path) -> list[Path]:
    return sorted(layer_dir.glob("chunk_*.pt"))


def _latest_checkpoint(path: Path) -> Path | None:
    if not path.exists():
        return None
    latest = path / "latest.pt"
    if latest.exists():
        return latest
    ckpts = sorted(path.glob("checkpoint_step_*.pt"))
    return ckpts[-1] if ckpts else None


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_sae_for_layer(
    cfg: SaeExperimentConfig, layer: int, device: torch.device, resume: bool = True
):
    layer_dir = cfg.acts_dir / f"layer_{layer}"
    chunk_paths = _list_chunks(layer_dir)
    if not chunk_paths:
        raise FileNotFoundError(f"No activation chunks found in {layer_dir}")

    train_cut = max(1, int(len(chunk_paths) * (1 - cfg.val_fraction)))
    train_paths = chunk_paths[:train_cut]
    val_paths = chunk_paths[train_cut:] or train_paths[-1:]

    train_data = ActivationChunkDataset(
        train_paths, batch_size=cfg.sae_batch_size, shuffle=True, seed=cfg.seed
    )
    val_data = ActivationChunkDataset(
        val_paths, batch_size=cfg.sae_batch_size, shuffle=False, seed=cfg.seed
    )

    d_model = torch.load(chunk_paths[0], map_location="cpu").shape[1]
    ae = Autoencoder(
        n_latents=cfg.n_latents,
        n_inputs=d_model,
        activation=TopK(cfg.topk),
        normalize=True,
    ).to(device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=cfg.sae_lr)

    start_epoch = 0
    global_step = 0
    ckpt_dir = cfg.ckpt_dir / f"layer_{layer}"
    
    # Early stopping and loss tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    if resume:
        latest = _latest_checkpoint(ckpt_dir)
        if latest is not None:
            state = torch.load(latest, map_location=device)
            ae.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            global_step = state["step"]
            best_val_loss = state.get("best_val_loss", float('inf'))
            train_losses = state.get("train_losses", [])
            val_losses = state.get("val_losses", [])

    for epoch in range(start_epoch, cfg.sae_epochs):
        ae.train()
        train_loss = 0.0
        num_batches = 0
        for batch in train_data:
            batch = batch.to(device).float()
            _, latents, recons = ae(batch)
            loss = autoencoder_loss(recons, batch, latents, cfg.l1_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            num_batches += 1
            global_step += 1
            if global_step % 1000 == 0:
                _save_checkpoint(
                    ckpt_dir / f"checkpoint_step_{global_step}.pt",
                    {
                        "model": ae.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                )
                _save_checkpoint(
                    ckpt_dir / "latest.pt",
                    {
                        "model": ae.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                )

        ae.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_data:
                batch = batch.to(device).float()
                _, latents, recons = ae(batch)
                loss = autoencoder_loss(recons, batch, latents, cfg.l1_weight)
                val_loss += loss.item()
                val_batches += 1

        train_loss_avg = train_loss / max(1, num_batches)
        val_loss_avg = val_loss / max(1, val_batches)
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # Early stopping check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # Save best model
            _save_checkpoint(
                ckpt_dir / "best.pt",
                {
                    "model": ae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
            )
        else:
            patience_counter += 1
        
        _save_checkpoint(
            ckpt_dir / f"checkpoint_step_{global_step}.pt",
            {
                "model": ae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
        )
        _save_checkpoint(
            ckpt_dir / "latest.pt",
            {
                "model": ae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
        )

        print(
            f"[layer {layer}] epoch {epoch+1}/{cfg.sae_epochs} "
            f"train_loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f} "
            f"(best={best_val_loss:.4f}, patience={patience_counter}/{cfg.early_stopping_patience})"
        )
        
        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(f"[layer {layer}] Early stopping triggered at epoch {epoch+1}")
            # Load best model
            best_state = torch.load(ckpt_dir / "best.pt", map_location=device)
            ae.load_state_dict(best_state["model"])
            break
    
    # Save loss plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'SAE Training - Layer {layer}')
    plt.legend()
    plt.grid(True)
    plt.savefig(ckpt_dir / f"layer_{layer}_loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return ae


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
        # Use CLS token (first token) instead of mean pooling
        cls_latent = latents[:, 0, :]
        feats.append(cls_latent.cpu())
        labels.append(label)
        label_masks.append(label_mask)
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy(), torch.cat(label_masks, dim=0).numpy()


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Train linear probe with stochastic solver for variance across seeds."""
    # Use 'saga' solver for stochastic optimization (allows random_state to affect results)
    clf = LogisticRegression(max_iter=1000, random_state=random_state, solver='saga')
    clf.fit(X_train, y_train)
    return clf


def eval_roc_auc(clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    probs = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, probs)


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compute_original_features(
    model: RobertaForMaskedLM,
    dataloader: DataLoader,
    layer: int,
    device: torch.device,
):
    """Extract original attention output features (no SAE) for baseline comparison."""
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
        # Use CLS token (ChemBERTa standard)
        cls_feat = attn[:, 0, :]
        feats.append(cls_feat.detach().cpu())
        labels.append(label)
        label_masks.append(label_mask)
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy(), torch.cat(label_masks, dim=0).numpy()


def evaluate_baseline_frozen(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    layer: int,
    device: torch.device,
):
    """Baseline: Frozen backbone + original features + linear probe."""
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

        X_train, y_train, y_train_mask = compute_original_features(
            model, train_loader, layer, device
        )
        X_test, y_test, y_test_mask = compute_original_features(
            model, test_loader, layer, device
        )

        # Take first task only (single-label classification)
        if y_train.ndim > 1:
            y_train, y_train_mask = y_train[:, 0], y_train_mask[:, 0]
            y_test, y_test_mask = y_test[:, 0], y_test_mask[:, 0]

        # Filter out NaN labels
        mask_train = y_train_mask.astype(bool)
        mask_test = y_test_mask.astype(bool)
        
        if mask_train.sum() == 0 or mask_test.sum() == 0:
            print(f"[layer {layer}] {task} - No valid samples, skipping")
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
        
        print(f"[BASELINE-FROZEN layer {layer}] {task} ROC-AUC={roc_mean:.4f}±{roc_std:.4f}")

        row = {
            "run_id": f"{int(time.time())}",
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
        append_csv_row(cfg.log_path, row)


def evaluate_downstream(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    ae: Autoencoder,
    tokenizer: RobertaTokenizerFast,
    layer: int,
    device: torch.device,
):
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

        X_train, y_train, y_train_mask = compute_latent_features(
            model, ae, train_loader, layer, device
        )
        X_test, y_test, y_test_mask = compute_latent_features(
            model, ae, test_loader, layer, device
        )

        # Take first task only (single-label classification)
        if y_train.ndim > 1:
            y_train, y_train_mask = y_train[:, 0], y_train_mask[:, 0]
            y_test, y_test_mask = y_test[:, 0], y_test_mask[:, 0]

        # Filter out NaN labels
        mask_train = y_train_mask.astype(bool)
        mask_test = y_test_mask.astype(bool)
        
        if mask_train.sum() == 0 or mask_test.sum() == 0:
            print(f"[layer {layer}] {task} - No valid samples, skipping")
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
        
        print(f"Using tasks {label_cols} from available tasks for {task}: {list(tasks)}")
        print(f"[layer {layer}] {task} ROC-AUC={roc_mean:.4f}±{roc_std:.4f}")

        row = {
            "run_id": f"{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": cfg.model_name,
            "layer": layer,
            "attn_source": "attn_output",
            "sae_apply_mode": "model_mod",
            "sae_training_mode": "pretrain_only",
            "backbone_frozen": True,
            "sae_type": "TopK",
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
            "pooling": "cls",  # CLS token (ChemBERTa standard)
            "probe_model": "linear",
            "roc_auc_mean": roc_mean,
            "roc_auc_std": roc_std,
            "split_type": split_type,
        }
        append_csv_row(cfg.log_path, row)


def run_all(cfg: SaeExperimentConfig, resume: bool = True):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)

    mlm_loader = prepare_mlm_loader(cfg, tokenizer)
    extract_attn_activations(cfg, model, mlm_loader, device)

    # Train SAE for all layers
    for layer in cfg.layers:
        print(f"\n{'='*60}")
        print(f"SAE TRAINING & EVALUATION - Layer {layer}")
        print(f"{'='*60}")
        ae = train_sae_for_layer(cfg, layer, device, resume=resume)
        evaluate_downstream(cfg, model, ae, tokenizer, layer, device)
    
    # Baseline evaluation (once at the end for all layers)
    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION (Frozen backbone)")
    print(f"{'='*60}")
    for layer in cfg.layers:
        print(f"\n--- Layer {layer} ---")
        evaluate_baseline_frozen(cfg, model, tokenizer, layer, device)


if __name__ == "__main__":
    cfg = SaeExperimentConfig()
    run_all(cfg, resume=True)
