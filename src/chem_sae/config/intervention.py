from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .experiment import _default_mlm_data_path
from .paths import INTERVENTION_RUN_ROOT, LOGS_ROOT


@dataclass
class SaeInterventionConfig:
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    local_only: bool = True
    mlm_data_path: Path = field(default_factory=_default_mlm_data_path)
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

    runs_dir: Path = field(default_factory=lambda: INTERVENTION_RUN_ROOT)
    acts_dir: Path = field(default_factory=lambda: INTERVENTION_RUN_ROOT / "acts")
    ckpt_dir: Path = field(default_factory=lambda: INTERVENTION_RUN_ROOT / "checkpoints")
    log_path: Path = field(
        default_factory=lambda: INTERVENTION_RUN_ROOT / "intervention_results.csv"
    )
    logs_dir: Path = field(default_factory=lambda: LOGS_ROOT / "sae_intervention")

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.acts_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
