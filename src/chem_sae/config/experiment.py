from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Literal

from .paths import (
    LEGACY_CHEMBERTA_ROOT,
    LOGS_ROOT,
    PROJECT_ROOT,
    SAE_RUN_ROOT,
)


MOLNET_SPLITS = {
    "bace_classification": "scaffold",
    "bbbp": "scaffold",
    "clintox": "scaffold",
}

BATCHTOPK_K_SWEEP_VALUES = (8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96)
# Backward-compatible alias kept for older imports/log naming.
TOPK_SWEEP_VALUES = BATCHTOPK_K_SWEEP_VALUES
JUMPRELU_L0_SWEEP_VALUES = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
)
ARCH_CHOICES = ("batchtopk", "jumprelu")


def _default_mlm_data_path() -> Path:
    candidates = [
        # Primary path for the new project.
        PROJECT_ROOT / "data" / "100k_rndm_zinc_drugs_clean.txt",
        # Temporary fallback during migration.
        LEGACY_CHEMBERTA_ROOT
        / "code"
        / "bert-loves-chemistry"
        / "chemberta"
        / "data"
        / "100k_rndm_zinc_drugs_clean.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _short_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lower()
        )
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_short_git_sha()}"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_root: Path
    acts_dir: Path
    ckpt_dir: Path
    sweep_dir: Path
    final_dir: Path
    reports_dir: Path
    plots_dir: Path
    downstream_log_path: Path
    quality_log_path: Path
    registry_path: Path
    top_candidates_path: Path
    run_meta_path: Path


@dataclass(frozen=True)
class TrialConfig:
    arch: Literal["batchtopk", "jumprelu"]
    stage: Literal["stage1", "stage2", "stage3"]
    trial_id: int
    epoch_budget: int
    seed: int
    k: int | None = None
    l0_coefficient: float | None = None


@dataclass(frozen=True)
class TrialResult:
    run_id: str
    arch: Literal["batchtopk", "jumprelu"]
    stage: Literal["stage1", "stage2", "stage3"]
    trial_id: int
    k: int | None
    l0_coefficient: float | None
    epochs: int
    seed: int
    nmse_mean: float
    nmse_std: float
    dead_ratio: float
    mean_l0: float
    max_node_share: float
    active_cosine_mean: float
    decoder_cosine_max: float
    global_step: int
    status: Literal["ok", "failed"]
    trial_root: Path


@dataclass
class LoggingConfig:
    # SAELens-compatible W&B logger contract.
    log_to_wandb: bool = False
    log_activations_store_to_wandb: bool = False
    log_optimizer_state_to_wandb: bool = False
    log_weights_to_wandb: bool = True
    wandb_project: str = "sae_lens_training"
    wandb_id: str | None = None
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100


@dataclass
class SaeExperimentConfig:
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    local_only: bool = True
    mlm_data_path: Path = field(default_factory=_default_mlm_data_path)
    max_len: int = 128
    mlm_batch_size: int = 8

    n_latents: int = 4096
    batchtopk_k: int = 32
    topk: int = 32
    sae_lr: float = 1e-4
    sae_batch_size: int = 2048
    sae_epochs: int = 20
    l1_weight: float = 0.0
    jumprelu_l0_coefficient: float = 1.0
    jumprelu_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"] = "tanh"
    jumprelu_tanh_scale: float = 4.0
    jumprelu_l0_warm_up_steps: int = 0
    jumprelu_dead_feature_window: int = 1000
    jumprelu_pre_act_loss_coefficient: float | None = None
    probe_dashboard_enabled: bool = True
    probe_dashboard_every_n_epochs: int = 3
    chunk_size: int = 20000
    val_fraction: float = 0.05
    early_stopping_patience: int = 5
    seed: int = 42
    num_seeds: int = 5

    layers_spec: str = "all"
    layers: tuple[int, ...] = ()
    downstream_batch_size: int = 64
    downstream_tasks: tuple[str, ...] = ("bbbp", "bace_classification", "clintox")
    stage1_epochs: int = 8
    stage2_epochs: int = 20
    stage3_epochs: int = 25
    batchtopk_k_sweep_values: tuple[int, ...] = BATCHTOPK_K_SWEEP_VALUES
    # Backward-compatible alias; sweeps should use batchtopk_k_sweep_values.
    topk_sweep_values: tuple[int, ...] = TOPK_SWEEP_VALUES
    jumprelu_l0_sweep_values: tuple[float, ...] = JUMPRELU_L0_SWEEP_VALUES
    dead_feature_ratio_max: float = 0.35
    node_concentration_max: float = 0.20
    activation_similarity_max: float = 0.35
    decoder_redundancy_max: float = 0.92
    quality_metric_feature_cap: int = 64
    quality_metric_sample_cap: int = 4096

    logger: LoggingConfig = field(default_factory=LoggingConfig)

    run_id: str | None = None
    runs_dir: Path = field(default_factory=lambda: SAE_RUN_ROOT)
    acts_dir: Path = field(default_factory=lambda: SAE_RUN_ROOT / "acts")
    ckpt_dir: Path = field(default_factory=lambda: SAE_RUN_ROOT / "checkpoints")
    log_path: Path = field(default_factory=lambda: SAE_RUN_ROOT / "downstream_records.csv")
    logs_dir: Path = field(default_factory=lambda: LOGS_ROOT / "sae")
    run_context: RunContext | None = field(default=None, init=False, repr=False)

    def resolve_layers(self, num_hidden_layers: int) -> tuple[int, ...]:
        if self.layers_spec.strip().lower() == "all":
            self.layers = tuple(range(num_hidden_layers))
            return self.layers

        parsed = tuple(
            int(x.strip())
            for x in self.layers_spec.split(",")
            if x.strip()
        )
        if not parsed:
            raise ValueError("layers_spec must be 'all' or a comma-separated list of layer ids")
        invalid = [layer for layer in parsed if layer < 0 or layer >= num_hidden_layers]
        if invalid:
            raise ValueError(
                f"Invalid layer ids {invalid}; model has {num_hidden_layers} hidden layers"
            )
        self.layers = parsed
        return self.layers

    def build_run_context(self) -> RunContext:
        run_id = self.run_id or generate_run_id()
        run_root = self.runs_dir / run_id
        return RunContext(
            run_id=run_id,
            run_root=run_root,
            acts_dir=run_root / "acts",
            ckpt_dir=run_root / "checkpoints",
            sweep_dir=run_root / "sweep",
            final_dir=run_root / "final",
            reports_dir=run_root / "reports",
            plots_dir=run_root / "plots",
            downstream_log_path=run_root / "reports" / "downstream_records.csv",
            quality_log_path=run_root / "reports" / "sae_quality_summary.csv",
            registry_path=run_root / "sweep" / "registry.csv",
            top_candidates_path=run_root / "sweep" / "top_candidates.csv",
            run_meta_path=run_root / "run_meta.json",
        )

    def ensure_dirs(self) -> None:
        if self.run_context is None:
            self.run_context = self.build_run_context()
        self.run_id = self.run_context.run_id
        self.topk = self.batchtopk_k

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_context.run_root.mkdir(parents=True, exist_ok=True)
        self.run_context.acts_dir.mkdir(parents=True, exist_ok=True)
        self.run_context.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_context.sweep_dir.mkdir(parents=True, exist_ok=True)
        self.run_context.final_dir.mkdir(parents=True, exist_ok=True)
        self.run_context.reports_dir.mkdir(parents=True, exist_ok=True)
        self.run_context.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Keep legacy attributes synchronized so older helpers still work.
        self.acts_dir = self.run_context.acts_dir
        self.ckpt_dir = self.run_context.ckpt_dir
        self.log_path = self.run_context.downstream_log_path
