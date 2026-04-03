"""W&B logging utilities — SAELens-compatible metric naming.

Provides:
- ``WandbRunLogger``: thin lifecycle wrapper around ``wandb`` (init/log/finish).
- ``build_train_step_log_dict`` / ``build_sparsity_log_dict``:
  shared metric builders used by both ``sae_training`` and ``sae_training_probe``.
  Keys are **flat** (e.g. ``losses/overall_loss``, ``metrics/l0``) to match
  SAELens conventions so that standard W&B dashboards work out of the box.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import torch


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def to_wandb_config(value: Any) -> dict[str, Any]:
    cfg = _jsonable(value)
    if isinstance(cfg, dict):
        return cfg
    return {"value": cfg}


def _sanitize_artifact_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
    return cleaned or "artifact"


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def build_probe_wandb_metadata(
    *,
    run_id: str,
    layers: Iterable[int],
    base_l0: float,
    n_latents: int,
    schedule_mode: str,
    sparsity_loss_mode: str,
    run_name: str | None,
) -> dict[str, Any]:
    layer_values = tuple(int(layer) for layer in layers)
    if len(layer_values) == 1:
        layer_token = f"{layer_values[0]:02d}"
        layer_label = f"L{layer_token}"
        group = f"probe_layer_{layer_token}"
        layer_tags = [f"layer:{layer_token}"]
    else:
        joined = "_".join(f"{layer:02d}" for layer in layer_values)
        layer_label = f"L[{','.join(f'{layer:02d}' for layer in layer_values)}]"
        group = f"probe_layers_{joined}"
        layer_tags = [f"layers:{joined}"]

    prefix = f"{layer_label} | l0={base_l0:.4f} | n={n_latents} | "
    source_name = (run_name or run_id).strip()
    display_name = source_name if source_name.startswith(prefix) else f"{prefix}{source_name}"

    tags = _dedupe_preserve_order(
        [
            "probe",
            "jumprelu",
            *layer_tags,
            f"l0:{base_l0:.4f}",
            f"n_latents:{n_latents}",
            f"schedule:{schedule_mode}",
            f"sparsity:{sparsity_loss_mode}",
        ]
    )
    return {
        "display_name": display_name,
        "group": group,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Shared metric builders (SAELens-compatible flat keys)
# ---------------------------------------------------------------------------

def _unwrap_item(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


@torch.no_grad()
def build_train_step_log_dict(
    *,
    sae_in: torch.Tensor,
    sae_out: torch.Tensor,
    feature_acts: torch.Tensor,
    overall_loss: torch.Tensor,
    losses: dict[str, torch.Tensor | float | int],
    metrics: dict[str, torch.Tensor | float | int],
    current_learning_rate: float,
    n_training_samples: int,
    n_forward_passes_since_fired: torch.Tensor,
    dead_feature_window: int,
    coefficients: dict[str, float],
    global_step: int,
) -> dict[str, Any]:
    """Build per-step log dict with SAELens-compatible flat keys."""
    l0 = feature_acts.bool().float().sum(-1).to_dense().mean()
    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).reshape(-1)
    total_variance = (
        (sae_in - sae_in.mean(0)).pow(2).sum(dim=-1).reshape(-1).clamp(min=1e-8)
    )
    explained_variance_legacy = 1 - per_token_l2_loss / total_variance
    explained_variance = 1 - per_token_l2_loss.mean() / total_variance.mean()
    dead_features = (
        (n_forward_passes_since_fired > dead_feature_window).sum().item()
    )

    log_dict: dict[str, Any] = {
        "losses/overall_loss": float(overall_loss.item()),
        "metrics/explained_variance_legacy": explained_variance_legacy.mean().item(),
        "metrics/explained_variance_legacy_std": explained_variance_legacy.std().item(),
        "metrics/explained_variance": explained_variance.item(),
        "metrics/l0": l0.item(),
        "sparsity/mean_passes_since_fired": (
            n_forward_passes_since_fired.float().mean().item()
        ),
        "sparsity/dead_features": dead_features,
        "details/current_learning_rate": current_learning_rate,
        "details/n_training_samples": n_training_samples,
        "details/global_step": global_step,
    }
    for name, coefficient in coefficients.items():
        log_dict[f"details/{name}_coefficient"] = float(coefficient)
    for loss_name, loss_value in losses.items():
        log_dict[f"losses/{loss_name}"] = _unwrap_item(loss_value)
    for metric_name, metric_value in metrics.items():
        log_dict[f"metrics/{metric_name}"] = _unwrap_item(metric_value)
    return log_dict


@torch.no_grad()
def build_sparsity_log_dict(
    *,
    act_freq_scores: torch.Tensor,
    n_frac_active_samples: int,
) -> dict[str, Any]:
    """Build sparsity log dict with SAELens-compatible flat keys.

    Returns raw numpy arrays for histogram values — callers should wrap
    with ``wandb.Histogram()`` if W&B is available.
    """
    if n_frac_active_samples <= 0:
        return {}
    feature_sparsity = act_freq_scores / float(n_frac_active_samples)
    log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
    return {
        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
        "plots/feature_density_line_chart": log_feature_sparsity.numpy(),
        "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
        "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
    }


# ---------------------------------------------------------------------------
# Thin W&B lifecycle wrapper
# ---------------------------------------------------------------------------

class WandbRunLogger:
    """Minimal wrapper that delegates directly to ``wandb``.

    Keeps the ``enabled`` flag so callers can avoid ``if log_to_wandb:``
    everywhere, and preserves the resume-from-json convenience. All metric
    keys are **flat** (SAELens-compatible).
    """

    def __init__(
        self,
        *,
        logger_cfg: Any,
        run_root: Path,
        run_id: str,
        config_payload: dict[str, Any],
        job_type: str,
        tags: Iterable[str] | None = None,
        group: str | None = None,
    ) -> None:
        self.cfg = logger_cfg
        self.run_root = run_root
        self.run_id = run_id
        self.config_payload = config_payload
        self.job_type = job_type
        self.tags = list(tags or [])
        self.group = group or getattr(logger_cfg, "wandb_group", None)
        self.enabled = bool(getattr(logger_cfg, "log_to_wandb", False))
        self._wandb = None
        self._run = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import wandb
        except Exception as exc:
            raise RuntimeError(
                "W&B logging is enabled but `wandb` is not importable. "
                "Install wandb or disable with --disable-wandb."
            ) from exc

        # Auto-reuse previous wandb_id for resume continuity.
        run_meta_path = self.run_root / "reports" / "wandb_run.json"
        if getattr(self.cfg, "wandb_id", None) is None and run_meta_path.exists():
            try:
                prev = json.loads(run_meta_path.read_text(encoding="utf-8"))
                prev_id = prev.get("wandb_id")
                if isinstance(prev_id, str) and prev_id:
                    self.cfg.wandb_id = prev_id
            except Exception:
                pass

        self._wandb = wandb
        self._run = wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            config=self.config_payload,
            name=self.cfg.run_name or self.run_id,
            id=self.cfg.wandb_id,
            group=self.group,
            job_type=self.job_type,
            tags=self.tags,
            resume="allow",
        )
        if self._run is None:
            self.enabled = False
            return
        if getattr(self.cfg, "wandb_id", None) is None:
            self.cfg.wandb_id = self._run.id
        self._save_run_metadata()

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

    # -- logging -------------------------------------------------------------

    def should_log(self, step: int) -> bool:
        if not self.enabled:
            return False
        freq = max(1, int(getattr(self.cfg, "wandb_log_frequency", 10)))
        return (step + 1) % freq == 0  # SAELens convention: (step+1) % freq

    def should_eval_log(self, step: int) -> bool:
        if not self.enabled:
            return False
        freq = max(1, int(getattr(self.cfg, "wandb_log_frequency", 10)))
        eval_mult = max(1, int(getattr(self.cfg, "eval_every_n_wandb_logs", 100)))
        return (step + 1) % (freq * eval_mult) == 0

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(_jsonable(values), step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        for key, value in _jsonable(values).items():
            self._run.summary[key] = value

    def histogram(self, values: Any) -> Any:
        if not self.enabled or self._wandb is None:
            return values
        return self._wandb.Histogram(values)

    def log_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        files: Iterable[Path],
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._wandb is None:
            return
        file_list = [p for p in files if p.exists()]
        if not file_list:
            return
        artifact = self._wandb.Artifact(
            _sanitize_artifact_name(name),
            type=artifact_type,
            metadata=_jsonable(metadata or {}),
        )
        for p in file_list:
            artifact.add_file(str(p))
        self._wandb.log_artifact(artifact, aliases=aliases or None)

    @property
    def run_url(self) -> str | None:
        if self._run is None:
            return None
        return getattr(self._run, "url", None)

    # -- internal ------------------------------------------------------------

    def _save_run_metadata(self) -> None:
        reports_dir = self.run_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "wandb_id": getattr(self.cfg, "wandb_id", None),
            "wandb_project": getattr(self.cfg, "wandb_project", None),
            "wandb_entity": getattr(self.cfg, "wandb_entity", None),
            "wandb_group": self.group,
            "wandb_run_name": getattr(self.cfg, "run_name", None),
            "job_type": self.job_type,
            "tags": self.tags,
        }
        (reports_dir / "wandb_run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
