#!/usr/bin/env python3
"""Probe runner for JumpReLU L0 schedule validation."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_sae.config import SaeExperimentConfig
from chem_sae.train.sae_training_probe import run_probe
from chem_sae.utils import build_probe_wandb_metadata


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _parse_layers(spec: str) -> tuple[int, ...]:
    values = [x.strip() for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("--layers must contain at least one layer id")
    parsed = tuple(int(x) for x in values)
    if any(layer < 0 for layer in parsed):
        raise ValueError("--layers must use non-negative integers")
    return parsed


def _validate_args(args: argparse.Namespace) -> None:
    if args.base_l0 < 0:
        raise ValueError("--base-l0 must be >= 0")
    if args.decay_ratio < 0:
        raise ValueError("--decay-ratio must be >= 0")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be >= 1")
    if args.n_latents is not None and args.n_latents <= 0:
        raise ValueError("--n-latents must be >= 1")
    if args.l0_warmup_steps is not None and args.l0_warmup_steps < 0:
        raise ValueError("--l0-warmup-steps must be >= 0")
    if args.dead_feature_window is not None and args.dead_feature_window < 1:
        raise ValueError("--dead-feature-window must be >= 1")
    if args.dashboard_every_epochs is not None and args.dashboard_every_epochs < 1:
        raise ValueError("--dashboard-every-epochs must be >= 1")
    if args.wandb_log_frequency is not None and args.wandb_log_frequency < 1:
        raise ValueError("--wandb-log-frequency must be >= 1")
    if args.eval_every_n_wandb_logs is not None and args.eval_every_n_wandb_logs < 1:
        raise ValueError("--eval-every-n-wandb-logs must be >= 1")


def _print_header(
    *,
    cfg: SaeExperimentConfig,
    launch_log_path: Path,
    layers: tuple[int, ...],
    schedule_mode: Literal["none", "two_step", "exp"],
    base_l0: float,
    warmup_epochs: int,
    decay_ratio: float,
    epochs: int,
    n_latents: int,
    disable_early_stopping: bool,
    resume: bool,
    l0_warmup_steps: int,
    dead_feature_window: int,
    dashboard_enabled: bool,
    dashboard_every_epochs: int,
    log_to_wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_id: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_resolved_run_name: str | None,
    wandb_log_frequency: int,
    eval_every_n_wandb_logs: int,
    log_weights_to_wandb: bool,
    log_optimizer_state_to_wandb: bool,
    log_activations_store_to_wandb: bool,
    sparsity_loss_mode: str,
) -> None:
    print("[run sae-probe] ready")
    print(f"run_id={cfg.run_id}")
    print(f"run_root={cfg.run_context.run_root if cfg.run_context else '<unset>'}")
    print(f"launch_log_path={launch_log_path}")
    print(f"layers={list(layers)}")
    print(f"schedule={schedule_mode}")
    print(f"base_l0={base_l0}")
    print(f"warmup_epochs={warmup_epochs}")
    print(f"decay_ratio={decay_ratio}")
    print(f"epochs={epochs}")
    print(f"n_latents={n_latents}")
    print(f"early_stopping_patience={cfg.early_stopping_patience}")
    print(f"early_stopping_enabled={not disable_early_stopping}")
    print(f"resume={resume}")
    print(f"l0_warmup_steps={l0_warmup_steps}")
    print(f"dead_feature_window={dead_feature_window}")
    print(f"dashboard_enabled={dashboard_enabled}")
    print(f"dashboard_every_epochs={dashboard_every_epochs}")
    print(f"log_to_wandb={log_to_wandb}")
    print(f"wandb_project={wandb_project}")
    print(f"wandb_entity={wandb_entity}")
    print(f"wandb_id={wandb_id}")
    print(f"wandb_group={wandb_group}")
    print(f"wandb_run_name={wandb_run_name}")
    if wandb_resolved_run_name != wandb_run_name:
        print(f"wandb_resolved_run_name={wandb_resolved_run_name}")
    print(f"wandb_log_frequency={wandb_log_frequency}")
    print(f"eval_every_n_wandb_logs={eval_every_n_wandb_logs}")
    print(f"log_weights_to_wandb={log_weights_to_wandb}")
    print(f"log_optimizer_state_to_wandb={log_optimizer_state_to_wandb}")
    print(f"log_activations_store_to_wandb={log_activations_store_to_wandb}")
    print(f"sparsity_loss_mode={sparsity_loss_mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JumpReLU probe runner")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run id. If omitted, one is auto-generated.",
    )
    parser.add_argument(
        "--base-l0",
        type=float,
        default=0.001,
        help="Base l0 coefficient for JumpReLU probe.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=("none", "two_step", "exp"),
        default="two_step",
        help="L0 schedule mode.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Warmup epochs to keep base L0 before decay.",
    )
    parser.add_argument(
        "--decay-ratio",
        type=float,
        default=0.1,
        help="Decay ratio applied after warmup in two_step/exp modes.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0",
        help="Comma-separated layer ids. Default is layer 0 only.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs for probe run.",
    )
    parser.add_argument(
        "--n-latents",
        type=int,
        default=None,
        help="Override SAE latent width for probe only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved configuration and exit.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping for this probe run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume probe from existing run checkpoints. Requires --run-id.",
    )
    parser.add_argument(
        "--l0-warmup-steps",
        type=int,
        default=None,
        help="Optional JumpReLU L0 coefficient warm-up steps (batch-level).",
    )
    parser.add_argument(
        "--dead-feature-window",
        type=int,
        default=None,
        help="Optional dead feature window in steps for pre-activation loss mask.",
    )
    parser.add_argument(
        "--dashboard-every-epochs",
        type=int,
        default=None,
        help="Write/update probe HTML dashboard every N epochs.",
    )
    parser.add_argument(
        "--disable-dashboard",
        action="store_true",
        help="Disable probe HTML dashboard generation.",
    )
    parser.add_argument(
        "--sparsity-loss-mode",
        type=str,
        choices=("step", "tanh"),
        default=None,
        help="JumpReLU sparsity loss mode: 'step' (DeepMind L0) or 'tanh' (Anthropic).",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Enable W&B logging (SAELens-compatible logger fields).",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user).",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="W&B run id for resume/continuation.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B display run name.",
    )
    parser.add_argument(
        "--wandb-log-frequency",
        type=int,
        default=None,
        help="W&B train-step logging frequency.",
    )
    parser.add_argument(
        "--eval-every-n-wandb-logs",
        type=int,
        default=None,
        help="W&B eval logging frequency multiplier.",
    )
    parser.add_argument(
        "--disable-wandb-log-weights",
        action="store_true",
        help="Do not upload weight artifacts to W&B.",
    )
    parser.add_argument(
        "--log-optimizer-state-to-wandb",
        action="store_true",
        help="Upload optimizer-containing checkpoints as W&B artifacts.",
    )
    parser.add_argument(
        "--log-activations-store-to-wandb",
        action="store_true",
        help="Upload activation cache metadata as W&B artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id so probe checkpoints map to one run.")

    cfg = SaeExperimentConfig(run_id=args.run_id)
    if args.n_latents is not None:
        cfg.n_latents = args.n_latents
    if args.l0_warmup_steps is not None:
        cfg.jumprelu_l0_warm_up_steps = args.l0_warmup_steps
    if args.dead_feature_window is not None:
        cfg.jumprelu_dead_feature_window = args.dead_feature_window
    if args.sparsity_loss_mode is not None:
        cfg.jumprelu_sparsity_loss_mode = args.sparsity_loss_mode
    if args.dashboard_every_epochs is not None:
        cfg.probe_dashboard_every_n_epochs = args.dashboard_every_epochs
    if args.disable_dashboard:
        cfg.probe_dashboard_enabled = False
    if args.log_to_wandb:
        cfg.logger.log_to_wandb = True
    if args.disable_wandb:
        cfg.logger.log_to_wandb = False
    if args.wandb_project is not None:
        cfg.logger.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        cfg.logger.wandb_entity = args.wandb_entity
    if args.wandb_id is not None:
        cfg.logger.wandb_id = args.wandb_id
    if args.wandb_run_name is not None:
        cfg.logger.run_name = args.wandb_run_name
    requested_wandb_run_name = cfg.logger.run_name
    if args.wandb_log_frequency is not None:
        cfg.logger.wandb_log_frequency = args.wandb_log_frequency
    if args.eval_every_n_wandb_logs is not None:
        cfg.logger.eval_every_n_wandb_logs = args.eval_every_n_wandb_logs
    if args.disable_wandb_log_weights:
        cfg.logger.log_weights_to_wandb = False
    if args.log_optimizer_state_to_wandb:
        cfg.logger.log_optimizer_state_to_wandb = True
    if args.log_activations_store_to_wandb:
        cfg.logger.log_activations_store_to_wandb = True
    if args.disable_early_stopping:
        cfg.early_stopping_patience = args.epochs + 1
    cfg.layers_spec = args.layers
    cfg.ensure_dirs()

    layers = _parse_layers(args.layers)
    wandb_meta = build_probe_wandb_metadata(
        run_id=cfg.run_id or "unknown_run",
        layers=layers,
        base_l0=args.base_l0,
        n_latents=cfg.n_latents,
        schedule_mode=args.schedule,
        sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
        run_name=cfg.logger.run_name,
    )
    cfg.logger.run_name = wandb_meta["display_name"]
    cfg.logger.wandb_group = wandb_meta["group"]
    launch_log_path = cfg.runs_dir / f"{cfg.run_id}.probe.launch.log"

    _print_header(
        cfg=cfg,
        launch_log_path=launch_log_path,
        layers=layers,
        schedule_mode=args.schedule,
        base_l0=args.base_l0,
        warmup_epochs=args.warmup_epochs,
        decay_ratio=args.decay_ratio,
        epochs=args.epochs,
        n_latents=cfg.n_latents,
        disable_early_stopping=args.disable_early_stopping,
        resume=args.resume,
        l0_warmup_steps=cfg.jumprelu_l0_warm_up_steps,
        dead_feature_window=cfg.jumprelu_dead_feature_window,
        dashboard_enabled=cfg.probe_dashboard_enabled,
        dashboard_every_epochs=cfg.probe_dashboard_every_n_epochs,
        log_to_wandb=cfg.logger.log_to_wandb,
        wandb_project=cfg.logger.wandb_project,
        wandb_entity=cfg.logger.wandb_entity,
        wandb_id=cfg.logger.wandb_id,
        wandb_group=cfg.logger.wandb_group,
        wandb_run_name=requested_wandb_run_name,
        wandb_resolved_run_name=cfg.logger.run_name,
        wandb_log_frequency=cfg.logger.wandb_log_frequency,
        eval_every_n_wandb_logs=cfg.logger.eval_every_n_wandb_logs,
        log_weights_to_wandb=cfg.logger.log_weights_to_wandb,
        log_optimizer_state_to_wandb=cfg.logger.log_optimizer_state_to_wandb,
        log_activations_store_to_wandb=cfg.logger.log_activations_store_to_wandb,
        sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
    )

    if args.dry_run:
        print("[run sae-probe] dry-run complete")
        return

    with open(launch_log_path, "w", encoding="utf-8") as log_file:
        out_tee = _Tee(sys.stdout, log_file)
        err_tee = _Tee(sys.stderr, log_file)
        with redirect_stdout(out_tee), redirect_stderr(err_tee):
            _print_header(
                cfg=cfg,
                launch_log_path=launch_log_path,
                layers=layers,
                schedule_mode=args.schedule,
                base_l0=args.base_l0,
                warmup_epochs=args.warmup_epochs,
                decay_ratio=args.decay_ratio,
                epochs=args.epochs,
                n_latents=cfg.n_latents,
                disable_early_stopping=args.disable_early_stopping,
                resume=args.resume,
                l0_warmup_steps=cfg.jumprelu_l0_warm_up_steps,
                dead_feature_window=cfg.jumprelu_dead_feature_window,
                dashboard_enabled=cfg.probe_dashboard_enabled,
                dashboard_every_epochs=cfg.probe_dashboard_every_n_epochs,
                log_to_wandb=cfg.logger.log_to_wandb,
                wandb_project=cfg.logger.wandb_project,
                wandb_entity=cfg.logger.wandb_entity,
                wandb_id=cfg.logger.wandb_id,
                wandb_group=cfg.logger.wandb_group,
                wandb_run_name=requested_wandb_run_name,
                wandb_resolved_run_name=cfg.logger.run_name,
                wandb_log_frequency=cfg.logger.wandb_log_frequency,
                eval_every_n_wandb_logs=cfg.logger.eval_every_n_wandb_logs,
                log_weights_to_wandb=cfg.logger.log_weights_to_wandb,
                log_optimizer_state_to_wandb=cfg.logger.log_optimizer_state_to_wandb,
                log_activations_store_to_wandb=cfg.logger.log_activations_store_to_wandb,
                sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
            )
            result = run_probe(
                cfg=cfg,
                layers=layers,
                epochs=args.epochs,
                base_l0=args.base_l0,
                schedule_mode=args.schedule,
                warmup_epochs=args.warmup_epochs,
                decay_ratio=args.decay_ratio,
                resume=args.resume,
            )
            print("[run sae-probe] done")
            print(f"schedule_trace_path={result['schedule_trace_path']}")
            print(f"metrics_path={result['metrics_path']}")
            if "dashboard_index_path" in result:
                print(f"dashboard_index_path={result['dashboard_index_path']}")
            if "dashboard_live_index_path" in result:
                print(f"dashboard_live_index_path={result['dashboard_live_index_path']}")
            if "wandb_run_url" in result:
                print(f"wandb_run_url={result['wandb_run_url']}")


if __name__ == "__main__":
    main()
