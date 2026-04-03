#!/usr/bin/env python3
"""Unified CLI entrypoint for chemberta_SAE experiments."""

import argparse
from pathlib import Path
import sys

import torch
from transformers import RobertaTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_sae.config import SaeExperimentConfig
from chem_sae.eval import evaluate_baseline_frozen, evaluate_final_hidden_state
from chem_sae.modeling import build_mlm_model
from chem_sae.train import (
    resolve_layers_from_model,
    run_all,
    run_intervention_experiment,
    run_sweep,
)
from chem_sae.utils import set_seed


def _build_sae_cfg(args: argparse.Namespace) -> SaeExperimentConfig:
    cfg = SaeExperimentConfig(run_id=args.run_id)
    cfg.layers_spec = args.layers
    cfg.ensure_dirs()
    return cfg


def _apply_wandb_args(cfg: SaeExperimentConfig, args: argparse.Namespace) -> None:
    if getattr(args, "log_to_wandb", False):
        cfg.logger.log_to_wandb = True
    if getattr(args, "disable_wandb", False):
        cfg.logger.log_to_wandb = False
    if getattr(args, "wandb_project", None) is not None:
        cfg.logger.wandb_project = args.wandb_project
    if getattr(args, "wandb_entity", None) is not None:
        cfg.logger.wandb_entity = args.wandb_entity
    if getattr(args, "wandb_id", None) is not None:
        cfg.logger.wandb_id = args.wandb_id
    if getattr(args, "wandb_run_name", None) is not None:
        cfg.logger.run_name = args.wandb_run_name
    if getattr(args, "wandb_log_frequency", None) is not None:
        if args.wandb_log_frequency < 1:
            raise ValueError("--wandb-log-frequency must be >= 1")
        cfg.logger.wandb_log_frequency = args.wandb_log_frequency
    if getattr(args, "eval_every_n_wandb_logs", None) is not None:
        if args.eval_every_n_wandb_logs < 1:
            raise ValueError("--eval-every-n-wandb-logs must be >= 1")
        cfg.logger.eval_every_n_wandb_logs = args.eval_every_n_wandb_logs
    if getattr(args, "disable_wandb_log_weights", False):
        cfg.logger.log_weights_to_wandb = False
    if getattr(args, "log_optimizer_state_to_wandb", False):
        cfg.logger.log_optimizer_state_to_wandb = True
    if getattr(args, "log_activations_store_to_wandb", False):
        cfg.logger.log_activations_store_to_wandb = True


def cmd_sae(args: argparse.Namespace) -> None:
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id so checkpoints are restored from a single run.")

    cfg = _build_sae_cfg(args)
    _apply_wandb_args(cfg, args)
    set_seed(cfg.seed)

    print("[run sae] ready")
    print(f"run_id={cfg.run_id}")
    print(f"mlm_data_path={cfg.mlm_data_path}")
    print(f"run_root={cfg.run_context.run_root}")
    print(f"acts_dir={cfg.acts_dir}")
    print(f"checkpoints_dir={cfg.ckpt_dir}")
    print(f"downstream_log_path={cfg.log_path}")
    print(f"quality_log_path={cfg.run_context.quality_log_path}")
    print(f"layers={cfg.layers_spec}")
    print(f"log_to_wandb={cfg.logger.log_to_wandb}")
    print(f"wandb_project={cfg.logger.wandb_project}")
    print(f"wandb_entity={cfg.logger.wandb_entity}")
    print(f"wandb_id={cfg.logger.wandb_id}")
    print(f"wandb_run_name={cfg.logger.run_name}")
    print(f"wandb_log_frequency={cfg.logger.wandb_log_frequency}")
    print(f"eval_every_n_wandb_logs={cfg.logger.eval_every_n_wandb_logs}")
    print(f"log_weights_to_wandb={cfg.logger.log_weights_to_wandb}")
    print(f"log_optimizer_state_to_wandb={cfg.logger.log_optimizer_state_to_wandb}")
    print(f"log_activations_store_to_wandb={cfg.logger.log_activations_store_to_wandb}")
    if args.dry_run:
        print("[run sae] dry-run complete")
        return

    if args.sweep:
        run_sweep(cfg, arch_mode=args.arch, resume=args.resume)
    else:
        run_all(cfg, resume=args.resume, arch=args.arch)


def cmd_baseline(args: argparse.Namespace) -> None:
    cfg = _build_sae_cfg(args)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[run baseline] ready")
    print(f"device={device}")
    print(f"run_id={cfg.run_id}")
    print(f"model_name={cfg.model_name}")
    print(f"layers={cfg.layers_spec}")
    print(f"log_path={cfg.log_path}")
    if args.dry_run:
        print("[run baseline] dry-run complete")
        return

    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)
    layers = resolve_layers_from_model(cfg, model)

    print(f"\n{'=' * 60}")
    print("BASELINE EVALUATION (Frozen backbone)")
    print(f"{'=' * 60}")
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        evaluate_baseline_frozen(
            cfg,
            model,
            tokenizer,
            layer,
            device,
            log_path=cfg.run_context.downstream_log_path,
            run_id=cfg.run_id,
            extra_fields={"arch": "baseline", "candidate": "manual"},
        )

    print("\nBaseline evaluation complete.")


def cmd_final_hidden(args: argparse.Namespace) -> None:
    cfg = _build_sae_cfg(args)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[run final-hidden] ready")
    print(f"run_id={cfg.run_id}")
    print(f"device={device}")
    print(f"model_name={cfg.model_name}")
    print(f"log_path={cfg.log_path}")
    if args.dry_run:
        print("[run final-hidden] dry-run complete")
        return

    tokenizer = RobertaTokenizerFast.from_pretrained(
        cfg.model_name, local_files_only=cfg.local_only
    )
    model, _ = build_mlm_model(cfg, device)

    print(f"\n{'=' * 60}")
    print("FINAL HIDDEN STATE EVALUATION")
    print(f"{'=' * 60}\n")
    evaluate_final_hidden_state(
        cfg,
        model,
        tokenizer,
        device,
        log_path=cfg.run_context.downstream_log_path,
        run_id=cfg.run_id,
        extra_fields={"arch": "final_hidden", "candidate": "manual"},
    )
    print("\nFinal hidden state evaluation complete.")


def cmd_intervention(args: argparse.Namespace) -> None:
    pattern_ids = [int(x) for x in args.pattern_ids.split(",") if x.strip()]
    if not pattern_ids:
        raise ValueError("No valid pattern_ids provided.")

    print("[run intervention] ready")
    print(f"gpu_id={args.gpu_id}")
    print(f"pattern_ids={pattern_ids}")
    if args.dry_run:
        print("[run intervention] dry-run complete")
        return

    run_intervention_experiment(pattern_ids, args.gpu_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="chemberta_SAE unified runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sae_parser = subparsers.add_parser("sae", help="Run SAE training pipeline")
    sae_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run_id. If omitted, one is generated automatically.",
    )
    sae_parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run staged sweep (8 -> 20 -> 25 epochs) with BatchTopK/JumpReLU search.",
    )
    sae_parser.add_argument(
        "--arch",
        choices=("batchtopk", "jumprelu", "both"),
        default="both",
        help="Architecture selection for run/sweep.",
    )
    sae_parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer selection: 'all' or comma-separated layer ids.",
    )
    sae_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training. Requires --run-id.",
    )
    sae_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config and exit without starting training.",
    )
    sae_parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Enable W&B logging (SAELens-compatible logger fields).",
    )
    sae_parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    sae_parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name.",
    )
    sae_parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user).",
    )
    sae_parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="W&B run id for resume/continuation.",
    )
    sae_parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B display run name.",
    )
    sae_parser.add_argument(
        "--wandb-log-frequency",
        type=int,
        default=None,
        help="W&B train-step logging frequency.",
    )
    sae_parser.add_argument(
        "--eval-every-n-wandb-logs",
        type=int,
        default=None,
        help="W&B eval logging frequency multiplier.",
    )
    sae_parser.add_argument(
        "--disable-wandb-log-weights",
        action="store_true",
        help="Do not upload weight artifacts to W&B.",
    )
    sae_parser.add_argument(
        "--log-optimizer-state-to-wandb",
        action="store_true",
        help="Upload optimizer-containing checkpoints as W&B artifacts.",
    )
    sae_parser.add_argument(
        "--log-activations-store-to-wandb",
        action="store_true",
        help="Upload activation cache metadata as W&B artifacts.",
    )
    sae_parser.set_defaults(func=cmd_sae)

    baseline_parser = subparsers.add_parser(
        "baseline", help="Run baseline frozen-backbone evaluation"
    )
    baseline_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id. If omitted, a new one is generated.",
    )
    baseline_parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer selection: 'all' or comma-separated layer ids.",
    )
    baseline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config and exit.",
    )
    baseline_parser.set_defaults(func=cmd_baseline)

    final_hidden_parser = subparsers.add_parser(
        "final-hidden", help="Run final hidden state evaluation"
    )
    final_hidden_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id. If omitted, a new one is generated.",
    )
    final_hidden_parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer selection metadata saved in run context.",
    )
    final_hidden_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config and exit.",
    )
    final_hidden_parser.set_defaults(func=cmd_final_hidden)

    intervention_parser = subparsers.add_parser(
        "intervention", help="Run SAE intervention experiment"
    )
    intervention_parser.add_argument(
        "--pattern_ids",
        type=str,
        default="0",
        help="Comma-separated pattern IDs (e.g. '0,1,2').",
    )
    intervention_parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id to use. Falls back to CPU if CUDA is unavailable.",
    )
    intervention_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse inputs and print them.",
    )
    intervention_parser.set_defaults(func=cmd_intervention)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
