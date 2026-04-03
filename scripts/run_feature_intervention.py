#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_sae.config import SaeExperimentConfig
from chem_sae.eval import parse_feature_indices, run_feature_intervention
from chem_sae.utils import set_seed


def _parse_feature_values(text: str | None) -> float | list[float] | None:
    if text is None:
        return None
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run feature-level SAE intervention")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run id")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="Target transformer layer")
    parser.add_argument("--task", type=str, required=True, help="Downstream task")
    parser.add_argument("--features", type=str, required=True, help="Comma-separated feature ids")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("zero", "mean_clamp", "force_on"),
        required=True,
        help="Feature intervention mode",
    )
    parser.add_argument(
        "--feature-values",
        type=str,
        default=None,
        help="Optional scalar or comma-separated values for mean_clamp/force_on",
    )
    parser.add_argument(
        "--control",
        type=str,
        choices=("none", "matched_random"),
        default="none",
        help="Optional control condition",
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=None,
        help="Seed for matched-random control sampling",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow HF model/dataset downloads instead of local-only mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.layer < 0:
        raise ValueError("--layer must be >= 0")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    feature_indices = parse_feature_indices(args.features)
    feature_values = _parse_feature_values(args.feature_values)

    cfg = SaeExperimentConfig(run_id=args.run_id)
    cfg.layers_spec = str(args.layer)
    cfg.local_only = not args.allow_downloads
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    print("[run feature-intervention] ready")
    print(f"run_id={cfg.run_id}")
    print(f"checkpoint_path={args.checkpoint}")
    print(f"layer={args.layer}")
    print(f"task={args.task}")
    print(f"feature_indices={feature_indices}")
    print(f"mode={args.mode}")
    print(f"feature_values={feature_values}")
    print(f"control={args.control}")
    print(f"control_seed={args.control_seed}")
    print(f"local_only={cfg.local_only}")
    print(f"run_root={cfg.run_context.run_root}")
    print(f"reports_dir={cfg.run_context.reports_dir}")

    if args.dry_run:
        print("[run feature-intervention] dry-run complete")
        return

    result = run_feature_intervention(
        cfg,
        checkpoint_path=args.checkpoint,
        layer=args.layer,
        task=args.task,
        feature_indices=feature_indices,
        mode=args.mode,
        control_kind=args.control,
        control_seed=args.control_seed,
        explicit_feature_values=feature_values,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
