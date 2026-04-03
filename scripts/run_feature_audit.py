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
from chem_sae.eval import run_feature_audit
from chem_sae.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run feature audit for a fixed SAE checkpoint")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run id")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="Target transformer layer")
    parser.add_argument(
        "--tasks",
        type=str,
        default="bbbp,bace_classification",
        help="Comma-separated downstream tasks",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top features per sign bucket")
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
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    if not tasks:
        raise ValueError("at least one task must be provided via --tasks")

    cfg = SaeExperimentConfig(run_id=args.run_id)
    cfg.layers_spec = str(args.layer)
    cfg.local_only = not args.allow_downloads
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    print("[run feature-audit] ready")
    print(f"run_id={cfg.run_id}")
    print(f"checkpoint_path={args.checkpoint}")
    print(f"layer={args.layer}")
    print(f"tasks={tasks}")
    print(f"top_k={args.top_k}")
    print(f"local_only={cfg.local_only}")
    print(f"run_root={cfg.run_context.run_root}")
    print(f"reports_dir={cfg.run_context.reports_dir}")

    if args.dry_run:
        print("[run feature-audit] dry-run complete")
        return

    result = run_feature_audit(
        cfg,
        checkpoint_path=args.checkpoint,
        layer=args.layer,
        tasks=tasks,
        top_k=args.top_k,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
