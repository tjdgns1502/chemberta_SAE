#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_sae.analysis import run_feature_substructure_analysis
from chem_sae.config import SaeExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ground task-linked feature cards in chemical substructures"
    )
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run id")
    parser.add_argument(
        "--audit-reports-dir",
        type=Path,
        required=True,
        help="Path to an audit reports directory containing feature_cards/",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Optional comma-separated task filter",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Optional comma-separated feature id filter",
    )
    parser.add_argument(
        "--top-n-examples",
        type=int,
        default=10,
        help="Max combined top examples per feature card",
    )
    parser.add_argument(
        "--top-n-scaffolds",
        type=int,
        default=5,
        help="Max scaffolds to retain per feature",
    )
    parser.add_argument(
        "--mcs-threshold",
        type=float,
        default=0.6,
        help="RDKit MCS threshold over the selected top examples",
    )
    parser.add_argument(
        "--mcs-timeout-seconds",
        type=int,
        default=3,
        help="Per-feature RDKit MCS timeout in seconds",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.audit_reports_dir.exists():
        raise FileNotFoundError(f"audit reports directory not found: {args.audit_reports_dir}")
    if args.top_n_examples < 1:
        raise ValueError("--top-n-examples must be >= 1")
    if args.top_n_scaffolds < 1:
        raise ValueError("--top-n-scaffolds must be >= 1")
    if not (0.0 < args.mcs_threshold <= 1.0):
        raise ValueError("--mcs-threshold must be in (0, 1]")
    if args.mcs_timeout_seconds < 1:
        raise ValueError("--mcs-timeout-seconds must be >= 1")

    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    feature_ids = tuple(
        int(feature_id.strip())
        for feature_id in args.features.split(",")
        if feature_id.strip()
    )

    cfg = SaeExperimentConfig(run_id=args.run_id)
    cfg.ensure_dirs()

    print("[run feature-substructure] ready")
    print(f"run_id={cfg.run_id}")
    print(f"audit_reports_dir={args.audit_reports_dir}")
    print(f"tasks={tasks}")
    print(f"feature_ids={feature_ids}")
    print(f"top_n_examples={args.top_n_examples}")
    print(f"top_n_scaffolds={args.top_n_scaffolds}")
    print(f"mcs_threshold={args.mcs_threshold}")
    print(f"mcs_timeout_seconds={args.mcs_timeout_seconds}")
    print(f"run_root={cfg.run_context.run_root}")
    print(f"reports_dir={cfg.run_context.reports_dir}")

    if args.dry_run:
        print("[run feature-substructure] dry-run complete")
        return

    result = run_feature_substructure_analysis(
        audit_reports_dir=args.audit_reports_dir,
        output_dir=cfg.run_context.reports_dir,
        tasks=tasks or None,
        feature_ids=feature_ids or None,
        top_n_examples=args.top_n_examples,
        top_n_scaffolds=args.top_n_scaffolds,
        mcs_threshold=args.mcs_threshold,
        mcs_timeout_seconds=args.mcs_timeout_seconds,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
