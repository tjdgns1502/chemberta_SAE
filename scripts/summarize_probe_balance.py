#!/usr/bin/env python3
"""Summarize probe runs by layer with a balance-oriented score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS_ROOT = Path("/home/yoo122333/capstone/chemberta_SAE/artifacts/runs/sae")


def _parse_launch_metadata(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("wandb:"):
            break
        if "=" not in raw or raw.startswith("["):
            continue
        key, value = raw.split("=", 1)
        meta[key] = value
    return meta


def _infer_layer(run_id: str, meta: dict[str, str]) -> int | None:
    layers = meta.get("layers", "")
    if layers.startswith("[") and layers.endswith("]"):
        payload = [item.strip() for item in layers.strip("[]").split(",") if item.strip()]
        if len(payload) == 1:
            try:
                return int(payload[0])
            except ValueError:
                return None
    if "_L" in run_id and "_l0_" in run_id:
        try:
            return int(run_id.split("_L", 1)[1].split("_", 1)[0])
        except ValueError:
            return None
    if "layer" in run_id:
        suffix = run_id.split("layer", 1)[1]
        digits = []
        for char in suffix:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            return int("".join(digits))
    return None


def _balance_score(
    *,
    active_ratio: float,
    nmse: float,
    dead_ratio: float,
    target_min: float,
    target_max: float,
    dead_max: float,
) -> float:
    if active_ratio < target_min:
        active_penalty = target_min - active_ratio
    elif active_ratio > target_max:
        active_penalty = active_ratio - target_max
    else:
        active_penalty = 0.0
    dead_penalty = max(0.0, dead_ratio - dead_max) * 10.0
    return nmse + active_penalty + dead_penalty


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(RUNS_ROOT.glob("*/probe/metrics.json")):
        run_id = metrics_path.parents[1].name
        launch_path = RUNS_ROOT / f"{run_id}.probe.launch.log"
        if not launch_path.exists():
            continue
        meta = _parse_launch_metadata(launch_path)
        layer = _infer_layer(run_id, meta)
        if layer is None:
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate", {})
        n_latents = int(meta.get("n_latents", "0") or 0)
        mean_l0 = float(aggregate.get("mean_l0", 0.0))
        rows.append(
            {
                "run_id": run_id,
                "layer": layer,
                "n_latents": n_latents,
                "base_l0": float(meta.get("base_l0", "nan")),
                "schedule": meta.get("schedule", ""),
                "sparsity_loss_mode": meta.get(
                    "sparsity_loss_mode",
                    "tanh" if "tanh" in run_id.lower() else "",
                ),
                "nmse": float(aggregate.get("nmse_mean", 0.0)),
                "dead_ratio": float(aggregate.get("dead_ratio_max", 0.0)),
                "mean_l0": mean_l0,
                "active_ratio": mean_l0 / n_latents if n_latents else float("nan"),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize probe balance by layer")
    parser.add_argument("--sparsity-loss-mode", default="tanh")
    parser.add_argument("--n-latents", type=int, default=1536)
    parser.add_argument("--target-active-min", type=float, default=0.05)
    parser.add_argument("--target-active-max", type=float, default=0.12)
    parser.add_argument("--dead-max", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [
        row
        for row in _load_rows()
        if row["sparsity_loss_mode"] == args.sparsity_loss_mode
        and row["n_latents"] == args.n_latents
    ]
    for row in rows:
        row["balance_score"] = _balance_score(
            active_ratio=row["active_ratio"],
            nmse=row["nmse"],
            dead_ratio=row["dead_ratio"],
            target_min=args.target_active_min,
            target_max=args.target_active_max,
            dead_max=args.dead_max,
        )

    layers = sorted({row["layer"] for row in rows})
    print(
        f"# target active_ratio={args.target_active_min:.1%}..{args.target_active_max:.1%}, "
        f"dead_ratio<={args.dead_max:.1%}, n_latents={args.n_latents}, "
        f"sparsity={args.sparsity_loss_mode}"
    )
    for layer in layers:
        layer_rows = [row for row in rows if row["layer"] == layer]
        layer_rows.sort(key=lambda row: (row["balance_score"], row["nmse"], row["dead_ratio"]))
        print(f"\n[layer {layer}]")
        for row in layer_rows[: args.top_k]:
            print(
                f"{row['run_id']}: base_l0={row['base_l0']:.4f} "
                f"active={row['active_ratio']:.2%} dead={row['dead_ratio']:.2%} "
                f"nmse={row['nmse']:.6f} score={row['balance_score']:.6f}"
            )
            print(f"metrics={row['metrics_path']}")


if __name__ == "__main__":
    main()
