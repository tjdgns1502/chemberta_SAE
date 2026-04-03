from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import RobertaTokenizerFast

from chem_sae.config import ARCH_CHOICES, SaeExperimentConfig, TrialConfig, TrialResult
from chem_sae.modeling import build_mlm_model
from chem_sae.train.sae_training import (
    prepare_activation_cache,
    resolve_layers_from_model,
    run_architecture_trial,
    run_reference_evaluations,
    trial_result_to_dict,
)
from chem_sae.utils import WandbRunLogger, append_csv_row, save_json, set_seed, to_wandb_config


def _trial_param_label_from_cfg(cfg: TrialConfig) -> str:
    if cfg.arch in {"batchtopk", "topk"}:
        return f"k{cfg.k}"
    return f"l0_{cfg.l0_coefficient}"


def _trial_root(cfg: SaeExperimentConfig, arch: str, trial_id: int, param_label: str) -> Path:
    assert cfg.run_context is not None
    return cfg.run_context.sweep_dir / arch / f"trial_{trial_id:04d}_{param_label}"


def _build_stage1_trials(cfg: SaeExperimentConfig, arch: str) -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    if arch in {"batchtopk", "topk"}:
        for idx, k in enumerate(cfg.batchtopk_k_sweep_values, start=1):
            trials.append(
                TrialConfig(
                    arch=arch,
                    stage="stage1",
                    trial_id=idx,
                    epoch_budget=cfg.stage1_epochs,
                    seed=cfg.seed + idx,
                    k=k,
                )
            )
    elif arch == "jumprelu":
        for idx, l0 in enumerate(cfg.jumprelu_l0_sweep_values, start=1):
            trials.append(
                TrialConfig(
                    arch="jumprelu",
                    stage="stage1",
                    trial_id=idx,
                    epoch_budget=cfg.stage1_epochs,
                    seed=cfg.seed + idx,
                    l0_coefficient=l0,
                )
            )
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")
    return trials


def _compute_l0_cv(layer_rows: list[dict[str, Any]]) -> float:
    values = [float(row["mean_l0"]) for row in layer_rows]
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    if m <= 0:
        return float("inf")
    return pstdev(values) / m


def _score_for_ranking(result: TrialResult, mean_l0_cv: float) -> float:
    # Primary metric is NMSE; structural penalties are added for stable ranking fallback.
    return (
        result.nmse_mean
        + 0.10 * result.nmse_std
        + 0.20 * result.max_node_share
        + 0.10 * result.active_cosine_mean
        + 0.10 * result.decoder_cosine_max
        + 0.05 * mean_l0_cv
        + 1e-8 * result.global_step
    )


def _append_registry(cfg: SaeExperimentConfig, result: TrialResult) -> None:
    assert cfg.run_context is not None
    row = {
        "run_id": result.run_id,
        "arch": result.arch,
        "stage": result.stage,
        "trial_id": result.trial_id,
        "k": result.k if result.k is not None else "",
        "l0_coefficient": result.l0_coefficient if result.l0_coefficient is not None else "",
        "epochs": result.epochs,
        "seed": result.seed,
        "nmse_mean": result.nmse_mean,
        "nmse_std": result.nmse_std,
        "dead_ratio": result.dead_ratio,
        "mean_l0": result.mean_l0,
        "max_node_share": result.max_node_share,
        "active_cosine_mean": result.active_cosine_mean,
        "decoder_cosine_max": result.decoder_cosine_max,
        "status": result.status,
    }
    append_csv_row(cfg.run_context.registry_path, row)


def _plot_stage1_curve(cfg: SaeExperimentConfig, arch: str, scored: list[dict[str, Any]]) -> None:
    if not scored:
        return
    assert cfg.run_context is not None
    if arch in {"batchtopk", "topk"}:
        x = [float(item["result"].k) for item in scored]
        x_label = "k"
        plot_name = "batchtopk_k_curve.png"
    else:
        x = [float(item["result"].l0_coefficient) for item in scored]
        x_label = "l0_coefficient"
        plot_name = "jumprelu_l0_curve.png"
    y = [float(item["result"].nmse_mean) for item in scored]

    order = np.argsort(x)
    x_sorted = [x[i] for i in order]
    y_sorted = [y[i] for i in order]

    plt.figure(figsize=(8, 5))
    plt.plot(x_sorted, y_sorted, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("val_recon_nmse_mean_over_layers")
    plt.title(f"{arch.upper()} stage1 sweep")
    plt.grid(True)
    plt.savefig(cfg.run_context.plots_dir / plot_name, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_dead_ratio_heatmap(
    cfg: SaeExperimentConfig,
    stage3_layer_rows: dict[str, list[dict[str, Any]]],
) -> None:
    if not stage3_layer_rows:
        return
    assert cfg.run_context is not None
    arches = sorted(stage3_layer_rows.keys())
    if not arches:
        return
    layers = list(cfg.layers)
    matrix = np.zeros((len(arches), len(layers)), dtype=float)

    for arch_idx, arch in enumerate(arches):
        row_map = {int(row["layer"]): float(row["dead_ratio"]) for row in stage3_layer_rows[arch]}
        for layer_idx, layer in enumerate(layers):
            matrix[arch_idx, layer_idx] = row_map.get(layer, math.nan)

    plt.figure(figsize=(max(6, len(layers) * 0.9), 3 + len(arches)))
    im = plt.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    plt.colorbar(im, label="dead_ratio")
    plt.yticks(range(len(arches)), [arch.upper() for arch in arches])
    plt.xticks(range(len(layers)), [str(layer) for layer in layers])
    plt.xlabel("Layer")
    plt.ylabel("Architecture")
    plt.title("Stage3 dead feature ratio by layer")
    plt.savefig(cfg.run_context.plots_dir / "dead_ratio_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close()


def _save_top_candidates(cfg: SaeExperimentConfig, rows: list[dict[str, Any]]) -> None:
    assert cfg.run_context is not None
    for row in rows:
        append_csv_row(cfg.run_context.top_candidates_path, row)


def _candidate_trial_cfg(base: TrialResult, stage: str, epoch_budget: int, trial_id: int) -> TrialConfig:
    return TrialConfig(
        arch=base.arch,
        stage=stage,  # type: ignore[arg-type]
        trial_id=trial_id,
        epoch_budget=epoch_budget,
        seed=base.seed,
        k=base.k,
        l0_coefficient=base.l0_coefficient,
    )


def run_sweep(
    cfg: SaeExperimentConfig,
    *,
    arch_mode: str,
    resume: bool = False,
) -> dict[str, TrialResult]:
    if cfg.run_context is None:
        cfg.ensure_dirs()
    assert cfg.run_context is not None

    config_payload = to_wandb_config(
        {
            "config": cfg,
            "mode": "sae_sweep",
            "arch_mode": arch_mode,
            "resume": resume,
        }
    )
    wandb_logger = WandbRunLogger(
        logger_cfg=cfg.logger,
        run_root=cfg.run_context.run_root,
        run_id=cfg.run_id or "unknown_run",
        config_payload=config_payload,
        job_type="sae_sweep",
        tags=["sae", "sweep"],
    )
    wandb_logger.start()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(
            cfg.model_name, local_files_only=cfg.local_only
        )
        model, _ = build_mlm_model(cfg, device)
        resolve_layers_from_model(cfg, model)
        prepare_activation_cache(cfg, model, tokenizer, device)

        arch_list = list(ARCH_CHOICES) if arch_mode == "both" else [arch_mode]
        stage3_winners: dict[str, TrialResult] = {}
        heatmap_rows: dict[str, list[dict[str, Any]]] = {}
        top_candidate_rows: list[dict[str, Any]] = []

        for arch in arch_list:
            stage1_trials = _build_stage1_trials(cfg, arch)
            stage1_scored: list[dict[str, Any]] = []

            for trial_cfg in stage1_trials:
                trial_root = _trial_root(
                    cfg,
                    arch,
                    trial_cfg.trial_id,
                    _trial_param_label_from_cfg(trial_cfg),
                )
                try:
                    result, layer_rows = run_architecture_trial(
                        cfg=cfg,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        trial_cfg=trial_cfg,
                        trial_root=trial_root,
                        run_downstream=False,
                        candidate_label="",
                        resume=resume,
                        wandb_logger=wandb_logger,
                    )
                except Exception:
                    result = TrialResult(
                        run_id=cfg.run_context.run_id,
                        arch=trial_cfg.arch,
                        stage=trial_cfg.stage,
                        trial_id=trial_cfg.trial_id,
                        k=trial_cfg.k,
                        l0_coefficient=trial_cfg.l0_coefficient,
                        epochs=trial_cfg.epoch_budget,
                        seed=trial_cfg.seed,
                        nmse_mean=float("inf"),
                        nmse_std=float("inf"),
                        dead_ratio=1.0,
                        mean_l0=float("inf"),
                        max_node_share=float("inf"),
                        active_cosine_mean=float("inf"),
                        decoder_cosine_max=float("inf"),
                        global_step=0,
                        status="failed",
                        trial_root=trial_root,
                    )
                    layer_rows = []
                    save_json(trial_root / "metrics.json", trial_result_to_dict(result))

                _append_registry(cfg, result)
                mean_l0_cv = _compute_l0_cv(layer_rows) if layer_rows else float("inf")
                score = _score_for_ranking(result, mean_l0_cv)
                stage1_scored.append(
                    {
                        "result": result,
                        "layer_rows": layer_rows,
                        "mean_l0_cv": mean_l0_cv,
                        "score": score,
                    }
                )

            _plot_stage1_curve(cfg, arch, stage1_scored)
            eligible = [
                item
                for item in stage1_scored
                if item["result"].status == "ok"
                and item["result"].dead_ratio <= cfg.dead_feature_ratio_max
                and item["result"].max_node_share <= cfg.node_concentration_max
                and item["result"].active_cosine_mean <= cfg.activation_similarity_max
                and item["result"].decoder_cosine_max <= cfg.decoder_redundancy_max
            ]
            ranked_stage1 = sorted(
                eligible,
                key=lambda item: (
                    item["result"].nmse_mean,
                    item["result"].nmse_std,
                    item["result"].max_node_share,
                    item["result"].active_cosine_mean,
                    item["result"].decoder_cosine_max,
                    item["mean_l0_cv"],
                    item["result"].global_step,
                ),
            )
            if not ranked_stage1:
                ranked_stage1 = sorted(stage1_scored, key=lambda item: item["score"])
            top3_stage1 = ranked_stage1[:3]

            stage2_scored: list[dict[str, Any]] = []
            for rank_idx, item in enumerate(top3_stage1, start=1):
                base_result: TrialResult = item["result"]
                top_candidate_rows.append(
                    {
                        "arch": arch,
                        "rank": rank_idx,
                        "trial_id": base_result.trial_id,
                        "k": base_result.k if base_result.k is not None else "",
                        "l0_coefficient": (
                            base_result.l0_coefficient
                            if base_result.l0_coefficient is not None
                            else ""
                        ),
                        "score": item["score"],
                    }
                )
                stage2_cfg = _candidate_trial_cfg(
                    base=base_result,
                    stage="stage2",
                    epoch_budget=cfg.stage2_epochs,
                    trial_id=rank_idx,
                )
                stage2_root = cfg.run_context.final_dir / arch / f"candidate_{rank_idx}"
                try:
                    result2, layer_rows2 = run_architecture_trial(
                        cfg=cfg,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        trial_cfg=stage2_cfg,
                        trial_root=stage2_root,
                        run_downstream=False,
                        candidate_label=f"candidate_{rank_idx}",
                        resume=False,
                        wandb_logger=wandb_logger,
                    )
                except Exception:
                    result2 = TrialResult(
                        run_id=cfg.run_context.run_id,
                        arch=stage2_cfg.arch,
                        stage=stage2_cfg.stage,
                        trial_id=stage2_cfg.trial_id,
                        k=stage2_cfg.k,
                        l0_coefficient=stage2_cfg.l0_coefficient,
                        epochs=stage2_cfg.epoch_budget,
                        seed=stage2_cfg.seed,
                        nmse_mean=float("inf"),
                        nmse_std=float("inf"),
                        dead_ratio=1.0,
                        mean_l0=float("inf"),
                        max_node_share=float("inf"),
                        active_cosine_mean=float("inf"),
                        decoder_cosine_max=float("inf"),
                        global_step=0,
                        status="failed",
                        trial_root=stage2_root,
                    )
                    layer_rows2 = []
                    save_json(stage2_root / "metrics.json", trial_result_to_dict(result2))
                _append_registry(cfg, result2)
                for row in layer_rows2:
                    append_csv_row(cfg.run_context.quality_log_path, row)
                stage2_scored.append(
                    {
                        "result": result2,
                        "layer_rows": layer_rows2,
                        "mean_l0_cv": _compute_l0_cv(layer_rows2),
                    }
                )

            if not stage2_scored:
                continue

            eligible_stage2 = [
                item
                for item in stage2_scored
                if item["result"].status == "ok"
                and item["result"].dead_ratio <= cfg.dead_feature_ratio_max
                and item["result"].max_node_share <= cfg.node_concentration_max
                and item["result"].active_cosine_mean <= cfg.activation_similarity_max
                and item["result"].decoder_cosine_max <= cfg.decoder_redundancy_max
            ]
            ranked_stage2 = sorted(
                eligible_stage2,
                key=lambda item: (
                    item["result"].nmse_mean,
                    item["result"].nmse_std,
                    item["result"].max_node_share,
                    item["result"].active_cosine_mean,
                    item["result"].decoder_cosine_max,
                    item["mean_l0_cv"],
                    item["result"].global_step,
                ),
            )
            if not ranked_stage2:
                ranked_stage2 = sorted(
                    stage2_scored,
                    key=lambda item: _score_for_ranking(item["result"], item["mean_l0_cv"]),
                )
            winner_base = ranked_stage2[0]["result"]
            stage3_cfg = _candidate_trial_cfg(
                base=winner_base,
                stage="stage3",
                epoch_budget=cfg.stage3_epochs,
                trial_id=winner_base.trial_id,
            )
            stage3_root = cfg.run_context.final_dir / arch / "winner"
            stage3_result, stage3_layer_rows = run_architecture_trial(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                device=device,
                trial_cfg=stage3_cfg,
                trial_root=stage3_root,
                run_downstream=True,
                candidate_label="winner",
                resume=False,
                wandb_logger=wandb_logger,
            )
            _append_registry(cfg, stage3_result)
            for row in stage3_layer_rows:
                append_csv_row(cfg.run_context.quality_log_path, row)
            heatmap_rows[arch] = stage3_layer_rows
            stage3_winners[arch] = stage3_result

            run_reference_evaluations(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                device=device,
                candidate_label="winner",
                arch=arch,
            )

        _save_top_candidates(cfg, top_candidate_rows)
        _plot_dead_ratio_heatmap(cfg, heatmap_rows)
        save_json(
            cfg.run_context.run_meta_path,
            {
                "run_id": cfg.run_context.run_id,
                "mode": "sweep",
                "arch_mode": arch_mode,
                "stage_epochs": [cfg.stage1_epochs, cfg.stage2_epochs, cfg.stage3_epochs],
                "layers": list(cfg.layers),
                "model_name": cfg.model_name,
                "seed": cfg.seed,
                "batchtopk_k_values": list(cfg.batchtopk_k_sweep_values),
                "jumprelu_l0_values": list(cfg.jumprelu_l0_sweep_values),
                "stage_gate": {
                    "dead_feature_ratio_max": cfg.dead_feature_ratio_max,
                    "node_concentration_max": cfg.node_concentration_max,
                    "activation_similarity_max": cfg.activation_similarity_max,
                    "decoder_redundancy_max": cfg.decoder_redundancy_max,
                },
                "winners": {arch: trial_result_to_dict(result) for arch, result in stage3_winners.items()},
            },
        )
        wandb_logger.log_artifact(
            name=f"{cfg.run_id}_sweep_meta",
            artifact_type="sae_sweep",
            files=[cfg.run_context.run_meta_path, cfg.run_context.registry_path, cfg.run_context.top_candidates_path],
            aliases=["latest", "sweep"],
            metadata={"run_id": cfg.run_id, "mode": "sweep", "arch_mode": arch_mode},
        )
        if stage3_winners:
            wandb_logger.update_summary(
                {
                    "sweep_winner_nmse_mean_best": min(result.nmse_mean for result in stage3_winners.values()),
                    "sweep_winner_dead_ratio_max": max(result.dead_ratio for result in stage3_winners.values()),
                }
            )
        if wandb_logger.run_url is not None:
            print(f"wandb_run_url={wandb_logger.run_url}")
        return stage3_winners
    finally:
        wandb_logger.finish()
