from __future__ import annotations

from dataclasses import asdict, dataclass
import html
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import SaeExperimentConfig
from chem_sae.data import ActivationChunkDataset, prepare_mlm_loader
from chem_sae.modeling import RobertaForMaskedLM, build_mlm_model
from chem_sae.train.quality_metrics import LayerQuality, evaluate_layer_quality
from chem_sae.utils import (
    WandbRunLogger,
    build_probe_wandb_metadata,
    build_sparsity_log_dict,
    build_train_step_log_dict,
    capture_rng_state,
    latest_checkpoint,
    list_chunks,
    restore_rng_state,
    save_checkpoint,
    save_json,
    set_seed,
    to_wandb_config,
    write_chunk,
)
from chem_sae.vendor import JumpReLUAutoencoder, jumprelu_loss_with_details


@dataclass(frozen=True)
class ProbeLayerResult:
    layer: int
    nmse: float
    mean_l0: float
    dead_ratio: float
    max_node_share: float
    active_cosine_mean: float
    decoder_cosine_max: float
    global_step: int
    checkpoint_path: Path


def _format_metric(value: Any, precision: int = 6) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{precision}f}"
    return "-"


def _line_chart_svg(
    values: list[float],
    *,
    color: str,
    width: int = 760,
    height: int = 340,
) -> str:
    if not values:
        return '<div class="empty-chart">No data yet</div>'

    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-12:
        max_v = min_v + 1e-12

    padding_x = 36.0
    padding_y = 24.0
    usable_w = max(1.0, width - (2 * padding_x))
    usable_h = max(1.0, height - (2 * padding_y))
    denom = max(1, len(values) - 1)

    points: list[str] = []
    for idx, value in enumerate(values):
        x = padding_x + usable_w * (idx / denom)
        y = padding_y + (1.0 - ((value - min_v) / (max_v - min_v))) * usable_h
        points.append(f"{x:.2f},{y:.2f}")

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />'
        f'<line x1="{padding_x}" y1="{padding_y}" x2="{padding_x}" y2="{height - padding_y}" '
        f'stroke="#d1d5db" stroke-width="1"/>'
        f'<line x1="{padding_x}" y1="{height - padding_y}" x2="{width - padding_x}" y2="{height - padding_y}" '
        f'stroke="#d1d5db" stroke-width="1"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.25"/>'
        f'<text x="{padding_x}" y="12" class="axis-label">max {_format_metric(max(values), 6)}</text>'
        f'<text x="{padding_x}" y="{height - 3}" class="axis-label">min {_format_metric(min(values), 6)}</text>'
        "</svg>"
    )


def _write_probe_layer_dashboard(
    *,
    probe_root: Path,
    run_id: str,
    layer: int,
    rows: list[dict[str, Any]],
    schedule_mode: str,
    base_l0: float,
    warmup_epochs: int,
    decay_ratio: float,
    dashboard_every_n_epochs: int,
    mirror_dashboard_dir: Path | None = None,
) -> Path:
    dashboard_dir = probe_root / "dashboards"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = dashboard_dir / f"layer_{layer}.html"

    nmse_values = [float(r["nmse"]) for r in rows if isinstance(r.get("nmse"), (int, float))]
    l0_values = [float(r["mean_l0"]) for r in rows if isinstance(r.get("mean_l0"), (int, float))]
    dead_values = [
        float(r["dead_ratio"]) for r in rows if isinstance(r.get("dead_ratio"), (int, float))
    ]
    coef_values = [
        float(r["l0_coef_eff"]) for r in rows if isinstance(r.get("l0_coef_eff"), (int, float))
    ]

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row.get('epoch', '-')}</td>"
            f"<td>{_format_metric(row.get('train_loss'), 6)}</td>"
            f"<td>{_format_metric(row.get('nmse'), 6)}</td>"
            f"<td>{_format_metric(row.get('mean_l0'), 4)}</td>"
            f"<td>{_format_metric(row.get('dead_ratio'), 6)}</td>"
            f"<td>{_format_metric(row.get('l0_coef_eff'), 8)}</td>"
            f"<td>{row.get('global_step', '-')}</td>"
            f"<td>{_format_metric(row.get('best_nmse'), 6)}</td>"
            "</tr>"
        )

    latest_epoch = rows[-1]["epoch"] if rows else "-"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="20" />
  <title>Probe Dashboard - layer {layer}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --line: #e2e8f0;
      --muted: #475569;
    }}
    body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--text); font-family: "IBM Plex Sans", "Noto Sans KR", sans-serif; }}
    .top {{ margin-bottom: 16px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 22px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .card h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .chart-svg {{ width: 100%; height: auto; }}
    .axis-label {{ font-size: 11px; fill: #64748b; }}
    .empty-chart {{ min-height: 80px; display: grid; place-items: center; color: #94a3b8; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; font-size: 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: center; }}
    th {{ background: #f1f5f9; font-weight: 700; }}
    tr:last-child td {{ border-bottom: none; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="top">
    <h1>Probe Dashboard - Layer {layer}</h1>
    <div class="meta">
      run_id={html.escape(run_id)} | latest_epoch={latest_epoch} | schedule={html.escape(schedule_mode)} |
      base_l0={base_l0} | warmup_epochs={warmup_epochs} | decay_ratio={decay_ratio} |
      dashboard_every_n_epochs={dashboard_every_n_epochs}
    </div>
    <div class="meta"><a href="./index.html">Back to index</a></div>
  </div>

  <div class="grid">
    <div class="card"><h3>NMSE</h3>{_line_chart_svg(nmse_values, color="#1d4ed8")}</div>
    <div class="card"><h3>Mean L0</h3>{_line_chart_svg(l0_values, color="#0f766e")}</div>
    <div class="card"><h3>Dead Ratio</h3>{_line_chart_svg(dead_values, color="#b45309")}</div>
    <div class="card"><h3>L0 Coef Eff</h3>{_line_chart_svg(coef_values, color="#9333ea")}</div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Epoch</th>
        <th>Train Loss</th>
        <th>NMSE</th>
        <th>Mean L0</th>
        <th>Dead Ratio</th>
        <th>L0 Coef Eff</th>
        <th>Global Step</th>
        <th>Best NMSE</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    dashboard_path.write_text(html_text, encoding="utf-8")
    if mirror_dashboard_dir is not None:
        mirror_dashboard_dir.mkdir(parents=True, exist_ok=True)
        (mirror_dashboard_dir / f"layer_{layer}.html").write_text(html_text, encoding="utf-8")
    return dashboard_path


def _write_probe_index_dashboard(
    *,
    probe_root: Path,
    run_id: str,
    layer_results: list[ProbeLayerResult],
    schedule_mode: str,
    base_l0: float,
    warmup_epochs: int,
    decay_ratio: float,
    mirror_dashboard_dir: Path | None = None,
) -> Path:
    dashboard_dir = probe_root / "dashboards"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    index_path = dashboard_dir / "index.html"

    rows = []
    for res in sorted(layer_results, key=lambda x: x.layer):
        rows.append(
            "<tr>"
            f"<td>{res.layer}</td>"
            f"<td>{res.nmse:.6f}</td>"
            f"<td>{res.mean_l0:.4f}</td>"
            f"<td>{res.dead_ratio:.6f}</td>"
            f"<td>{res.global_step}</td>"
            f'<td><a href="./layer_{res.layer}.html">open</a></td>'
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Probe Dashboard Index</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f8fafc; color: #0f172a; font-family: "IBM Plex Sans", "Noto Sans KR", sans-serif; }}
    .meta {{ color: #475569; font-size: 14px; margin-bottom: 12px; }}
    .links a {{ margin-right: 16px; color: #1d4ed8; text-decoration: none; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; font-size: 13px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: center; }}
    th:last-child, td:last-child {{ text-align: center; }}
    th {{ background: #f1f5f9; font-weight: 700; }}
    tr:last-child td {{ border-bottom: none; }}
  </style>
</head>
<body>
  <h1>Probe Dashboard Index</h1>
  <div class="meta">
    run_id={html.escape(run_id)} | schedule={html.escape(schedule_mode)} | base_l0={base_l0} |
    warmup_epochs={warmup_epochs} | decay_ratio={decay_ratio}
  </div>
  <div class="links">
    <a href="../metrics.json">metrics.json</a>
    <a href="../schedule_trace.json">schedule_trace.json</a>
  </div>
  <table>
    <thead>
      <tr>
        <th>Layer</th>
        <th>NMSE</th>
        <th>Mean L0</th>
        <th>Dead Ratio</th>
        <th>Global Step</th>
        <th>Dashboard</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    index_path.write_text(html_text, encoding="utf-8")
    if mirror_dashboard_dir is not None:
        mirror_dashboard_dir.mkdir(parents=True, exist_ok=True)
        (mirror_dashboard_dir / "index.html").write_text(html_text, encoding="utf-8")
    return index_path


def _ensure_ready(cfg: SaeExperimentConfig) -> None:
    if cfg.run_context is None:
        cfg.ensure_dirs()


@torch.no_grad()
def extract_attn_activations(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    loader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    for layer in cfg.layers:
        layer_dir = cfg.acts_dir / f"layer_{layer}"
        chunk_idx = 0
        buffered: list[torch.Tensor] = []
        buffered_tokens = 0
        total_tokens = 0
        chunk_paths: list[Path] = []
        d_model = None

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, attn_outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attn_outputs=True,
                attn_output_layers={layer},
            )
            attn = attn_outputs[layer]
            d_model = attn.shape[-1]
            flat = attn[attention_mask.bool()].detach().cpu().to(torch.float16)
            buffered.append(flat)
            buffered_tokens += flat.shape[0]
            total_tokens += flat.shape[0]

            if buffered_tokens >= cfg.chunk_size:
                chunk = torch.cat(buffered, dim=0)
                chunk_paths.append(write_chunk(layer_dir, chunk_idx, chunk))
                chunk_idx += 1
                buffered = []
                buffered_tokens = 0

        if buffered:
            chunk = torch.cat(buffered, dim=0)
            chunk_paths.append(write_chunk(layer_dir, chunk_idx, chunk))

        if d_model is None:
            raise RuntimeError(
                f"No activations extracted for layer {layer}. "
                f"Check MLM loader and dataset at {cfg.mlm_data_path}."
            )

        save_json(
            layer_dir / "meta.json",
            {
                "layer": layer,
                "d_model": d_model,
                "num_tokens": total_tokens,
                "num_chunks": len(chunk_paths),
                "chunk_size": cfg.chunk_size,
                "dtype": "float16",
                "model_name": cfg.model_name,
                "mlm_data_path": str(cfg.mlm_data_path),
            },
        )


def _has_activation_cache(cfg: SaeExperimentConfig) -> bool:
    if not cfg.layers:
        return False
    for layer in cfg.layers:
        layer_dir = cfg.acts_dir / f"layer_{layer}"
        if not layer_dir.exists() or not any(layer_dir.glob("chunk_*.pt")):
            return False
    return True


def prepare_activation_cache(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    device: torch.device,
) -> None:
    if not _has_activation_cache(cfg):
        loader = prepare_mlm_loader(cfg, tokenizer)
        extract_attn_activations(cfg, model, loader, device)


def resolve_layers_from_model(cfg: SaeExperimentConfig, model: RobertaForMaskedLM) -> tuple[int, ...]:
    return cfg.resolve_layers(model.config.num_hidden_layers)


def _split_chunk_paths(cfg: SaeExperimentConfig, layer: int) -> tuple[list[Path], list[Path]]:
    layer_dir = cfg.acts_dir / f"layer_{layer}"
    chunk_paths = list_chunks(layer_dir)
    if not chunk_paths:
        raise FileNotFoundError(f"No activation chunks found in {layer_dir}")

    train_cut = max(1, int(len(chunk_paths) * (1 - cfg.val_fraction)))
    train_paths = chunk_paths[:train_cut]
    val_paths = chunk_paths[train_cut:] or train_paths[-1:]
    return train_paths, val_paths


def _resolve_effective_l0(
    *,
    base_l0: float,
    schedule_mode: Literal["none", "two_step", "exp"],
    epoch_idx: int,
    warmup_epochs: int,
    decay_ratio: float,
) -> float:
    if schedule_mode == "none":
        return base_l0
    warmup = max(0, warmup_epochs)
    if epoch_idx < warmup:
        return base_l0
    if schedule_mode == "two_step":
        return base_l0 * decay_ratio
    if schedule_mode == "exp":
        decay_steps = (epoch_idx - warmup) + 1
        return base_l0 * (decay_ratio**decay_steps)
    raise ValueError(f"Unsupported schedule mode: {schedule_mode}")


def _resolve_warmup_coefficient(
    final_value: float,
    warm_up_steps: int,
    current_step: int,
) -> float:
    if warm_up_steps <= 0:
        return final_value
    if current_step <= 0:
        return 0.0
    return final_value * min(1.0, current_step / warm_up_steps)


def _evaluate_quality(
    cfg: SaeExperimentConfig,
    model: torch.nn.Module,
    val_data,
    device: torch.device,
    *,
    compute_structure_metrics: bool,
) -> LayerQuality:
    return evaluate_layer_quality(
        model,
        val_data,
        device,
        compute_structure_metrics=compute_structure_metrics,
        similarity_feature_cap=cfg.quality_metric_feature_cap,
        similarity_sample_cap=cfg.quality_metric_sample_cap,
    )


def train_probe_for_layer(
    cfg: SaeExperimentConfig,
    layer: int,
    device: torch.device,
    *,
    epochs: int,
    base_l0: float,
    schedule_mode: Literal["none", "two_step", "exp"],
    warmup_epochs: int,
    decay_ratio: float,
    checkpoint_root: Path,
    resume: bool = False,
    trial_seed: int | None = None,
    dashboard_mirror_dir: Path | None = None,
    wandb_logger: WandbRunLogger | None = None,
) -> tuple[ProbeLayerResult, list[dict[str, Any]]]:
    effective_seed = cfg.seed if trial_seed is None else trial_seed
    train_paths, val_paths = _split_chunk_paths(cfg, layer)
    train_data = ActivationChunkDataset(
        train_paths, batch_size=cfg.sae_batch_size, shuffle=True, seed=effective_seed
    )
    val_data = ActivationChunkDataset(
        val_paths, batch_size=cfg.sae_batch_size, shuffle=False, seed=effective_seed
    )

    d_model = torch.load(train_paths[0], map_location="cpu", weights_only=True).shape[1]
    model = JumpReLUAutoencoder(
        n_latents=cfg.n_latents,
        n_inputs=d_model,
        threshold_init=cfg.jumprelu_threshold,
        bandwidth=cfg.jumprelu_bandwidth,
        normalize=True,
        sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
        tanh_scale=cfg.jumprelu_tanh_scale,
        pre_act_loss_coefficient=cfg.jumprelu_pre_act_loss_coefficient,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.sae_lr)

    ckpt_dir = checkpoint_root / f"layer_{layer}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_nmse = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_nmse_history: list[float] = []
    schedule_trace: list[dict[str, Any]] = []
    n_training_samples = 0
    act_freq_scores = torch.zeros(cfg.n_latents, device=device)
    n_forward_passes_since_fired = torch.zeros(
        cfg.n_latents, device=device, dtype=torch.long
    )
    n_frac_active_samples = 0
    probe_root = checkpoint_root.parent

    if resume:
        latest = latest_checkpoint(ckpt_dir)
        if latest is not None:
            state = torch.load(latest, map_location=device, weights_only=False)
            model_state = dict(state["model"])
            model.process_state_dict_for_loading(model_state)
            model.load_state_dict(model_state, strict=False)
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state.get("epoch", -1) + 1
            global_step = state.get("step", 0)
            best_nmse = state.get("best_nmse", float("inf"))
            patience_counter = state.get("patience_counter", 0)
            train_losses = state.get("train_losses", [])
            val_nmse_history = state.get("val_nmse_history", [])
            schedule_trace = state.get("schedule_trace", [])
            n_training_samples = int(state.get("n_training_samples", 0))
            loaded_act_freq_scores = state.get("act_freq_scores")
            if isinstance(loaded_act_freq_scores, torch.Tensor):
                act_freq_scores = loaded_act_freq_scores.to(device)
            loaded_passes_since_fired = state.get("n_forward_passes_since_fired")
            if isinstance(loaded_passes_since_fired, torch.Tensor):
                n_forward_passes_since_fired = loaded_passes_since_fired.to(
                    device=device, dtype=torch.long
                )
            n_frac_active_samples = int(state.get("n_frac_active_samples", 0))
            restore_rng_state(state.get("rng_state"))

    for epoch in range(start_epoch, epochs):
        effective_l0 = _resolve_effective_l0(
            base_l0=base_l0,
            schedule_mode=schedule_mode,
            epoch_idx=epoch,
            warmup_epochs=warmup_epochs,
            decay_ratio=decay_ratio,
        )
        schedule_trace.append(
            {
                "epoch": epoch + 1,
                "base_l0": base_l0,
                "effective_l0": effective_l0,
                "schedule_mode": schedule_mode,
                "warmup_epochs": warmup_epochs,
                "decay_ratio": decay_ratio,
            }
        )

        model.train()
        train_loss_total = 0.0
        num_batches = 0
        last_batch_l0_coefficient = effective_l0

        for batch in train_data:
            batch = batch.to(device).float()
            hidden_pre, latents, recons = model(batch)
            batch_l0_coefficient = _resolve_warmup_coefficient(
                final_value=effective_l0,
                warm_up_steps=cfg.jumprelu_l0_warm_up_steps,
                current_step=global_step,
            )
            last_batch_l0_coefficient = batch_l0_coefficient
            dead_neuron_mask = (
                n_forward_passes_since_fired > cfg.jumprelu_dead_feature_window
            ).detach()
            loss, loss_components = jumprelu_loss_with_details(
                reconstruction=recons,
                original_input=batch,
                latent_activations=latents,
                hidden_pre=hidden_pre,
                l0_coefficient=batch_l0_coefficient,
                l1_weight=cfg.l1_weight,
                model=model,
                sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
                tanh_scale=cfg.jumprelu_tanh_scale,
                pre_act_loss_coefficient=cfg.jumprelu_pre_act_loss_coefficient,
                dead_neuron_mask=dead_neuron_mask,
            )
            with torch.no_grad():
                firing_feats = latents.detach().bool().float()
                did_fire = firing_feats.sum(dim=0).bool()
                n_forward_passes_since_fired += 1
                n_forward_passes_since_fired[did_fire] = 0
                act_freq_scores += firing_feats.sum(dim=0)
                n_frac_active_samples += int(batch.shape[0])
                n_training_samples += int(batch.shape[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_total += float(loss.item())
            num_batches += 1
            global_step += 1

            if wandb_logger is not None and wandb_logger.should_log(global_step):
                wandb_logger.log(
                    build_train_step_log_dict(
                        sae_in=batch,
                        sae_out=recons,
                        feature_acts=latents,
                        overall_loss=loss,
                        losses=loss_components,
                        metrics={},
                        current_learning_rate=float(optimizer.param_groups[0]["lr"]),
                        n_training_samples=n_training_samples,
                        n_forward_passes_since_fired=n_forward_passes_since_fired,
                        dead_feature_window=cfg.jumprelu_dead_feature_window,
                        coefficients={
                            "l0": float(batch_l0_coefficient),
                            "l1": float(cfg.l1_weight),
                        },
                        global_step=global_step,
                    ),
                    step=global_step,
                )

        quality = _evaluate_quality(
            cfg,
            model,
            val_data,
            device,
            compute_structure_metrics=False,
        )
        train_avg = train_loss_total / max(1, num_batches)
        train_losses.append(train_avg)
        val_nmse_history.append(quality.nmse)

        improved = quality.nmse < best_nmse
        if improved:
            best_nmse = quality.nmse
            patience_counter = 0
        else:
            patience_counter += 1

        schedule_trace[-1].update(
            {
                "train_loss": train_avg,
                "nmse": quality.nmse,
                "mean_l0": quality.mean_l0,
                "dead_ratio": quality.dead_ratio,
                "l0_coef_eff": last_batch_l0_coefficient,
                "global_step": global_step,
                "best_nmse": best_nmse,
                "patience_counter": patience_counter,
            }
        )

        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "epoch/train_loss": train_avg,
                    "epoch/nmse": quality.nmse,
                    "epoch/mean_l0": quality.mean_l0,
                    "epoch/dead_ratio": quality.dead_ratio,
                    "epoch/l0_coefficient": last_batch_l0_coefficient,
                    "epoch/patience_counter": patience_counter,
                    "epoch/index": epoch + 1,
                },
                step=global_step,
            )
            if wandb_logger.should_eval_log(global_step):
                eval_log: dict[str, Any] = {
                    "eval/val_nmse": quality.nmse,
                    "eval/val_mean_l0": quality.mean_l0,
                    "eval/val_dead_ratio": quality.dead_ratio,
                    "eval/l0_coefficient": last_batch_l0_coefficient,
                }
                sparsity_dict = build_sparsity_log_dict(
                    act_freq_scores=act_freq_scores,
                    n_frac_active_samples=n_frac_active_samples,
                )
                if "plots/feature_density_line_chart" in sparsity_dict:
                    sparsity_dict["plots/feature_density_line_chart"] = wandb_logger.histogram(
                        sparsity_dict["plots/feature_density_line_chart"]
                    )
                eval_log.update(sparsity_dict)
                for key, value in model.log_histograms().items():
                    eval_log[key] = wandb_logger.histogram(value)
                wandb_logger.log(eval_log, step=global_step)
                act_freq_scores = torch.zeros_like(act_freq_scores)
                n_frac_active_samples = 0

        model_state = model.state_dict()
        model.process_state_dict_for_saving(model_state)
        checkpoint_state = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "best_nmse": best_nmse,
            "patience_counter": patience_counter,
            "train_losses": train_losses,
            "val_nmse_history": val_nmse_history,
            "schedule_trace": schedule_trace,
            "n_training_samples": n_training_samples,
            "act_freq_scores": act_freq_scores.detach().cpu(),
            "n_forward_passes_since_fired": n_forward_passes_since_fired.detach().cpu(),
            "n_frac_active_samples": n_frac_active_samples,
            "quality": asdict(quality),
            "rng_state": capture_rng_state(),
        }
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", checkpoint_state)

        save_checkpoint(
            ckpt_dir / "latest.pt",
            checkpoint_state,
        )

        should_write_layer_dashboard = (
            cfg.probe_dashboard_enabled
            and cfg.probe_dashboard_every_n_epochs > 0
            and (
                ((epoch + 1) % cfg.probe_dashboard_every_n_epochs == 0)
                or (epoch == epochs - 1)
            )
        )
        if should_write_layer_dashboard:
            _write_probe_layer_dashboard(
                probe_root=probe_root,
                run_id=cfg.run_id or "unknown_run",
                layer=layer,
                rows=schedule_trace,
                schedule_mode=schedule_mode,
                base_l0=base_l0,
                warmup_epochs=warmup_epochs,
                decay_ratio=decay_ratio,
                dashboard_every_n_epochs=cfg.probe_dashboard_every_n_epochs,
                mirror_dashboard_dir=dashboard_mirror_dir,
            )

        print(
            f"[probe jumprelu layer {layer}] epoch {epoch+1}/{epochs} "
            f"train_loss={train_avg:.4f} nmse={quality.nmse:.4f} "
            f"l0={quality.mean_l0:.2f} dead={quality.dead_ratio:.4f} "
            f"l0_coef_eff={last_batch_l0_coefficient:.8f} "
            f"(best_nmse={best_nmse:.4f}, patience={patience_counter}/{cfg.early_stopping_patience})"
        )

        if patience_counter >= cfg.early_stopping_patience:
            if (
                cfg.probe_dashboard_enabled
                and cfg.probe_dashboard_every_n_epochs > 0
                and not should_write_layer_dashboard
            ):
                _write_probe_layer_dashboard(
                    probe_root=probe_root,
                    run_id=cfg.run_id or "unknown_run",
                    layer=layer,
                    rows=schedule_trace,
                    schedule_mode=schedule_mode,
                    base_l0=base_l0,
                    warmup_epochs=warmup_epochs,
                    decay_ratio=decay_ratio,
                    dashboard_every_n_epochs=cfg.probe_dashboard_every_n_epochs,
                    mirror_dashboard_dir=dashboard_mirror_dir,
                )
            break

    best_state = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    best_model_state = dict(best_state["model"])
    model.process_state_dict_for_loading(best_model_state)
    model.load_state_dict(best_model_state, strict=False)
    best_quality = _evaluate_quality(
        cfg,
        model,
        val_data,
        device,
        compute_structure_metrics=True,
    )

    if wandb_logger is not None:
        wandb_logger.log(
            {
                "final/nmse": best_quality.nmse,
                "final/mean_l0": best_quality.mean_l0,
                "final/dead_ratio": best_quality.dead_ratio,
                "final/max_node_share": best_quality.max_node_share,
                "final/active_cosine_mean": best_quality.active_cosine_mean,
                "final/decoder_cosine_max": best_quality.decoder_cosine_max,
            },
            step=global_step,
        )

    return (
        ProbeLayerResult(
            layer=layer,
            nmse=best_quality.nmse,
            mean_l0=best_quality.mean_l0,
            dead_ratio=best_quality.dead_ratio,
            max_node_share=best_quality.max_node_share,
            active_cosine_mean=best_quality.active_cosine_mean,
            decoder_cosine_max=best_quality.decoder_cosine_max,
            global_step=global_step,
            checkpoint_path=ckpt_dir / "best.pt",
        ),
        schedule_trace,
    )


def _aggregate_probe_metrics(layer_results: list[ProbeLayerResult]) -> dict[str, float]:
    nmse_values = [res.nmse for res in layer_results]
    l0_values = [res.mean_l0 for res in layer_results]
    dead_values = [res.dead_ratio for res in layer_results]
    global_steps = [res.global_step for res in layer_results]
    return {
        "nmse_mean": mean(nmse_values),
        "nmse_std": pstdev(nmse_values) if len(nmse_values) > 1 else 0.0,
        "mean_l0": mean(l0_values),
        "dead_ratio_max": max(dead_values),
        "global_step_sum": float(sum(global_steps)),
    }


def run_probe(
    cfg: SaeExperimentConfig,
    *,
    layers: tuple[int, ...],
    epochs: int,
    base_l0: float,
    schedule_mode: Literal["none", "two_step", "exp"],
    warmup_epochs: int,
    decay_ratio: float,
    resume: bool = False,
) -> dict[str, Any]:
    _ensure_ready(cfg)
    assert cfg.run_context is not None

    wandb_meta = build_probe_wandb_metadata(
        run_id=cfg.run_id or "unknown_run",
        layers=layers,
        base_l0=base_l0,
        n_latents=cfg.n_latents,
        schedule_mode=schedule_mode,
        sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
        run_name=cfg.logger.run_name,
    )
    cfg.logger.run_name = wandb_meta["display_name"]
    cfg.logger.wandb_group = wandb_meta["group"]

    set_seed(cfg.seed)
    config_payload = to_wandb_config(
        {
            "config": cfg,
            "mode": "probe",
            "arch": "jumprelu",
            "layers": list(layers),
            "epochs": epochs,
            "base_l0": base_l0,
            "schedule_mode": schedule_mode,
            "warmup_epochs": warmup_epochs,
            "decay_ratio": decay_ratio,
            "resume": resume,
        }
    )
    wandb_logger = WandbRunLogger(
        logger_cfg=cfg.logger,
        run_root=cfg.run_context.run_root,
        run_id=cfg.run_id or "unknown_run",
        config_payload=config_payload,
        job_type="sae_probe",
        tags=wandb_meta["tags"],
        group=cfg.logger.wandb_group,
    )
    wandb_logger.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(
            cfg.model_name, local_files_only=cfg.local_only
        )
        model, _ = build_mlm_model(cfg, device)
        all_layers = set(resolve_layers_from_model(cfg, model))
        invalid_layers = [layer for layer in layers if layer not in all_layers]
        if invalid_layers:
            raise ValueError(
                f"Invalid probe layers {invalid_layers}; model supports layers "
                f"{sorted(all_layers)}"
            )

        cfg.layers = layers
        prepare_activation_cache(cfg, model, tokenizer, device)

        probe_root = cfg.run_context.run_root / "probe"
        probe_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root = probe_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        dashboard_mirror_dir: Path | None = None
        if cfg.probe_dashboard_enabled:
            dashboard_mirror_dir = cfg.runs_dir / "probe_dashboard_live"
            dashboard_mirror_dir.mkdir(parents=True, exist_ok=True)

        layer_results: list[ProbeLayerResult] = []
        schedule_payload: dict[str, Any] = {
            "run_id": cfg.run_id,
            "schedule_mode": schedule_mode,
            "base_l0": base_l0,
            "warmup_epochs": warmup_epochs,
            "decay_ratio": decay_ratio,
            "l0_warm_up_steps": cfg.jumprelu_l0_warm_up_steps,
            "dead_feature_window": cfg.jumprelu_dead_feature_window,
            "layers": list(layers),
            "epochs": epochs,
            "per_layer": {},
        }

        for layer in layers:
            layer_result, schedule_trace = train_probe_for_layer(
                cfg=cfg,
                layer=layer,
                device=device,
                epochs=epochs,
                base_l0=base_l0,
                schedule_mode=schedule_mode,
                warmup_epochs=warmup_epochs,
                decay_ratio=decay_ratio,
                checkpoint_root=checkpoint_root,
                resume=resume,
                trial_seed=cfg.seed,
                dashboard_mirror_dir=dashboard_mirror_dir,
                wandb_logger=wandb_logger,
            )
            layer_results.append(layer_result)
            schedule_payload["per_layer"][str(layer)] = schedule_trace

        schedule_trace_path = probe_root / "schedule_trace.json"
        save_json(schedule_trace_path, schedule_payload)

        metrics_payload = {
            "run_id": cfg.run_id,
            "mode": "probe",
            "arch": "jumprelu",
            "base_l0": base_l0,
            "schedule_mode": schedule_mode,
            "warmup_epochs": warmup_epochs,
            "decay_ratio": decay_ratio,
            "l0_warm_up_steps": cfg.jumprelu_l0_warm_up_steps,
            "dead_feature_window": cfg.jumprelu_dead_feature_window,
            "layers": list(layers),
            "epochs": epochs,
            "aggregate": _aggregate_probe_metrics(layer_results),
            "layer_results": [
                {
                    "layer": res.layer,
                    "nmse": res.nmse,
                    "mean_l0": res.mean_l0,
                    "dead_ratio": res.dead_ratio,
                    "max_node_share": res.max_node_share,
                    "active_cosine_mean": res.active_cosine_mean,
                    "decoder_cosine_max": res.decoder_cosine_max,
                    "global_step": res.global_step,
                    "checkpoint_path": str(res.checkpoint_path),
                }
                for res in layer_results
            ],
        }
        metrics_path = probe_root / "metrics.json"
        save_json(metrics_path, metrics_payload)

        dashboard_index_path: Path | None = None
        if cfg.probe_dashboard_enabled:
            dashboard_index_path = _write_probe_index_dashboard(
                probe_root=probe_root,
                run_id=cfg.run_id or "unknown_run",
                layer_results=layer_results,
                schedule_mode=schedule_mode,
                base_l0=base_l0,
                warmup_epochs=warmup_epochs,
                decay_ratio=decay_ratio,
                mirror_dashboard_dir=dashboard_mirror_dir,
            )

        wandb_logger.log(
            {
                "summary/nmse_mean": metrics_payload["aggregate"]["nmse_mean"],
                "summary/mean_l0": metrics_payload["aggregate"]["mean_l0"],
                "summary/dead_ratio_max": metrics_payload["aggregate"]["dead_ratio_max"],
                "summary/global_step_sum": metrics_payload["aggregate"]["global_step_sum"],
            }
        )
        wandb_logger.update_summary(
            {
                "probe_nmse_mean": metrics_payload["aggregate"]["nmse_mean"],
                "probe_mean_l0": metrics_payload["aggregate"]["mean_l0"],
                "probe_dead_ratio_max": metrics_payload["aggregate"]["dead_ratio_max"],
            }
        )

        artifact_files: list[Path] = [schedule_trace_path, metrics_path]
        if dashboard_index_path is not None:
            artifact_files.append(dashboard_index_path)
        if cfg.logger.log_weights_to_wandb:
            artifact_files.extend(res.checkpoint_path for res in layer_results)
        if cfg.logger.log_optimizer_state_to_wandb:
            artifact_files.extend(
                checkpoint_root / f"layer_{res.layer}" / "latest.pt" for res in layer_results
            )
        if cfg.logger.log_activations_store_to_wandb:
            artifact_files.extend(cfg.acts_dir / f"layer_{layer}" / "meta.json" for layer in layers)
        wandb_logger.log_artifact(
            name=f"{cfg.run_id}_probe_outputs",
            artifact_type="probe",
            files=artifact_files,
            aliases=["latest", "probe"],
            metadata={
                "run_id": cfg.run_id,
                "schedule_mode": schedule_mode,
                "base_l0": base_l0,
                "layers": list(layers),
                "epochs": epochs,
            },
        )

        result = {
            "run_id": cfg.run_id,
            "run_root": str(cfg.run_context.run_root),
            "schedule_trace_path": str(schedule_trace_path),
            "metrics_path": str(metrics_path),
        }
        if dashboard_index_path is not None:
            result["dashboard_index_path"] = str(dashboard_index_path)
        if dashboard_mirror_dir is not None:
            result["dashboard_live_index_path"] = str(dashboard_mirror_dir / "index.html")
        if wandb_logger.run_url is not None:
            result["wandb_run_url"] = wandb_logger.run_url
        return result
    finally:
        wandb_logger.finish()
