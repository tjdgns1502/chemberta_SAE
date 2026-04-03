from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from chem_sae.config import ARCH_CHOICES, SaeExperimentConfig, TrialConfig, TrialResult
from chem_sae.data import ActivationChunkDataset, prepare_mlm_loader
from chem_sae.eval.downstream import evaluate_baseline_frozen, evaluate_downstream
from chem_sae.eval.final_hidden import evaluate_final_hidden_state
from chem_sae.modeling import RobertaForMaskedLM, build_mlm_model
from chem_sae.train.quality_metrics import LayerQuality, evaluate_layer_quality
from chem_sae.utils import (
    WandbRunLogger,
    append_csv_row,
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
from chem_sae.vendor import (
    Autoencoder,
    BatchTopK,
    JumpReLUAutoencoder,
    TopK,
    jumprelu_loss_with_details,
)


@dataclass(frozen=True)
class LayerTrainResult:
    layer: int
    nmse: float
    mean_l0: float
    dead_ratio: float
    max_node_share: float
    active_cosine_mean: float
    decoder_cosine_max: float
    global_step: int
    checkpoint_path: Path


def trial_result_to_dict(result: TrialResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["trial_root"] = str(result.trial_root)
    return payload


def _ensure_sae_ready(cfg: SaeExperimentConfig) -> None:
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
                f"Check MLM loader and input dataset at {cfg.mlm_data_path}."
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
    force_refresh: bool = False,
) -> None:
    if force_refresh or not _has_activation_cache(cfg):
        loader = prepare_mlm_loader(cfg, tokenizer)
        extract_attn_activations(cfg, model, loader, device)


def resolve_layers_from_model(cfg: SaeExperimentConfig, model: RobertaForMaskedLM) -> tuple[int, ...]:
    return cfg.resolve_layers(model.config.num_hidden_layers)


def _build_autoencoder(
    arch: str,
    cfg: SaeExperimentConfig,
    d_model: int,
    k: int | None,
) -> torch.nn.Module:
    if arch == "batchtopk":
        if k is None:
            raise ValueError("BatchTopK trial requires k")
        return Autoencoder(
            n_latents=cfg.n_latents,
            n_inputs=d_model,
            activation=BatchTopK(float(k)),
            normalize=True,
        )

    if arch == "topk":
        if k is None:
            raise ValueError("TopK trial requires k")
        return Autoencoder(
            n_latents=cfg.n_latents,
            n_inputs=d_model,
            activation=TopK(k),
            normalize=True,
        )

    if arch == "jumprelu":
        return JumpReLUAutoencoder(
            n_latents=cfg.n_latents,
            n_inputs=d_model,
            threshold_init=cfg.jumprelu_threshold,
            bandwidth=cfg.jumprelu_bandwidth,
            normalize=True,
            sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
            tanh_scale=cfg.jumprelu_tanh_scale,
            pre_act_loss_coefficient=cfg.jumprelu_pre_act_loss_coefficient,
        )

    raise ValueError(f"Unsupported architecture '{arch}'")


def _split_chunk_paths(cfg: SaeExperimentConfig, layer: int) -> tuple[list[Path], list[Path]]:
    layer_dir = cfg.acts_dir / f"layer_{layer}"
    chunk_paths = list_chunks(layer_dir)
    if not chunk_paths:
        raise FileNotFoundError(f"No activation chunks found in {layer_dir}")

    train_cut = max(1, int(len(chunk_paths) * (1 - cfg.val_fraction)))
    train_paths = chunk_paths[:train_cut]
    val_paths = chunk_paths[train_cut:] or train_paths[-1:]
    return train_paths, val_paths


def _compute_train_loss(
    arch: str,
    cfg: SaeExperimentConfig,
    model: torch.nn.Module,
    batch: torch.Tensor,
    hidden_pre: torch.Tensor,
    latents: torch.Tensor,
    recons: torch.Tensor,
    l0_coefficient: float | None,
    dead_neuron_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if arch == "batchtopk":
        recon_loss = ((recons - batch) ** 2).sum(dim=-1).mean()
        l1_loss = latents.abs().sum(dim=-1).mean() * cfg.l1_weight
        total = recon_loss + l1_loss
        return total, {
            "mse_loss": recon_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "l0_loss": batch.new_tensor(0.0),
            "pre_act_loss": batch.new_tensor(0.0),
        }

    if arch == "topk":
        recon_loss = ((recons - batch) ** 2).sum(dim=-1).mean()
        l1_loss = latents.abs().sum(dim=-1).mean() * cfg.l1_weight
        total = recon_loss + l1_loss
        return total, {
            "mse_loss": recon_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "l0_loss": batch.new_tensor(0.0),
            "pre_act_loss": batch.new_tensor(0.0),
        }

    if arch == "jumprelu":
        if l0_coefficient is None:
            raise ValueError("JumpReLU trial requires l0_coefficient")
        assert isinstance(model, JumpReLUAutoencoder)
        return jumprelu_loss_with_details(
            reconstruction=recons,
            original_input=batch,
            latent_activations=latents,
            hidden_pre=hidden_pre,
            l0_coefficient=l0_coefficient,
            l1_weight=cfg.l1_weight,
            model=model,
            sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
            tanh_scale=cfg.jumprelu_tanh_scale,
            pre_act_loss_coefficient=cfg.jumprelu_pre_act_loss_coefficient,
            dead_neuron_mask=dead_neuron_mask,
        )

    raise ValueError(f"Unsupported architecture '{arch}'")


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


def train_sae_for_layer(
    cfg: SaeExperimentConfig,
    layer: int,
    device: torch.device,
    *,
    arch: str,
    epochs: int,
    checkpoint_root: Path,
    plot_root: Path,
    k: int | None = None,
    l0_coefficient: float | None = None,
    resume: bool = False,
    trial_seed: int | None = None,
    wandb_logger: WandbRunLogger | None = None,
) -> tuple[torch.nn.Module, LayerTrainResult]:
    effective_seed = cfg.seed if trial_seed is None else trial_seed
    train_paths, val_paths = _split_chunk_paths(cfg, layer)
    train_data = ActivationChunkDataset(
        train_paths, batch_size=cfg.sae_batch_size, shuffle=True, seed=effective_seed
    )
    val_data = ActivationChunkDataset(
        val_paths, batch_size=cfg.sae_batch_size, shuffle=False, seed=effective_seed
    )

    d_model = torch.load(train_paths[0], map_location="cpu", weights_only=True).shape[1]
    model = _build_autoencoder(arch=arch, cfg=cfg, d_model=d_model, k=k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.sae_lr)

    ckpt_dir = checkpoint_root / f"layer_{layer}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_nmse = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_nmse_history: list[float] = []
    n_training_samples = 0
    act_freq_scores = torch.zeros(cfg.n_latents, device=device)
    n_forward_passes_since_fired = torch.zeros(
        cfg.n_latents, device=device, dtype=torch.long
    )
    n_frac_active_samples = 0

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
        model.train()
        train_loss_total = 0.0
        num_batches = 0

        for batch in train_data:
            batch = batch.to(device).float()
            hidden_pre, latents, recons = model(batch)
            batch_l0_coefficient = l0_coefficient
            dead_neuron_mask = None
            if arch == "jumprelu":
                if l0_coefficient is None:
                    raise ValueError("JumpReLU trial requires l0_coefficient")
                assert isinstance(model, JumpReLUAutoencoder)
                batch_l0_coefficient = _resolve_warmup_coefficient(
                    final_value=l0_coefficient,
                    warm_up_steps=cfg.jumprelu_l0_warm_up_steps,
                    current_step=global_step,
                )
                dead_neuron_mask = (
                    n_forward_passes_since_fired > cfg.jumprelu_dead_feature_window
                ).detach()
            loss, loss_components = _compute_train_loss(
                arch=arch,
                cfg=cfg,
                model=model,
                batch=batch,
                hidden_pre=hidden_pre,
                latents=latents,
                recons=recons,
                l0_coefficient=batch_l0_coefficient,
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
                coefficients = {"l1": float(cfg.l1_weight)}
                if batch_l0_coefficient is not None:
                    coefficients["l0"] = float(batch_l0_coefficient)
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
                        coefficients=coefficients,
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

        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "epoch/train_loss": train_avg,
                    "epoch/nmse": quality.nmse,
                    "epoch/mean_l0": quality.mean_l0,
                    "epoch/dead_ratio": quality.dead_ratio,
                    "epoch/index": epoch + 1,
                },
                step=global_step,
            )
            if wandb_logger.should_eval_log(global_step):
                eval_log: dict[str, Any] = {
                    "eval/val_nmse": quality.nmse,
                    "eval/val_mean_l0": quality.mean_l0,
                    "eval/val_dead_ratio": quality.dead_ratio,
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
                if hasattr(model, "log_histograms"):
                    for key, value in model.log_histograms().items():
                        eval_log[key] = wandb_logger.histogram(value)
                wandb_logger.log(eval_log, step=global_step)
                act_freq_scores = torch.zeros_like(act_freq_scores)
                n_frac_active_samples = 0

        improved = quality.nmse < best_nmse
        if improved:
            best_nmse = quality.nmse
            patience_counter = 0
        else:
            patience_counter += 1

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
            ckpt_dir / f"checkpoint_step_{global_step}.pt",
            checkpoint_state,
        )
        save_checkpoint(
            ckpt_dir / "latest.pt",
            checkpoint_state,
        )

        print(
            f"[{arch} layer {layer}] epoch {epoch+1}/{epochs} "
            f"train_loss={train_avg:.4f} nmse={quality.nmse:.4f} "
            f"l0={quality.mean_l0:.2f} dead={quality.dead_ratio:.4f} "
            f"(best_nmse={best_nmse:.4f}, patience={patience_counter}/{cfg.early_stopping_patience})"
        )

        if patience_counter >= cfg.early_stopping_patience:
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_root.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss", marker="o")
        plt.plot(
            range(1, len(val_nmse_history) + 1),
            val_nmse_history,
            label="val_nmse",
            marker="s",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(f"{arch.upper()} Layer {layer}")
        plt.legend()
        plt.grid(True)
        plot_path = plot_root / f"layer_{layer}_loss_curve.png"
        plt.savefig(plot_path, dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[warn] skipped loss-curve plot for layer {layer}: {exc}")

    return model, LayerTrainResult(
        layer=layer,
        nmse=best_quality.nmse,
        mean_l0=best_quality.mean_l0,
        dead_ratio=best_quality.dead_ratio,
        max_node_share=best_quality.max_node_share,
        active_cosine_mean=best_quality.active_cosine_mean,
        decoder_cosine_max=best_quality.decoder_cosine_max,
        global_step=global_step,
        checkpoint_path=ckpt_dir / "best.pt",
    )


def _aggregate_trial_metrics(
    layer_results: list[LayerTrainResult],
) -> tuple[float, float, float, float, float, float, float, int, float]:
    nmse_values = [res.nmse for res in layer_results]
    l0_values = [res.mean_l0 for res in layer_results]
    dead_values = [res.dead_ratio for res in layer_results]
    node_share_values = [res.max_node_share for res in layer_results]
    active_cosine_values = [res.active_cosine_mean for res in layer_results]
    decoder_cosine_values = [res.decoder_cosine_max for res in layer_results]

    nmse_mean = mean(nmse_values)
    nmse_std = pstdev(nmse_values) if len(nmse_values) > 1 else 0.0
    dead_ratio = max(dead_values)
    mean_l0 = mean(l0_values)
    max_node_share = max(node_share_values)
    active_cosine_mean = mean(active_cosine_values)
    decoder_cosine_max = max(decoder_cosine_values)
    mean_l0_cv = (pstdev(l0_values) / mean_l0) if len(l0_values) > 1 and mean_l0 > 0 else 0.0
    global_step = sum(res.global_step for res in layer_results)
    return (
        nmse_mean,
        nmse_std,
        dead_ratio,
        mean_l0,
        max_node_share,
        active_cosine_mean,
        decoder_cosine_max,
        global_step,
        mean_l0_cv,
    )


def run_architecture_trial(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    device: torch.device,
    trial_cfg: TrialConfig,
    trial_root: Path,
    *,
    run_downstream: bool = False,
    candidate_label: str = "",
    resume: bool = False,
    wandb_logger: WandbRunLogger | None = None,
) -> tuple[TrialResult, list[dict[str, Any]]]:
    _ensure_sae_ready(cfg)
    trial_root.mkdir(parents=True, exist_ok=True)
    save_json(trial_root / "config.json", asdict(trial_cfg))
    set_seed(trial_cfg.seed)

    layer_results: list[LayerTrainResult] = []
    quality_rows: list[dict[str, Any]] = []
    for layer in cfg.layers:
        sae, layer_result = train_sae_for_layer(
            cfg=cfg,
            layer=layer,
            device=device,
            arch=trial_cfg.arch,
            epochs=trial_cfg.epoch_budget,
            checkpoint_root=trial_root / "checkpoints",
            plot_root=trial_root / "plots",
            k=trial_cfg.k,
            l0_coefficient=trial_cfg.l0_coefficient,
            resume=resume,
            trial_seed=trial_cfg.seed,
            wandb_logger=wandb_logger,
        )
        layer_results.append(layer_result)
        quality_rows.append(
            {
                "run_id": cfg.run_id,
                "arch": trial_cfg.arch,
                "candidate": candidate_label,
                "layer": layer,
                "nmse": layer_result.nmse,
                "dead_ratio": layer_result.dead_ratio,
                "mean_l0": layer_result.mean_l0,
                "max_node_share": layer_result.max_node_share,
                "active_cosine_mean": layer_result.active_cosine_mean,
                "decoder_cosine_max": layer_result.decoder_cosine_max,
                "ckpt_path": str(layer_result.checkpoint_path),
            }
        )

        if run_downstream:
            evaluate_downstream(
                cfg,
                model,
                sae,
                tokenizer,
                layer,
                device,
                log_path=cfg.run_context.downstream_log_path if cfg.run_context else cfg.log_path,
                run_id=cfg.run_id,
                extra_fields={
                    "arch": trial_cfg.arch,
                    "candidate": candidate_label,
                    "trial_id": trial_cfg.trial_id,
                    "stage": trial_cfg.stage,
                    "k": trial_cfg.k if trial_cfg.k is not None else "",
                    "l0_coefficient": (
                        trial_cfg.l0_coefficient
                        if trial_cfg.l0_coefficient is not None
                        else ""
                    ),
                },
            )

    (
        nmse_mean,
        nmse_std,
        dead_ratio,
        mean_l0,
        max_node_share,
        active_cosine_mean,
        decoder_cosine_max,
        global_step,
        _,
    ) = _aggregate_trial_metrics(layer_results)

    result = TrialResult(
        run_id=cfg.run_id or "",
        arch=trial_cfg.arch,
        stage=trial_cfg.stage,
        trial_id=trial_cfg.trial_id,
        k=trial_cfg.k,
        l0_coefficient=trial_cfg.l0_coefficient,
        epochs=trial_cfg.epoch_budget,
        seed=trial_cfg.seed,
        nmse_mean=nmse_mean,
        nmse_std=nmse_std,
        dead_ratio=dead_ratio,
        mean_l0=mean_l0,
        max_node_share=max_node_share,
        active_cosine_mean=active_cosine_mean,
        decoder_cosine_max=decoder_cosine_max,
        global_step=global_step,
        status="ok",
        trial_root=trial_root,
    )
    save_json(trial_root / "metrics.json", trial_result_to_dict(result))

    if wandb_logger is not None:
        wandb_logger.log(
            {
                "summary/nmse_mean": result.nmse_mean,
                "summary/nmse_std": result.nmse_std,
                "summary/mean_l0": result.mean_l0,
                "summary/dead_ratio": result.dead_ratio,
                "summary/max_node_share": result.max_node_share,
                "summary/active_cosine_mean": result.active_cosine_mean,
                "summary/decoder_cosine_max": result.decoder_cosine_max,
                "summary/global_step": result.global_step,
            },
            step=result.global_step,
        )

        artifact_files: list[Path] = [trial_root / "config.json", trial_root / "metrics.json"]
        if cfg.logger.log_weights_to_wandb:
            artifact_files.extend(res.checkpoint_path for res in layer_results)
        if cfg.logger.log_optimizer_state_to_wandb:
            artifact_files.extend(
                trial_root / "checkpoints" / f"layer_{res.layer}" / "latest.pt"
                for res in layer_results
            )
        if cfg.logger.log_activations_store_to_wandb:
            artifact_files.extend(cfg.acts_dir / f"layer_{layer}" / "meta.json" for layer in cfg.layers)
        wandb_logger.log_artifact(
            name=(
                f"{cfg.run_id}_{trial_cfg.arch}_{trial_cfg.stage}_{candidate_key}"
                f"_trial_{trial_cfg.trial_id}"
            ),
            artifact_type="sae_trial",
            files=artifact_files,
            aliases=["latest", trial_cfg.stage, candidate_key],
            metadata={
                "run_id": cfg.run_id,
                "arch": trial_cfg.arch,
                "stage": trial_cfg.stage,
                "candidate": candidate_key,
                "trial_id": trial_cfg.trial_id,
                "k": trial_cfg.k,
                "l0_coefficient": trial_cfg.l0_coefficient,
            },
        )

    return result, quality_rows


def run_reference_evaluations(
    cfg: SaeExperimentConfig,
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    device: torch.device,
    *,
    candidate_label: str,
    arch: str,
) -> None:
    for layer in cfg.layers:
        evaluate_baseline_frozen(
            cfg,
            model,
            tokenizer,
            layer,
            device,
            log_path=cfg.run_context.downstream_log_path if cfg.run_context else cfg.log_path,
            run_id=cfg.run_id,
            extra_fields={
                "arch": arch,
                "candidate": candidate_label,
                "stage": "reference",
            },
        )

    evaluate_final_hidden_state(
        cfg,
        model,
        tokenizer,
        device,
        log_path=cfg.run_context.downstream_log_path if cfg.run_context else cfg.log_path,
        run_id=cfg.run_id,
        extra_fields={
            "arch": arch,
            "candidate": candidate_label,
            "stage": "reference",
        },
    )


def _default_trial_cfg(cfg: SaeExperimentConfig, arch: str) -> TrialConfig:
    if arch == "batchtopk":
        return TrialConfig(
            arch="batchtopk",
            stage="stage3",
            trial_id=0,
            epoch_budget=cfg.sae_epochs,
            seed=cfg.seed,
            k=cfg.batchtopk_k,
        )

    if arch == "topk":
        return TrialConfig(
            arch="topk",
            stage="stage3",
            trial_id=0,
            epoch_budget=cfg.sae_epochs,
            seed=cfg.seed,
            k=cfg.batchtopk_k,
        )
    if arch == "jumprelu":
        return TrialConfig(
            arch="jumprelu",
            stage="stage3",
            trial_id=0,
            epoch_budget=cfg.sae_epochs,
            seed=cfg.seed,
            l0_coefficient=cfg.jumprelu_l0_coefficient,
        )
    raise ValueError(f"Unsupported architecture '{arch}'")


def run_all(
    cfg: SaeExperimentConfig,
    *,
    resume: bool = False,
    arch: str = "batchtopk",
) -> list[TrialResult]:
    _ensure_sae_ready(cfg)
    assert cfg.run_context is not None
    config_payload = to_wandb_config(
        {
            "config": cfg,
            "mode": "sae_run_all",
            "resume": resume,
            "arch": arch,
        }
    )
    wandb_logger = WandbRunLogger(
        logger_cfg=cfg.logger,
        run_root=cfg.run_context.run_root,
        run_id=cfg.run_id or "unknown_run",
        config_payload=config_payload,
        job_type="sae_train",
        tags=["sae", "run_all"],
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

        arch_list = list(ARCH_CHOICES) if arch == "both" else [arch]
        results: list[TrialResult] = []

        for arch_name in arch_list:
            trial_cfg = _default_trial_cfg(cfg, arch_name)
            trial_root = cfg.run_context.final_dir / arch_name / "candidate_1"
            result, quality_rows = run_architecture_trial(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                device=device,
                trial_cfg=trial_cfg,
                trial_root=trial_root,
                run_downstream=True,
                candidate_label="candidate_1",
                resume=resume,
                wandb_logger=wandb_logger,
            )
            results.append(result)
            for row in quality_rows:
                append_csv_row(cfg.run_context.quality_log_path, row)

            run_reference_evaluations(
                cfg,
                model,
                tokenizer,
                device,
                candidate_label="candidate_1",
                arch=arch_name,
            )

        save_json(
            cfg.run_context.run_meta_path,
            {
                "run_id": cfg.run_context.run_id,
                "model_name": cfg.model_name,
                "layers": list(cfg.layers),
                "arch": arch,
                "sae_epochs": cfg.sae_epochs,
                "seed": cfg.seed,
                "jumprelu_l0_warm_up_steps": cfg.jumprelu_l0_warm_up_steps,
                "jumprelu_dead_feature_window": cfg.jumprelu_dead_feature_window,
                "jumprelu_sparsity_loss_mode": cfg.jumprelu_sparsity_loss_mode,
                "mlm_data_path": str(cfg.mlm_data_path),
                "local_only": cfg.local_only,
            },
        )
        wandb_logger.log_artifact(
            name=f"{cfg.run_id}_run_all_meta",
            artifact_type="sae_run",
            files=[cfg.run_context.run_meta_path, cfg.run_context.quality_log_path],
            aliases=["latest", "run_all"],
            metadata={"run_id": cfg.run_id, "arch": arch, "mode": "run_all"},
        )
        if results:
            wandb_logger.update_summary(
                {
                    "run_all_nmse_mean_best": min(result.nmse_mean for result in results),
                    "run_all_dead_ratio_max": max(result.dead_ratio for result in results),
                }
            )
        if wandb_logger.run_url is not None:
            print(f"wandb_run_url={wandb_logger.run_url}")
        return results
    finally:
        wandb_logger.finish()
