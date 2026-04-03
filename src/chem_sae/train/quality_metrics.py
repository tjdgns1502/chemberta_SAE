from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerQuality:
    nmse: float
    mean_l0: float
    dead_ratio: float
    max_node_share: float
    active_cosine_mean: float
    decoder_cosine_max: float


def _pairwise_abs_cosine_stats(vectors: torch.Tensor) -> tuple[float, float]:
    if vectors.ndim != 2 or vectors.shape[0] <= 1:
        return 0.0, 0.0

    vectors = vectors.float()
    norms = vectors.norm(dim=1, keepdim=True)
    valid = norms.squeeze(1) > 1e-12
    vectors = vectors[valid]
    norms = norms[valid]
    if vectors.shape[0] <= 1:
        return 0.0, 0.0

    unit = vectors / norms
    cos = (unit @ unit.t()).abs().clamp(0.0, 1.0)
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    values = cos[mask]
    if values.numel() == 0:
        return 0.0, 0.0
    return float(values.mean().item()), float(values.max().item())


def _extract_decoder_latent_vectors(
    model: torch.nn.Module,
    latent_dim: int,
) -> torch.Tensor | None:
    decoder = getattr(model, "decoder", None)
    weight = getattr(decoder, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None

    weight = weight.detach().float().cpu()
    # Linear decoder weight is [d_input, d_latent], so we transpose into [d_latent, d_input].
    if weight.shape[1] == latent_dim:
        return weight.t().contiguous()
    if weight.shape[0] == latent_dim:
        return weight.contiguous()
    return None


@torch.no_grad()
def evaluate_layer_quality(
    model: torch.nn.Module,
    batches,
    device: torch.device,
    *,
    compute_structure_metrics: bool = True,
    similarity_feature_cap: int = 64,
    similarity_sample_cap: int = 4096,
) -> LayerQuality:
    model.eval()
    nmse_vals: list[float] = []
    l0_vals: list[float] = []
    active_counts: torch.Tensor | None = None
    num_items = 0

    sampled_latents: list[torch.Tensor] = []
    sampled_rows = 0

    for batch in batches:
        batch = batch.to(device).float()
        _, latents, recons = model(batch)

        denom = batch.pow(2).mean(dim=1).clamp(min=1e-8)
        nmse = ((recons - batch).pow(2).mean(dim=1) / denom).mean()
        nmse_vals.append(float(nmse.item()))

        l0 = (latents > 0).float().sum(dim=1)
        l0_vals.append(float(l0.mean().item()))

        layer_active = (latents > 0).float().sum(dim=0).cpu()
        active_counts = layer_active if active_counts is None else active_counts + layer_active
        num_items += latents.shape[0]

        if compute_structure_metrics and sampled_rows < similarity_sample_cap:
            keep_rows = min(latents.shape[0], similarity_sample_cap - sampled_rows)
            sampled_latents.append(latents[:keep_rows].detach().cpu().to(torch.float16))
            sampled_rows += keep_rows

    if not nmse_vals:
        return LayerQuality(
            nmse=float("inf"),
            mean_l0=float("inf"),
            dead_ratio=1.0,
            max_node_share=1.0,
            active_cosine_mean=float("inf"),
            decoder_cosine_max=float("inf"),
        )

    assert active_counts is not None
    fire_rate = active_counts / max(1, num_items)
    dead_ratio = float((fire_rate <= 0.0).float().mean().item())

    max_node_share = float("nan")
    active_cosine_mean = float("nan")
    decoder_cosine_max = float("nan")

    if compute_structure_metrics:
        total_fires = float(active_counts.sum().item())
        if total_fires <= 0.0:
            max_node_share = 1.0
        else:
            max_node_share = float(active_counts.max().item() / total_fires)

        if sampled_latents:
            stacked = torch.cat(sampled_latents, dim=0).float()
            feature_freq = (stacked > 0).float().sum(dim=0)
        else:
            stacked = torch.empty((0, active_counts.numel()), dtype=torch.float32)
            feature_freq = active_counts.float()

        active_feature_indices = torch.nonzero(feature_freq > 0, as_tuple=False).squeeze(1)
        if active_feature_indices.numel() == 0:
            top_indices = torch.tensor([], dtype=torch.long)
        elif active_feature_indices.numel() <= similarity_feature_cap:
            top_indices = active_feature_indices
        else:
            top_values = feature_freq[active_feature_indices]
            local_top = torch.topk(top_values, k=similarity_feature_cap, dim=0).indices
            top_indices = active_feature_indices[local_top]

        if top_indices.numel() > 1 and stacked.shape[0] > 0:
            profiles = stacked[:, top_indices].t().contiguous()
            active_cosine_mean, _ = _pairwise_abs_cosine_stats(profiles)
        else:
            active_cosine_mean = 0.0

        decoder_vectors = _extract_decoder_latent_vectors(model, active_counts.numel())
        if decoder_vectors is not None and top_indices.numel() > 1:
            selected = decoder_vectors[top_indices].contiguous()
            _, decoder_cosine_max = _pairwise_abs_cosine_stats(selected)
        else:
            decoder_cosine_max = 0.0

    return LayerQuality(
        nmse=float(sum(nmse_vals) / len(nmse_vals)),
        mean_l0=float(sum(l0_vals) / len(l0_vals)),
        dead_ratio=dead_ratio,
        max_node_share=max_node_share,
        active_cosine_mean=active_cosine_mean,
        decoder_cosine_max=decoder_cosine_max,
    )
