"""Minimal SAE primitives vendored for this project.

This module contains only the components used by the current ChemBERTa SAE experiments:
- Autoencoder
- TopK / BatchTopK activation
- autoencoder_loss
"""

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batchtopk_ext import BatchTopK

def ln(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer-normalize per sample without learnable affine parameters."""
    mu = x.mean(dim=-1, keepdim=True)
    centered = x - mu
    std = centered.std(dim=-1, keepdim=True)
    normalized = centered / (std + eps)
    return normalized, mu, std


class Autoencoder(nn.Module):
    """Sparse autoencoder used for attention activation reconstruction."""

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        activation: Callable = nn.ReLU(),
        tied: bool = False,
        normalize: bool = False,
        apply_b_dec_to_input: bool = True,
    ) -> None:
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        self.normalize = normalize
        self.apply_b_dec_to_input = apply_b_dec_to_input

        if tied:
            self.decoder: nn.Module = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)

        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    @property
    def b_dec(self) -> nn.Parameter:
        return self.pre_bias

    @property
    def b_enc(self) -> nn.Parameter:
        return self.latent_bias

    @property
    def W_enc(self) -> torch.Tensor:
        return self.encoder.weight.t()

    @property
    def W_dec(self) -> torch.Tensor:
        return self.decoder.weight.t()

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        bias = self.pre_bias if self.apply_b_dec_to_input else 0.0
        centered = x - bias
        return F.linear(
            centered, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, {}
        normalized, mu, std = ln(x)
        return normalized, {"mu": mu, "std": std}

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] | None = None) -> torch.Tensor:
        recon = self.decoder(latents) + self.pre_bias
        if self.normalize:
            if info is None:
                raise ValueError("`info` is required when normalize=True.")
            recon = recon * info["std"] + info["mu"]
        return recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recon = self.decode(latents, info)

        # Track consecutive steps since each latent was last non-zero.
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recon

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor | str | dict[str, torch.Tensor]], strict: bool = True
    ) -> "Autoencoder":
        local = dict(state_dict)
        n_latents, d_model = local["encoder.weight"].shape  # type: ignore[index]

        activation_class_name = local.pop("activation", None)  # type: ignore[arg-type]
        activation_state_dict = local.pop("activation_state_dict", {})  # type: ignore[arg-type]
        if activation_class_name is None:
            activation = nn.ReLU()
            normalize = False
        else:
            activation_class = ACTIVATION_CLASSES.get(str(activation_class_name), nn.ReLU)
            if hasattr(activation_class, "from_state_dict"):
                activation = activation_class.from_state_dict(activation_state_dict, strict=strict)  # type: ignore[attr-defined]
            else:
                activation = activation_class()
                if hasattr(activation, "load_state_dict"):
                    activation.load_state_dict(activation_state_dict, strict=strict)  # type: ignore[attr-defined]
            normalize = str(activation_class_name) in {"TopK", "BatchTopK"}

        autoencoder = cls(n_latents, d_model, activation=activation, normalize=normalize)
        autoencoder.load_state_dict(local, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    @torch.no_grad()
    def log_histograms(self) -> dict[str, Any]:
        """SAELens-compatible histogram payload for periodic eval logging."""
        decoder_weight = getattr(self.decoder, "weight", None)
        if not isinstance(decoder_weight, torch.Tensor) or decoder_weight.ndim != 2:
            return {}
        w_dec_norm_dist = decoder_weight.detach().float().norm(dim=0).cpu().numpy()
        return {"weights/W_dec_norms": w_dec_norm_dist}

    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        decoder_weight = getattr(self.decoder, "weight", None)
        if not isinstance(decoder_weight, torch.Tensor) or decoder_weight.ndim != 2:
            return
        # Linear decoder uses [d_input, d_latent], so per-latent norms are over dim=0.
        w_dec_norms = decoder_weight.norm(dim=0).clamp(min=1e-8)
        self.decoder.weight.data = self.decoder.weight.data / w_dec_norms.unsqueeze(0)
        self.encoder.weight.data = self.encoder.weight.data * w_dec_norms.unsqueeze(1)
        self.latent_bias.data = self.latent_bias.data * w_dec_norms


class TiedTranspose(nn.Module):
    """Decoder that reuses encoder weights via transpose."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear.bias is not None:
            raise ValueError("TiedTranspose expects encoder bias to be None.")
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias


class TopK(nn.Module):
    """Keep only top-k latent values per sample and zero the rest."""

    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk.indices, values)
        return out

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + "k"] = self.k
        sd[prefix + "postact_fn"] = self.postact_fn.__class__.__name__
        return sd

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor | int | str], strict: bool = True) -> "TopK":
        del strict
        k = int(state_dict["k"])
        postact_name = str(state_dict["postact_fn"])
        postact_fn = ACTIVATION_CLASSES[postact_name]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATION_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
    "BatchTopK": BatchTopK,
}


def normalized_mean_squared_error(
    reconstruction: torch.Tensor, original_input: torch.Tensor
) -> torch.Tensor:
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1)
        / (original_input**2).mean(dim=1).clamp(min=1e-8)
    ).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor, original_input: torch.Tensor
) -> torch.Tensor:
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()


def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> torch.Tensor:
    return mse_reconstruction_loss(reconstruction, original_input) + (
        l1_sparsity_loss(latent_activations) * l1_weight
    )


def mse_reconstruction_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    return ((reconstruction - original_input) ** 2).sum(dim=-1).mean()


def l1_sparsity_loss(latent_activations: torch.Tensor) -> torch.Tensor:
    return latent_activations.abs().sum(dim=-1).mean()
