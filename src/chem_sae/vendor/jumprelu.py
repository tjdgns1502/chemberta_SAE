"""SAELens-backed JumpReLU SAE adapter for ChemBERTa training loops.

This module intentionally avoids reimplementing JumpReLU math.
It imports the canonical SAELens implementation and exposes a thin
compatibility layer used by the existing ChemBERTa code.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Literal

import torch

from .sae_core import l1_sparsity_loss, mse_reconstruction_loss


# Ensure local SAELens source is importable without requiring global install.
_SAELENS_ROOT = Path(__file__).resolve().parents[4] / "SAELens"
if _SAELENS_ROOT.exists() and str(_SAELENS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAELENS_ROOT))

from sae_lens.saes.jumprelu_sae import (  # type: ignore  # noqa: E402
    JumpReLU,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    Step,
)
from sae_lens.saes.sae import TrainStepInput  # type: ignore  # noqa: E402


# Backward-compatible names used by existing tests/contracts.
_StepFn = Step
_JumpReLUFn = JumpReLU


class _DecoderView:
    """Compatibility shim exposing `.weight` like torch.nn.Linear decoder."""

    def __init__(self, model: "JumpReLUAutoencoder") -> None:
        self._model = model

    @property
    def weight(self) -> torch.Tensor:
        # SAELens W_dec shape is [d_sae, d_in]; legacy code expects [d_in, d_sae].
        return self._model.W_dec.t()


class JumpReLUAutoencoder(JumpReLUTrainingSAE):
    """Compatibility wrapper around SAELens JumpReLUTrainingSAE."""

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        threshold_init: float = 0.01,
        bandwidth: float = 0.05,
        normalize: bool = True,
        sparsity_loss_mode: Literal["step", "tanh"] = "step",
        tanh_scale: float = 4.0,
        pre_act_loss_coefficient: float | None = None,
    ) -> None:
        cfg = JumpReLUTrainingSAEConfig(
            d_in=n_inputs,
            d_sae=n_latents,
            dtype="float32",
            device="cpu",
            apply_b_dec_to_input=True,
            normalize_activations="layer_norm" if normalize else "none",
            reshape_activations="none",
            decoder_init_norm=None,
            jumprelu_init_threshold=threshold_init,
            jumprelu_bandwidth=bandwidth,
            jumprelu_sparsity_loss_mode=sparsity_loss_mode,
            l0_coefficient=1.0,
            l0_warm_up_steps=0,
            pre_act_loss_coefficient=pre_act_loss_coefficient,
            jumprelu_tanh_scale=tanh_scale,
        )
        super().__init__(cfg, use_error_term=False)
        self.register_buffer(
            "stats_last_nonzero",
            torch.zeros(n_latents, dtype=torch.long, device=self.W_dec.device),
        )
        self._decoder_view = _DecoderView(self)

    @property
    def decoder(self) -> _DecoderView:
        return self._decoder_view

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts, {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_acts, hidden_pre = self.encode_with_hidden_pre(x)
        recons = self.decode(feature_acts)

        # Track consecutive steps since each latent was last non-zero.
        self.stats_last_nonzero *= (feature_acts == 0).all(dim=0).long()
        self.stats_last_nonzero += 1
        return hidden_pre, feature_acts, recons


@contextmanager
def _temporary_cfg_overrides(
    model: JumpReLUAutoencoder,
    *,
    sparsity_loss_mode: Literal["step", "tanh"] | None,
    tanh_scale: float | None,
    pre_act_loss_coefficient: float | None,
):
    old_mode = model.cfg.jumprelu_sparsity_loss_mode
    old_tanh_scale = model.cfg.jumprelu_tanh_scale
    old_pre_act = model.cfg.pre_act_loss_coefficient

    if sparsity_loss_mode is not None:
        model.cfg.jumprelu_sparsity_loss_mode = sparsity_loss_mode
    if tanh_scale is not None:
        model.cfg.jumprelu_tanh_scale = tanh_scale
    if pre_act_loss_coefficient is not None:
        model.cfg.pre_act_loss_coefficient = pre_act_loss_coefficient

    try:
        yield
    finally:
        model.cfg.jumprelu_sparsity_loss_mode = old_mode
        model.cfg.jumprelu_tanh_scale = old_tanh_scale
        model.cfg.pre_act_loss_coefficient = old_pre_act


def jumprelu_loss_with_details(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    hidden_pre: torch.Tensor,
    l0_coefficient: float,
    l1_weight: float,
    model: JumpReLUAutoencoder,
    *,
    sparsity_loss_mode: Literal["step", "tanh"] | None = None,
    tanh_scale: float | None = None,
    pre_act_loss_coefficient: float | None = None,
    dead_neuron_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with _temporary_cfg_overrides(
        model,
        sparsity_loss_mode=sparsity_loss_mode,
        tanh_scale=tanh_scale,
        pre_act_loss_coefficient=pre_act_loss_coefficient,
    ):
        mse_loss = mse_reconstruction_loss(reconstruction, original_input)
        l1_loss = l1_sparsity_loss(latent_activations) * l1_weight

        aux_losses = model.calculate_aux_loss(
            step_input=TrainStepInput(
                sae_in=original_input,
                coefficients={"l0": float(l0_coefficient)},
                dead_neuron_mask=dead_neuron_mask,
                n_training_steps=0,
            ),
            feature_acts=latent_activations,
            hidden_pre=hidden_pre,
            sae_out=reconstruction,
        )

        if not isinstance(aux_losses, dict):
            aux_losses = {"l0_loss": aux_losses}

        l0_loss = aux_losses.get("l0_loss", original_input.new_tensor(0.0))
        pre_act_loss = aux_losses.get("pre_act_loss", original_input.new_tensor(0.0))

        total = mse_loss + l1_loss
        for value in aux_losses.values():
            total = total + value

        return total, {
            "mse_loss": mse_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "l0_loss": l0_loss.detach(),
            "pre_act_loss": pre_act_loss.detach(),
        }


def jumprelu_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    hidden_pre: torch.Tensor,
    l0_coefficient: float,
    l1_weight: float,
    model: JumpReLUAutoencoder,
    *,
    sparsity_loss_mode: Literal["step", "tanh"] | None = None,
    tanh_scale: float | None = None,
    pre_act_loss_coefficient: float | None = None,
    dead_neuron_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    total, _ = jumprelu_loss_with_details(
        reconstruction=reconstruction,
        original_input=original_input,
        latent_activations=latent_activations,
        hidden_pre=hidden_pre,
        l0_coefficient=l0_coefficient,
        l1_weight=l1_weight,
        model=model,
        sparsity_loss_mode=sparsity_loss_mode,
        tanh_scale=tanh_scale,
        pre_act_loss_coefficient=pre_act_loss_coefficient,
        dead_neuron_mask=dead_neuron_mask,
    )
    return total
