"""BatchTopK activation mirrored from SAELens behavior."""

from __future__ import annotations

import torch
import torch.nn as nn


class BatchTopK(nn.Module):
    """Keep k active latents on average across the full batch."""

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = float(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x.relu()
        flat_acts = acts.flatten()
        # Match SAELens: k is average active features per sample over batch.
        num_samples = acts.shape[:-1].numel()
        k_total = int(self.k * num_samples)
        acts_topk_flat = torch.topk(flat_acts, k_total, dim=-1)
        return (
            torch.zeros_like(flat_acts)
            .scatter(-1, acts_topk_flat.indices, acts_topk_flat.values)
            .reshape(acts.shape)
        )

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + "k"] = self.k
        return sd

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor | float],
        strict: bool = True,
    ) -> "BatchTopK":
        del strict
        k = float(state_dict["k"])
        return cls(k=k)
