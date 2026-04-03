from __future__ import annotations

import torch

from chem_sae.config.experiment import SaeExperimentConfig
from chem_sae.utils.hf import load_config_from_hf, load_state_dict_from_hf

from .roberta_mlm import RobertaForMaskedLM


def build_mlm_model(cfg: SaeExperimentConfig, device: torch.device):
    config = load_config_from_hf(cfg.model_name, local_only=cfg.local_only)
    model = RobertaForMaskedLM(config)
    model.tie_weights()
    state_dict = load_state_dict_from_hf(cfg.model_name, local_only=cfg.local_only)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config

