import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import RobertaConfig


def load_state_dict_from_hf(model_name: str, local_only: bool = True):
    snapshot_dir = snapshot_download(repo_id=model_name, local_files_only=local_only)
    safetensors_path = Path(snapshot_dir) / "model.safetensors"
    bin_path = Path(snapshot_dir) / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    return torch.load(str(bin_path), map_location="cpu", weights_only=True)


def load_config_from_hf(model_name: str, local_only: bool = True) -> RobertaConfig:
    snapshot_dir = snapshot_download(repo_id=model_name, local_files_only=local_only)
    cfg_path = Path(snapshot_dir) / "config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    return RobertaConfig(**cfg_dict)
