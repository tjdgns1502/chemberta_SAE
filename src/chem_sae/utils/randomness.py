import random
from typing import Any, Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = [
            torch.cuda.get_rng_state(device_idx)
            for device_idx in range(torch.cuda.device_count())
        ]
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        for device_idx, rng_state in enumerate(cuda_state):
            if device_idx >= torch.cuda.device_count():
                break
            torch.cuda.set_rng_state(rng_state, device_idx)
