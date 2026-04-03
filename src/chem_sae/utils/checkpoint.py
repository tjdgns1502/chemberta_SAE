from pathlib import Path

import torch


def write_chunk(layer_dir: Path, chunk_idx: int, tensor: torch.Tensor) -> Path:
    layer_dir.mkdir(parents=True, exist_ok=True)
    path = layer_dir / f"chunk_{chunk_idx:05d}.pt"
    torch.save(tensor, path)
    return path


def list_chunks(layer_dir: Path) -> list[Path]:
    return sorted(layer_dir.glob("chunk_*.pt"))


def latest_checkpoint(path: Path) -> Path | None:
    if not path.exists():
        return None
    latest = path / "latest.pt"
    if latest.exists():
        return latest
    ckpts = sorted(path.glob("checkpoint_step_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

