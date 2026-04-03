"""Vendored external logic used by this project.

MolNet utilities are imported lazily to avoid pulling heavy dependencies
when only SAE core components are needed.
"""

from .jumprelu import JumpReLUAutoencoder, jumprelu_loss, jumprelu_loss_with_details
from .sae_core import Autoencoder, BatchTopK, TopK, autoencoder_loss


def load_molnet_dataset(*args, **kwargs):
    from .molnet_loader import load_molnet_dataset as _load_molnet_dataset

    return _load_molnet_dataset(*args, **kwargs)


def get_dataset_info(name: str):
    from .molnet_loader import get_dataset_info as _get_dataset_info

    return _get_dataset_info(name)


def get_molnet_directory():
    from .molnet_loader import MOLNET_DIRECTORY

    return MOLNET_DIRECTORY


__all__ = [
    "Autoencoder",
    "BatchTopK",
    "TopK",
    "JumpReLUAutoencoder",
    "autoencoder_loss",
    "jumprelu_loss",
    "jumprelu_loss_with_details",
    "load_molnet_dataset",
    "get_dataset_info",
    "get_molnet_directory",
]
