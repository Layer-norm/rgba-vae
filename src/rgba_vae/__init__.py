"""
RGBA Variational Autoencoder (RGBA-VAE)

Author: Faxuan Cai

License: MIT License
"""

from .train import train_vae, train_vaegan
from .vae import VAE
from .vaegan import VAEGAN
from .defaultconfig import DefaultConfig, VAEGANConfig
from .utils import extract_base64_image_data

__all__ = [
    "train_vae",
    "train_vaegan",
    "VAE",
    "VAEGAN",
    "DefaultConfig",
    "VAEGANConfig",
    "extract_base64_image_data"
]
