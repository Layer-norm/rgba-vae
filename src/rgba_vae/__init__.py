"""
RGBA Variational Autoencoder (RGBA-VAE)

Author: Faxuan Cai

License: MIT License
"""

from .train import train_vae
from .vae import VAE
from .defaultconfig import DefaultConfig

__all__ = [
    "train_vae",
    "VAE",
    "DefaultConfig"
]
