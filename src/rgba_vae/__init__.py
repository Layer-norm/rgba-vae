"""
RGBA Variational Autoencoder (RGBA-VAE)

Author: Faxuan Cai

License: MIT License
"""

from .train import train_vae, train_vaegan, train_vavae
from .vae import VAE
from .vaegan import VAEGAN
from .vavae import VAVAE
from .defaultconfig import DefaultConfig, VAEGANConfig, VAVAEConfig
from .utils import JSONLBase64Dataset

__all__ = [
    "train_vae",
    "train_vaegan",
    "train_vavae",
    "VAE",
    "VAEGAN",
    "VAVAE",
    "DefaultConfig",
    "VAEGANConfig",
    "VAVAEConfig",
    "JSONLBase64Dataset"
]
