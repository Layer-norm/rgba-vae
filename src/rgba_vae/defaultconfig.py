from dataclasses import dataclass
from typing import List

@dataclass
class DefaultConfig:
    # Model architecture
    hidden_dim: List[int] = [64, 128, 256, 256]
    latent_dim: int = 128
    num_channels: int = 4  # RGBA channels

    # Training parameters
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 16

    # Data preprocessing
    image_size: int = 64
    patch_size: int = 8