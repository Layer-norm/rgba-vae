import torch
from dataclasses import dataclass, field


from typing import List

@dataclass
class DefaultConfig:
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    latent_dim: int = 256
    in_channels: int = 4  # RGBA channels

    # Training parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    batch_size: int = 16
    save_every_n_epochs: int = 10
    dropout: float = 0.1
    checkpoint_dir: str = "checkpoints"

    # Data preprocessing
    image_size: int = 64
    patch_size: int = 8