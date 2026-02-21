import torch
from dataclasses import dataclass, field


# from typing import List

@dataclass
class DefaultConfig:
    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    latent_dim: int = 256
    in_channels: int = 4  # 4 channels for RGBA images

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

@dataclass
class VAEGANConfig:
    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    latent_dim: int = 256
    in_channels: int = 4  # 4 channels for RGBA images

    # Training parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    optimizer: str = "adamw"

    learning_rate: float = 1e-3
    gan_learning_rate: float = 1e-3

    num_epochs: int = 1000
    batch_size: int = 16
    save_every_n_epochs: int = 10
    dropout: float = 0.1
    checkpoint_dir: str = "checkpoints"

    # Data preprocessing
    image_size: int = 64

@dataclass
class VAVAEConfig:
    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    latent_dim: int = 256
    vf_feature_dim: int = 768  # Dimension of dinov2-base
    vf_model_name: str="facebook/dinov2-base"
    in_channels: int = 4  # 4 channels for RGBA images

    # Training parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_deterministic: bool = False

    optimizer: str = "adamw"

    learning_rate: float = 2e-4
    align_learning_rate: float = 1e-3
    gan_learning_rate: float = 1e-3

    num_epochs: int = 1000
    batch_size: int = 16
    save_every_n_epochs: int = 10
    dropout: float = 0.1
    checkpoint_dir: str = "checkpoints"

    # Loss weights
    beta_recon: float = 1.0
    beta_kl: float = 0.01
    beta_vf: float = 1.0

    # Data preprocessing
    image_size: int = 64