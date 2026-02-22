import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE
from .vaegan import Discriminator

class VisionFoundationAlignment(nn.Module):
    def __init__(self, 
                 latent_dim: int = 256, 
                 vf_feature_dim: int = 768, 
                 margin: float = 0.2
        ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.vf_feature_dim = vf_feature_dim
        self.margin = margin

        self.proj= nn.Linear(latent_dim, vf_feature_dim)

        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, vf_features: torch.Tensor):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
    
        z_projected = self.proj(z)

        z_projected = F.normalize(z_projected, p=2, dim=1)
        vf_features = F.normalize(vf_features, p=2, dim=1)

        vf_sim = self.cos_sim(z_projected, vf_features)

        return vf_sim


class VAVAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: list[int], 
                 latent_dim: int, 
                 vf_feature_dim: int,
                 use_refinement: bool,
                 refinement_blocks: int = 2,
                 dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.vae = VAE(in_channels, image_size, hidden_dims, latent_dim, use_refinement, refinement_blocks, dropout)
        self.visionalignment = VisionFoundationAlignment(latent_dim, vf_feature_dim)

        self.discriminator = Discriminator(in_channels, image_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.vae.encode(x)
        z = self.vae.reparameterize(mu, log_var)
        x_recon = self.vae.decode(z)

        return x_recon, mu, log_var