import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE

class SpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, channels: int=4, image_size: int=64):
        super().__init__()
        
        self.feature_size = image_size // (2 ** 4)

        self.conv = nn.Sequential(
            SpectralNormConv2d(channels, 32, kernel_size=4, stride=2, padding=1),  
            nn.SiLU(inplace=True),
            SpectralNormConv2d(32, 64, kernel_size=4, stride=2, padding=1),        
            nn.SiLU(inplace=True),
            SpectralNormConv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            SpectralNormConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv(x)
        pooled_features = self.pool(features)
        flat = pooled_features.view(pooled_features.size(0), -1)
        validity = self.fc(flat)

        return validity, features


class VAEGAN(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: list[int], 
                 latent_dim: int, 
                 dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.vae = VAE(in_channels, image_size, hidden_dims, latent_dim, dropout)
        self.discriminator = Discriminator(in_channels, image_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.vae.encode(x)
        z = self.vae.reparameterize(mu, log_var)
        x_recon = self.vae.decode(z)

        return x_recon, mu, log_var
