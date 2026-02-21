import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE

class Discriminator(nn.Module):
    def __init__(self, channels: int=4, image_size: int=64):
        super().__init__()
        
        self.feature_size = image_size // (2 ** 3)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),  
            nn.Mish(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),        
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),       
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * self.feature_size * self.feature_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
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
