import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE


class Discriminator(nn.Module):
    def __init__(self, channels: int=4):
        super().__init__()
        self.conv = nn.Sequential(
            # Input: (batch, channels, 32, 32)
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),        # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),       # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        validity = self.fc(flat)

        return validity


class VAEGAN(nn.Module):
    def __init__(self, vae: VAE, discriminator: Discriminator):
        super().__init__()
        self.vae = vae
        self.discriminator = discriminator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.vae.encode(x)
        z = self.vae.reparameterize(mu, log_var)
        x_recon = self.vae.decode(z)

        # gan
        real_validity = self.discriminator(x)
        fake_validity = self.discriminator(x_recon.detach())

        return x_recon, mu, log_var, real_validity, fake_validity
