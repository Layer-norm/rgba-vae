import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, List

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


# Encoder: q(z|x)
class Encoder(nn.modules):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: List[int], 
                 latent_dim:int, 
                 dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dims[-1]
        self.latent_dim = latent_dim

        # encoder
        self.encoder_layers = nn.ModuleList()
        input_dim = in_channels
        
        for h_dim in hidden_dims:
            self.encoder_layers.append(DoubleConv(input_dim, h_dim, dropout=dropout))
            self.encoder_layers.append(nn.MaxPool2d(2))
            input_dim = h_dim
        
        final_feature_map_size = image_size // (2 ** len(hidden_dims))
        self.flattened_dim = int(hidden_dims[-1] * (final_feature_map_size ** 2))

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.encoder_layers:
            x = layer(x)

        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


# Decoder/Generator: p(x|z)
class Decoder(nn.modules):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: List[int], 
                 latent_dim: int, 
                 dropout: float = 0.1
        ) -> None:

        self.feature_map_size = image_size // (2 ** len(hidden_dims))
        self.flattened_dim = int(hidden_dims[-1] * (self.feature_map_size ** 2))
        reversed_hidden_dims = hidden_dims[::-1] + [in_channels]

        #decoder
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder_layers = nn.ModuleList()
        decoder_input_dim = reversed_hidden_dims[0]

        for h_dim in reversed_hidden_dims[1:]:
            self.decoder_layers.append(DoubleConv(decoder_input_dim, h_dim, dropout=dropout))
            self.decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            decoder_input_dim = h_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_decoder(x)

        x = x.view(-1, self.hidden_dim, self.feature_map_size, self.feature_map_size)

        for layer in self.decoder_layers:
            x = layer(x)

        return torch.sigmoid(x)


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: List[int], 
                 latent_dim: int, 
                 dropout: float = 0.1
        ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dims[-1]
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.encode = Encoder(self.in_channels, self.image_size, self.latent_dim, self.dropout)
        self.decorde = Decoder(self.in_channels, self.image_size, self.latent_dim, self.dropout)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
        

