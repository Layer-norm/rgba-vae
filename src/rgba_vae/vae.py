import torch
from torch import nn
from torch.nn import functional as F

from .modules.ConNext import ConvNextBlock, GRN

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv_next1 = ConvNextBlock(in_channels, out_channels, dropout)
        self.conv_next2 = ConvNextBlock(out_channels, out_channels, dropout)
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_next1(x)
        x = self.conv_next2(x)
        x = self.downsample(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv_next1 = ConvNextBlock(in_channels, out_channels, dropout)
        self.conv_next2 = ConvNextBlock(out_channels, out_channels, dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv_next1(x)
        x = self.conv_next2(x)
        return x

# Encoder: q(z|x)
class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: list[int], 
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
            self.encoder_layers.append(DownBlock(input_dim, h_dim, dropout=dropout))
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
class Decoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: list[int], 
                 latent_dim: int,
                 num_groups: int,
                 dropout: float = 0.1
        ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dims[-1]

        self.feature_map_size = image_size // (2 ** len(hidden_dims))
        self.flattened_dim = int(hidden_dims[-1] * (self.feature_map_size ** 2))
        reversed_hidden_dims = hidden_dims[::-1] + [in_channels]

        #decoder
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder_layers = nn.ModuleList()


        decoder_input_dim = reversed_hidden_dims[0]

        for h_dim in reversed_hidden_dims[1:]:
            self.decoder_layers.append(UpBlock(decoder_input_dim, h_dim, dropout=dropout))
            decoder_input_dim = h_dim
        
        self.final_fc = nn.Sequential(
            nn.Linear(image_size, image_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_decoder(x)

        x = x.view(-1, self.hidden_dim, self.feature_map_size, self.feature_map_size)

        for layer in self.decoder_layers:
            x = layer(x)
        
        x = self.final_fc(x)

        return x


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 image_size: int, 
                 hidden_dims: list[int], 
                 latent_dim: int, 
                 dropout: float = 0.1
        ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.encode = Encoder(
            self.in_channels, 
            self.image_size, 
            self.hidden_dims, 
            self.latent_dim, 
            self.dropout
        )

        self.decode = Decoder(
            self.in_channels, 
            self.image_size, 
            self.hidden_dims, 
            self.latent_dim, 
            self.dropout
        )
    
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
        

