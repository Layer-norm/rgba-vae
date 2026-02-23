'''
ref: https://github.com/facebookresearch/ConvNeXt-V2
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}
'''
import torch
from torch import nn
from torch.nn import functional as F


class GRN(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 eps: float = 1e-6
        ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

class ConvNextBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dropout: float = 0.1
        ) -> None:
        super().__init__()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.dw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.ln = nn.LayerNorm(out_channels)
        self.pw_conv1 = nn.Linear(out_channels, 4 * out_channels)
        self.act = nn.SiLU(inplace=True)
        self.grn = GRN(4 * out_channels) 
        self.pw_conv2 = nn.Linear(4 * out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw_conv(x)
        h = h.permute(0, 2, 3, 1) # BCHW -> BHWC
        h = self.ln(h) # channel last for layernorm
        h = self.pw_conv1(h)
        h = self.act(h)
        # h = self.grn(h)  # grn will unstable vae training (KL loss), remove it for now
        h = self.pw_conv2(h)
        h = h.permute(0, 3, 1, 2) # BHWC -> BCHW
        h = self.dropout(h)
        return h + self.skip(x)

if __name__ == "__main__":
    #test grn
    print("Testing GRN...")
    input_x = torch.randn([16,64,32,32])

    grn = GRN(64)

    out_x = grn(input_x.permute(0, 2, 3, 1)) # BCHW -> BHCW

    assert out_x.shape == (16, 32, 32, 64)

    #test convnext block
    print("Testing ConvNextBlock...")

    skip = nn.Conv2d(64, 128, kernel_size=1) if 64 != 128 else nn.Identity()

    skip_x = skip(input_x)
    assert skip_x.shape == (16, 128, 32, 32)

    dw_conv = nn.Conv2d(64, 128, kernel_size=7, padding=3)

    dw_x = dw_conv(input_x)
    assert dw_x.shape == (16, 128, 32, 32)


    groups = min(8, 128)
    norm = nn.GroupNorm(groups, 128)

    norm_x = norm(dw_x)

    assert norm_x.shape == (16, 128, 32, 32)

    pw_conv1 = nn.Linear(128, 4 * 128)

    pw_x = pw_conv1(norm_x.permute(0, 2, 3, 1))

    assert pw_x.shape == (16, 32, 32, 4 * 128)

    pw_conv2 = nn.Linear(4 * 128, 128)

    pw_x = pw_conv2(pw_x)

    assert pw_x.shape == (16, 32, 32, 128)

    conv_next_block = ConvNextBlock(64, 128, num_groups=8)
    out_x = conv_next_block(input_x)

    assert out_x.shape == (16, 128, 32, 32)