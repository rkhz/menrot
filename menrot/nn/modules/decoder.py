import torch
import torch.nn as nn

from .conv_block import ConvBlock2d, ConvBlock3d, ConvTransposeBlock2d, ConvTransposeBlock3d
from .res_block import ResBlock2d, ResBlock3d

__all__ = [
    "Decode2d",
    "Decode3d",
]

# source: https://arxiv.org/abs/2006.07630

class _DecodeNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pre_activation: bool=True,
        dim: int=None,
        device=None
    ) -> None:
        super().__init__()
        assert dim in [2,3], "dim must be 2 (Conv2d) or 3 (Conv3d)"

        self.device = device
        self.in_channels= in_channels
        self.out_channels = out_channels 
        self.pre_activation = pre_activation
        self.dim = dim


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Decode2d(_DecodeNd):
    def __init__(self, in_channels, out_channels, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=2, device=device)
        self.layers = nn.Sequential(
            ResBlock2d(self.in_channels, 128, pre_activation=pre_activation),
            ResBlock2d(128, 128, pre_activation=pre_activation),
            
            ConvBlock2d(128, 256, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
            ResBlock2d(256, 256, pre_activation=pre_activation),
            
            ConvTransposeBlock2d(256, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
            ResBlock2d(128, 128, pre_activation=pre_activation),
            
            ConvTransposeBlock2d(128, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
            ResBlock2d(128, 128, pre_activation=pre_activation),
            
            ConvTransposeBlock2d(128, 64, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation), 
            ResBlock2d(64, 64, pre_activation=pre_activation),
            ResBlock2d(64, 64, pre_activation=pre_activation),

            nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1, padding=0)
        ).to(self.device)

class Decode3d(_DecodeNd):
    def __init__(self, in_channels, out_channels, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=3, device=device)
        self.layers = nn.Sequential(
            ResBlock3d(self.in_channels, 64, pre_activation=pre_activation),
            ResBlock3d(64, 64, pre_activation=pre_activation),
            
            ConvBlock3d(64, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
            ResBlock3d(128, 128, pre_activation=pre_activation),
            ResBlock3d(128, 128, pre_activation=pre_activation),
            
            ConvTransposeBlock3d(128, 32, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
            ResBlock3d(32, 32, pre_activation=pre_activation),
            ResBlock3d(32, 32, pre_activation=pre_activation),

            ConvBlock3d(32, self.out_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
        ).to(self.device)