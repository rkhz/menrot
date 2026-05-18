import torch.nn as nn
import einops.layers.torch as einops_nn

from .conv_block import ConvBlock2d

__all__ = [
    "Project2DTo3D",
    "Project3DTo2D"
]

# source: https://arxiv.org/abs/2006.07630

class Project2DTo3D(nn.Module):
    """
    Sources: https://arxiv.org/abs/1806.06575.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        device=None
    ) -> None:
        super().__init__()
        
        self.device = device
        self.in_channels= in_channels 
        self.out_channels = out_channels 
        
        self.layers = nn.Sequential(
            ConvBlock2d(self.in_channels, 2*self.in_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            ConvBlock2d(2*self.in_channels, 4*self.in_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            ConvBlock2d(4*self.in_channels, 8*self.in_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            einops_nn.Rearrange('b (c d) h w -> b c d h w', c=self.out_channels),
        )

    def forward(self, x):
        return self.layers(x)



class Project3DTo2D(nn.Module):
    """
    Sources: https://arxiv.org/abs/1806.06575.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        device=None
    ) -> None:
        super().__init__()
        
        self.device = device
        self.in_channels= in_channels 
        self.out_channels = out_channels 
        
        self.layers = nn.Sequential(
            einops_nn.Rearrange('b c d h w -> b (c d) h w', c=self.in_channels),
            ConvBlock2d(8*self.out_channels, 4*self.out_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),  
            ConvBlock2d(4*self.out_channels, 2*self.out_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),    
            ConvBlock2d(2*self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation)    
        )

    def forward(self, x):
        return self.layers(x)
