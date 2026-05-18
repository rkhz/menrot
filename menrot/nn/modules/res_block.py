import torch
import torch.nn as nn
from typing import Optional

__all__ = [
    "ResBlock2d",
    "ResBlock3d"
]


class _ResBlockNd(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        dim: int=None,
        device=None
    ) -> None:
        super().__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"
        
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_activation = pre_activation
        self.dim = dim
        
        # Configure residual mapping based on pre-activation 
        _ResMap = _PreActResMapNd if self.pre_activation else _ResMapNd
        self.residual_map = _ResMap(self.in_channels, self.out_channels, dim=self.dim, device=self.device)
        
        # 
        self._final_activation = None if self.pre_activation else nn.LeakyReLU(0.2, inplace=True)
        
        # Check if projection is needed (when input and output channels differ)
        if in_channels != out_channels:
            _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
            self.shortcut = {
                True:   _Conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False, device=self.device),
                False:  nn.Sequential(
                            _Conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False, device=self.device),
                            nn.GroupNorm(num_groups=out_channels//8, num_channels=out_channels, device=self.device))
            }[self.pre_activation]
        else:
            self.shortcut = nn.Identity(device=self.device)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_activation:
            return self.shortcut(x) + self.residual_map(x)
        else:
            return self._final_activation(self.shortcut(x) + self.residual_map(x))


class ResBlock2d(_ResBlockNd):
    def __init__(self, in_channels: int, out_channels: int, pre_activation: bool=True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=2, device=device)


class ResBlock3d(_ResBlockNd):
    def __init__(self, in_channels: int, out_channels: int, pre_activation: bool=True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=3, device=device)


class _PreActResMapNd(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bottleneck_factor: Optional[float]=1, 
        dim: int=None,
        device=None
    ) -> None:
        super(_PreActResMapNd, self).__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"

        self.dim = dim
        self.device = device 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(bottleneck_factor * out_channels)
        
        _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=self.in_channels//8, num_channels=self.in_channels),
            nn.LeakyReLU(0.2, True),
            _Conv(self.in_channels, self.bottleneck_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
            _Conv(self.bottleneck_channels, self.bottleneck_channels, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
            _Conv(self.bottleneck_channels, self.out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False)
        ).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
            
 
class _ResMapNd(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bottleneck_factor: Optional[float]=1, 
        dim: int=None,
        device=None
    ) -> None:
        super(_ResMapNd, self).__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"
        
        self.dim = dim
        self.device = device 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(bottleneck_factor * out_channels) 
        
        _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
        self.layers = nn.Sequential(
            _Conv(self.in_channels, self.bottleneck_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
            
            _Conv(self.bottleneck_channels, self.bottleneck_channels, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
    
            _Conv(self.bottleneck_channels, self.out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=self.out_channels//8, num_channels=self.out_channels)
        ).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
