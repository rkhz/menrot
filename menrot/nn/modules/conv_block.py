import torch
import torch.nn as nn

__all__ = [
    "ConvBlock2d",
    "ConvBlock3d",
    "ConvTransposeBlock2d",
    "ConvTransposeBlock3d"
]

# source: https://arxiv.org/abs/2006.07630

class _ConvBlockNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pre_activation: bool=True,
        dim: int=None,
        device=None
    ) -> None:

        super().__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pre_activation = pre_activation
        self.dim = dim

        _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]

        if self.pre_activation:
            self.layers = nn.Sequential( 
                nn.GroupNorm(num_groups=self.in_channels//8 if self.in_channels//8 else 1, num_channels=self.in_channels),
                nn.LeakyReLU(0.2, True),
                _Conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            )
        else:
            self.layers = nn.Sequential(
                _Conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
                nn.GroupNorm(num_groups=self.out_channels//8, num_channels=self.out_channels),
                nn.LeakyReLU(0.2, True)
            )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ConvBlock2d(_ConvBlockNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, pre_activation, dim=2, device=device)


class ConvBlock3d(_ConvBlockNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, pre_activation, dim=3, device=device)


class _ConvTransposeBlockNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pre_activation: bool=True,
        dim: int=None,
        device=None
    ) -> None:

        super().__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pre_activation = pre_activation
        self.dim = dim

        _ConvTranspose = {2:  nn.ConvTranspose2d, 3:  nn.ConvTranspose3d}[self.dim]

        if self.pre_activation:
            self.layers = nn.Sequential(
                nn.GroupNorm(num_groups=self.in_channels//8, num_channels=self.in_channels),
                nn.LeakyReLU(0.2, True),
                _ConvTranspose(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            )
        else:
            self.layers = nn.Sequential(
                _ConvTranspose(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
                nn.GroupNorm(num_groups=self.out_channels//8, num_channels=self.out_channels),
                nn.LeakyReLU(0.2, True)
            )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ConvTransposeBlock2d(_ConvTransposeBlockNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, pre_activation, dim=2, device=device)


class ConvTransposeBlock3d(_ConvTransposeBlockNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, pre_activation, dim=3, device=device)