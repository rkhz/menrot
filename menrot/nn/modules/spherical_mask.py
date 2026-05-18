# %%
import torch
import torch.nn as nn

__all__ = [
    "SphericalMask"
]


class SphericalMask(nn.Module):
    def __init__(
        self, 
        in_channels, 
        latent_dim,
        radius_fraction=1.0,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        c = (self.latent_dim - 1) / 2
        r2 = (radius_fraction * c) ** 2

        coords = torch.arange(self.latent_dim) - c          # (L,)
        d2 = coords[:, None, None]**2 + coords[None, :, None]**2 + coords[None, None, :]**2  # (L,L,L)

        mask = (d2 <= r2).unsqueeze(0).expand(in_channels, -1, -1, -1).float()  # (C,L,L,L)
        self.register_buffer('mask', mask)

    def forward(self, volume):
        return volume * self.mask


# %%
