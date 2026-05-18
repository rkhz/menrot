import torch
import torch.nn as nn

from menrot.nn import Encode2d, Encode3d, Decode2d, Decode3d, Project2DTo3D, Project3DTo2D, SphericalMask, Rotate3d

__all__ = [
    "NeuralRenderer"
]

# source: https://arxiv.org/abs/2006.07630

class NeuralRenderer(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        #latent_channels: int=None, # TO DO
        #latent_dim: int=None,      # TO DO 
        pre_activation: bool=True, 
        mode: str='bilinear',
        device=None
    ) -> None:
        super().__init__()
        self.device = device
        self.in_channels= in_channels
        self.mode = mode
        
        self.encoder = torch.nn.Sequential(
            Encode2d(self.in_channels, 128, pre_activation=pre_activation),
            Project2DTo3D(128, 32, pre_activation=pre_activation),
            Encode3d(32, 64, pre_activation=pre_activation),
            SphericalMask(64, latent_dim=32)    
        ).to(self.device)

        self.rotate = Rotate3d(self.mode).to(self.device)
        
        self.decoder = torch.nn.Sequential(
            Decode3d(64, 32, pre_activation=pre_activation),
            Project3DTo2D(32, 128, pre_activation=pre_activation),
            Decode2d(128, self.in_channels, pre_activation=pre_activation),
            torch.nn.Sigmoid()
        ).to(self.device)
    
    def forward(self, x, **kwargs):
        z = self.encoder(x)
        z_rotated = self.rotate(z, **kwargs)
        y = self.decoder(z_rotated)
        return y, z, z_rotated
    