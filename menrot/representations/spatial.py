import torch

import eqnr
from menrot.utils.checkpoint import Snapshot

__all__ = [
    "SpatialRep",
]

class SpatialRep:
    def __init__(self, snapshot_path, device=None):
        # load snapshot
        print(f'EqNR - Loading: {snapshot_path}')
        self.snapshot = Snapshot.load_from(snapshot_path)
        self.model_state = self.snapshot.model_state
        
        # Instantiate model and load weights on CPU
        self.model = eqnr.nn.NeuralRenderer(in_channels=3, pre_activation=True)
        self.model.load_state_dict(self.model_state)

        # Move to device
        self.device = torch.device(device)
        self.model.to(self.device)

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        

    def __call__(self, x, azimuth=None, rot_angle=None):
        with torch.no_grad():
            x = x.to(self.device)
            if rot_angle is not None:
                rot_matrix = eqnr.nn.functional.rotation_matrix_source_to_target(
                    azimuth_source=torch.zeros(x.shape[0], device=self.device), elevation_source=25.0*torch.ones(x.shape[0], device=self.device), 
                    azimuth_target=rot_angle.to(self.device) , elevation_target=25.0*torch.ones(x.shape[0], device=self.device)
                )
                return eqnr.nn.functional.rotate3d(self.model.encoder(x), rot_matrix)
            elif azimuth is not None:
                x_rot, _, z_rot = self.model(
                    x, 
                    azimuth=azimuth.to(self.device), 
                    elevation=25.0*torch.ones(x.shape[0], device=self.device)
                )
                return z_rot
            else: 
                return self.model.encoder(x)
        