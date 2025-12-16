import torch
import einops.layers.torch as einops_nn

from menrot.utils.checkpoint import Snapshot
from menrot.models import VisionSymbolicModel

__all__ = [
    "SymbolicRep",
    "NoSymbolicRep"
]

class SymbolicRep:
    def __init__(self, snapshot_path, return_logits=True, device=None):
        # load snapshot
        print(f'VSM - Loading: {snapshot_path}')
        self.snapshot = Snapshot.load_from(snapshot_path)
        self.model_state = self.snapshot.model_state
        
        # Instantiate model and load weights on CPU
        self.model = VisionSymbolicModel(model_config=self.snapshot.model_config)
        self.model.load_state_dict(self.model_state)

        # Move to device
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.return_logits = return_logits
        
    def __call__(self, x):
        with torch.no_grad():
            if self.return_logits:
                return self.model( x.to(self.device) )[1] 
            else:
                return self.model( x.to(self.device) )[0] 
    
            
class NoSymbolicRep:
    def __init__(self, snapshot_path, conv_version=True, only_patch=True, device=None):
        torch.manual_seed(30) # seed for same parameters
        if snapshot_path is None:
            print(f'VSM (Full Ablation) - Loading: no snapshot')
            self.model = torch.nn.Sequential(
                torch.nn.Conv3d(64, 54, kernel_size=32, stride=1, padding=0),
                einops_nn.Rearrange('b e 1 1 1-> b e', e=54)
            )
            
        else:
            # load snapshot
            print(f"VSM (No symbo. - {'Keep patch' if only_patch else 'keep ViT'}) - Loading: {snapshot_path}")
            self.snapshot = Snapshot.load_from(snapshot_path)

            # Instantiate model and load weights on CPU
            vsm_config = self.snapshot.model_config
            vsm_model = VisionSymbolicModel(model_config=vsm_config)
            vsm_model.load_state_dict(self.snapshot.model_state)

            f = vsm_config['frame_size']//vsm_config['frame_patch_size']
            h = w = vsm_config['frame_size']//vsm_config['patch_size']
            
            if conv_version:
                vsm_model.encoder.pool = 'spatial' # we don t keep the cls
                self.model = torch.nn.Sequential(
                    vsm_model.encoder.to_patch_embed if only_patch else vsm_model.encoder,
                    einops_nn.Rearrange('b (f h w) e -> b e f h w', e=vsm_config['embed_dim'], f=f, h=h, w=w ),
                    torch.nn.Conv3d(vsm_config['embed_dim'], 54, kernel_size=f, stride=1, groups=1),
                    einops_nn.Rearrange('b e 1 1 1-> b e', e=54)
                )
            else:
                self.model = torch.nn.Sequential(
                    vsm_model.encoder.to_patch_embed,
                    einops_nn.Rearrange('b t e -> b (t e)', e=vsm_config['embed_dim']),
                    torch.nn.Linear(vsm_config['embed_dim']*f*h*w, 54)
                ) 
                
        # Move to device
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def __call__(self, x):
        with torch.no_grad():
            return self.model(x)
        