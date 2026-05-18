import torch
import menrot.nn.functional as menrot_F

class Rotate3d(torch.nn.Module):
    def __init__(self, mode: str='bilinear'):
        super(Rotate3d, self).__init__()
        self.mode = mode
        self._pairwise_swap_indices = None

        
    def forward(self, volume: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.training:
            return self._train_forward(volume, kwargs['azimuth'], kwargs['elevation'])
        else:
            return self._eval_forward(volume, **kwargs)
        
        
    def _train_forward(self, volume: torch.Tensor, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        if self._pairwise_swap_indices is None or len(self._pairwise_swap_indices) != volume.shape[0]:
                self._pairwise_swap_indices = [idx^1 for idx in range(volume.shape[0])]
                
        rotation_matrix = menrot_F.rotation_matrix_source_to_target(
            azimuth, elevation, self._pairwise_swap(azimuth), self._pairwise_swap(elevation)
        )
        
        # Apply rotation and then swap the results back. This way
        # if volume contains representations z1, z2 of the same scene, 
        # we get z2', z1' which should match the original z1, z2
        return  self._pairwise_swap(
            menrot_F.rotate3d(volume, rotation_matrix, mode=self.mode)
        )
    
    def _eval_forward(self, volume: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs == {}:
            return volume
        
        if all(parm in kwargs for parm in ['azimuth','elevation']):
            if self._pairwise_swap_indices is None or len(self._pairwise_swap_indices) != volume.shape[0]:
                self._pairwise_swap_indices = [idx^1 for idx in range(volume.shape[0])]
            rotation_matrix = menrot_F.rotation_matrix_source_to_target(
                kwargs['azimuth'], kwargs['elevation'], self._pairwise_swap(kwargs['azimuth']), self._pairwise_swap(kwargs['elevation'])
            )
            return  self._pairwise_swap(
                menrot_F.rotate3d(volume, rotation_matrix, mode=self.mode)
            )
            
        if 'rotation_matrix' in kwargs and kwargs['rotation_matrix'] is not None:
            return menrot_F.rotate3d(volume, kwargs['rotation_matrix'], mode=self.mode)
        
        # Check if all angle parameters are provided
        angle_params = ['azimuth_source', 'elevation_source', 'azimuth_target', 'elevation_target']
        if all(param in kwargs for param in angle_params):
            rotation_matrix = menrot_F.rotation_matrix_source_to_target(
                kwargs['azimuth_source'], kwargs['elevation_source'],
                kwargs['azimuth_target'], kwargs['elevation_target']
            )
            # Check if we are generating new views
            if kwargs['azimuth_source'].shape[0] != kwargs['azimuth_target'].shape[0]:
                num_views = kwargs['azimuth_target'].shape[0]
                volume = volume.expand(num_views, -1, -1, -1, -1)
            return menrot_F.rotate3d(volume, rotation_matrix, mode=self.mode)
        
        raise ValueError("Either rotation_matrix or all four angle parameters must be provided in eval mode")
    
    def _pairwise_swap(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self._pairwise_swap_indices]
    
    
