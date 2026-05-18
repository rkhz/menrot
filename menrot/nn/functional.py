import torch
import torch.nn.functional as torch_F
from typing import Union

__all__ = ['rotate3d']

def rotate3d(
    volume: torch.Tensor,
    rotation_matrix: torch.Tensor, 
    mode: str = 'bilinear'
)-> torch.Tensor:
    """
    Notes:        
        The grid_sample function performs the inverse transformation of the input
        coordinates, so invert (ie.e transpose) matrix to get forward transformation:
            - (line 25): rotation_matrix.mT 
        
        The grid_sample function swaps x and z (i.e. it assumes the tensor
        dimensions are ordered as z, y, x), therefore we need to flip the rows and
        columns of the matrix (which we can verify is equivalent to multiplying by
        the appropriate permutation matrices):
            - (line 25): rotation_matrix.mT.flip(dims=(1,2))
    """
    affine_matrix = torch_F.pad(
        input=rotation_matrix.mT.flip(dims=(1,2)), 
        pad=(0, 1, 0, 0), value=0, mode='constant'
    )
    
    affine_grid = torch_F.affine_grid(
        affine_matrix, 
        size=volume.shape,
        align_corners=False
    )
    
    return torch_F.grid_sample(volume, affine_grid, mode=mode, align_corners=False)

def rotation_matrix_source_to_target(
    azimuth_source: torch.Tensor, elevation_source: torch.Tensor,
    azimuth_target: torch.Tensor, elevation_target: torch.Tensor
) -> torch.Tensor:
    rotation_source = _rotation_matrix_camera_to_world(azimuth_source, elevation_source)
    rotation_target = _rotation_matrix_camera_to_world(azimuth_target, elevation_target)
    return rotation_target @ (rotation_source.mT)


def _rotation_matrix_camera_to_world(
    azimuth: torch.Tensor, 
    elevation: torch.Tensor
) -> torch.Tensor:
    return (_rotation_matrix_y(-azimuth) @ _rotation_matrix_z(-elevation)).mT


def _rotation_matrix_y(
    angle_degrees: Union[torch.Tensor, float]
) -> torch.Tensor:
    angle = torch.deg2rad(angle_degrees)
    
    cos_ = torch.cos(angle)
    sin_ = torch.sin(angle)
    
    rotation_matrix = torch.zeros((*angle.shape, 3, 3), dtype=torch.float32, device=angle.device)

    rotation_matrix[:, 0, 0] =  cos_
    rotation_matrix[:, 0, 2] =  sin_
    rotation_matrix[:, 1, 1] =  1.0
    rotation_matrix[:, 2, 0] = -sin_
    rotation_matrix[:, 2, 2] =  cos_
    return rotation_matrix

def _rotation_matrix_z(
    angle_degrees: Union[torch.Tensor, float]
) -> torch.Tensor:
    angle = torch.deg2rad(angle_degrees)
    
    cos_ = torch.cos(angle)
    sin_ = torch.sin(angle)
    
    rotation_matrix = torch.zeros((*angle.shape, 3, 3), dtype=torch.float32, device=angle.device)
    rotation_matrix[:, 0, 0] =  cos_
    rotation_matrix[:, 0, 1] = -sin_
    rotation_matrix[:, 1, 0] =  sin_
    rotation_matrix[:, 1, 1] =  cos_
    rotation_matrix[:, 2, 2] =  1.0
    return rotation_matrix


