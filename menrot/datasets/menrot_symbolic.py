import os
import torch

from PIL import Image
from shaperenderer.geometry import ShapeString

from menrot.utils.data import MenrotDataset

__all__ = [
    "MenrotSymbolic",
]
 
class MenrotSymbolic(MenrotDataset):
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)

    def __getitem__(self, idx):
        data_id = self.metadata['id'][idx]
        imag_name = os.path.join(self.data_dir, 'imag', f'{data_id}.png')
        image = Image.open(imag_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        shape_string = str(self.metadata['shape_string'][idx])
        skeleton = torch.tensor(ShapeString(str(self.metadata['skeleton'][idx])).encode())
        angle_phi = float(self.metadata['phi_angle'][idx])
        qdrt = int(self.metadata['qdrt'][idx])

        return image, shape_string, skeleton, angle_phi, qdrt