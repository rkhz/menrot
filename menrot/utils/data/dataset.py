import os
import pandas as pd
import zipfile
import torch

__all__ = [
    "NeroskelDataset"
]

class NeroskelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # extract the ZIP file if the extraction path does not exist (TODO: download it from mirror if ZIP not here)
        # self.zip_path = os.path.join(root_dir, f'shepard_metzler_{kind}.zip')
        # self.extract_to_path = os.path.join(root_dir, kind)
        # self._extract_zip()
        self.split = split
        self.data_dir = os.path.join(self.root_dir, split)
        self.metadata = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'), 
            index_col=0
        )
    
    def _extract_zip(self):
        # Check if the extract path exists, if not, extract the ZIP file
        if not os.path.exists(self.extract_to_path):
            os.makedirs(self.extract_to_path)
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_to_path)
            print(f"Extracted all contents of '{self.zip_path}' to '{self.extract_to_path}'.")

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Must be implemented by subclass."""
        raise NotImplementedError






    