import os
import numpy as np 
import h5py
import torch    

from tqdm import tqdm
from PIL import Image
from shaperenderer.geometry import ShapeString

from menrot.utils.data import NeroskelDataset

__all__ = [
    "CogShapeSkel",
    "EncodedCogShapeSkel"
]

class CogShapeSkel(NeroskelDataset):
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)

    def __getitem__(self, idx):
        data_id = self.metadata['id'][idx]
        imag_name_1 = os.path.join(self.data_dir, 'imag', f'{data_id}_1.png')
        imag_name_2 = os.path.join(self.data_dir, 'imag', f'{data_id}_2.png')
        
        image_1 = Image.open(imag_name_1).convert('RGB')
        image_2 = Image.open(imag_name_2).convert('RGB')
      
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        delta_phi = int(self.metadata['delta_phi'][idx])
        is_mirror = int(self.metadata['is_mirror'][idx])
        rot_qdrt = int(self.metadata['rot_qdrt'][idx])
        azimuth_target = float(self.metadata['phi_1'][idx])
        azimuth_source = float(self.metadata['phi_2'][idx])

        skeleton_1 = torch.tensor(ShapeString(str(self.metadata['skeleton_1'][idx])).encode())
        skeleton_2 = torch.tensor(ShapeString(str(self.metadata['skeleton_2'][idx])).encode())

        return image_1, image_2, {'skeletons': [skeleton_1, skeleton_2], 'delta_phi': delta_phi, 'is_mirror': is_mirror, 'rot_qdrt': rot_qdrt, 'azimuth_target':azimuth_target, 'azimuth_source': azimuth_source}


class EncodedCogShapeSkel(torch.utils.data.Dataset):
    def __init__(self, dataset : 'CogShapeSkel', encoder_0, encoder_1):
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        
        self.metadata = dataset.metadata
        self.metadata['class'] = self.metadata['rot_qdrt'].apply(lambda x: x % 4)

        freq_class = torch.tensor(
            self.metadata['class'].value_counts(normalize=True).sort_index(ascending=True).to_list(), 
            dtype=torch.float32
        )
        self.weights_class = 1.0 / freq_class
        
        self.split = dataset.split
        self.root_dir = dataset.root_dir
        self.file_name = f'encoded_{self.split}_cog_data.pt'
        self.file_path = os.path.abspath(os.path.join(self.root_dir, self.file_name))

        try:
            if os.path.isfile(self.file_path):
                os.remove(self.file_path)
                self._precompute(dataset)
            else:
                self._precompute(dataset)

        except OSError as e:
            print(f"Error removing file: {e}")
            

        with h5py.File(self.file_path, 'r') as f:
            self.feat_1 = torch.from_numpy(f['features_1'][:]).float()
            self.feat_2 = torch.from_numpy(f['features_2'][:]).float()
            self.label = torch.from_numpy(f['labels'][:]).long()
            
    def _precompute(self,dataset):
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=240, shuffle=False, drop_last=False)
        
        all_features_1 = []
        all_features_2 = []
        all_labels = []
        with torch.no_grad():
            for im1, im2, meta in tqdm(dataloader, desc="Extracting features"):
                all_labels.append(meta['rot_qdrt'].long().cpu())
                all_features_1.append(self.encoder_1(self.encoder_0(im1)).cpu())
                all_features_2.append(self.encoder_1(self.encoder_0(im2)).cpu())
        
        all_labels = torch.cat(all_labels).numpy()
        all_features_1 = torch.cat(all_features_1).numpy()
        all_features_2 = torch.cat(all_features_2).numpy()
        
        print("Saving features...")
        with h5py.File(self.file_path, 'w') as f:
            f.create_dataset('features_1', data=np.array(all_features_1), compression='gzip')
            f.create_dataset('features_2', data=np.array(all_features_2), compression='gzip')
            f.create_dataset('labels', data=np.array(all_labels), compression='gzip')
        print(f"Features saved to {self.file_path}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return self.feat_1[idx], self.feat_2[idx], self.label[idx]
    
