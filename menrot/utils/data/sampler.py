import torch
from torch.utils.data import Sampler
import random
from itertools import chain

__all__ = [
    "RandomSymbolicPairSampler",
]

class RandomSymbolicPairSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        unique_shapes = list(self.dataset.metadata['shape_string'].unique())
        num_objects = len(unique_shapes)
        num_shapeviews = set(self.dataset.metadata['shape_string'].value_counts().to_list()).pop() # just to get first element
        assert num_shapeviews % 2 == 0, 'There is no even number of view pose per object'
        
        samples = []
        for object_id in range(num_objects):
            # we need to show every shapeviews onece in an epoch 
            img_pair_id = torch.randperm(num_shapeviews).view(num_shapeviews//2, 2)
            for img_id in img_pair_id:
                samples.append({'object_id': object_id, 'img_id_1': img_id[0].item(), 'img_id_2': img_id[1].item()})

        # we shuffle here because we don't want for the first x objects to be the same
        random.shuffle(samples)
        
        # her we get the corresponding indices, so if we have object_id = 10
        # and we have N possible shapeviews, and we paired shape 'a' and 'd', 
        # then  the correspondings data_indicies are (10*N)+a and (10*N)+d
        indices_pair = [[   
                elem['object_id']*num_shapeviews + elem['img_id_1'], 
                elem['object_id']*num_shapeviews + elem['img_id_2']
            ]  for elem in samples]
        indices = list(chain.from_iterable(indices_pair))
        return iter(indices)

    def __len__(self):
        return len(self.dataset)
    
    