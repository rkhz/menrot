import torch

__all__ = [
    "MenrotRenderer",
]

class MenrotRenderer(torch.utils.data.Dataset):
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError