import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

__all__ = ["DistributedWrapperSampler"]          

# source: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py

class DistributedWrapperSampler(Sampler):
    def __init__(
        self, 
        base_sampler: Sampler, 
        num_replicas=None,
        rank=None,
        seed: int = 0
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.base_sampler = base_sampler
        self.dataset = getattr(base_sampler, 'dataset', None)
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # Length of original sampler
        self.total_size = len(base_sampler)
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        self.seed = seed

    def __iter__(self):
        torch.manual_seed( self.seed + self.epoch) 
        # Get full list of indices from the base sampler
        indices = list(self.base_sampler)
        
        # Pad if necessary
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample for this rank
        start = self.rank * self.num_samples
        end = start + self.num_samples
        
        #print(f"{self.rank}/{self.num_replicas} : I am doing from {start} until {end} (out of {self.total_size})")
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
        