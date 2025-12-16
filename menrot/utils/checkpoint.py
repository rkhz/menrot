import torch 
import fsspec
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict

# source: https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    model_config: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    finished_epoch: int

    @classmethod
    def load_from(cls, path: str, device: str = "cpu"):
        snapshot_file = fsspec.open(path)
        with snapshot_file as f:
            snapshot_data = torch.load(f, map_location=device)
            snapshot_data.setdefault("model_config", {})
        return cls(**snapshot_data)
