import os
import gc 

import torch
import torch.distributed as dist

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
def ddp_cleanup():
    # Ensure all ranks reach this point
    # https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
    if dist.is_initialized():
        dist.barrier()
        # synchronize all CUDA operations
        torch.cuda.synchronize()
        # clear references to distributed tensors/ops
        gc.collect()
        torch.cuda.empty_cache()
        # one last sync just in case
        dist.barrier()
        # destroy process group
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[Rank {dist.get_rank()}] Failed to destroy process group cleanly: {e}")
            
