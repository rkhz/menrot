import os
import time 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist
import torch.nn.functional as F


from menrot.models import ViT3d, VisionSymbolicModel
from menrot.representations import SpatialRep
from menrot.optim import SchedulerHandler
from menrot.utils.checkpoint import Snapshot
from menrot.utils.data import (
    DistributedWrapperSampler, 
    RandomSkelPairSampler
)
    

__all__ = [
    "Trainer",
    "TrainerMLP",
    "TrainerViT",
    "TrainerVSM",
    "TrainerConfig"
]

# source: https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/trainer.py

@dataclass
class TrainerConfig:
    model_config: Dict[str, Any] = None
    max_epochs: int = None
    batch_size: int = None
    lr: float = None
    save_every: int = None
    data_loader_workers: int = None
    data_dir: str = None
    log_path: str = None
    snapshot_path: Optional[str] = None


class Trainer:
    def __init__(
        self, 
        trainer_config: TrainerConfig, 
        model, 
        optimizer, 
        criterion,
        train_dataset, 
        valid_dataset=None
    ):
        self.config = trainer_config

        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"]) 
        self.global_rank = int(os.environ["RANK"]) 
        
        # data stuff
        self.train_loader = self._prepare_dataloader(train_dataset, train=True)
        self.valid_loader = self._prepare_dataloader(valid_dataset, train=False) if valid_dataset else None
        
        if self.global_rank == 0:
            print(f'len train dl: {len(self.train_loader)} - len valid dl: {len(self.valid_loader)}')
        
        # initialize train states
        self.epochs_run = 0
        self.optimizer = optimizer        
        self.criterion = criterion
        self.save_every = self.config.save_every
        self.losses = {'train':[], 'val':[]}
        self.best_loss_so_far = torch.inf
        
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()
        
        # wrap with DDP. this step will synch model across all the processes.
        self.model = model.to(self.local_rank)
        self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

    def _prepare_dataloader(self, dataset: Dataset, train=True):
        if train:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
                sampler=DistributedWrapperSampler(
                    RandomSkelPairSampler(dataset)
                ),
                drop_last=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                shuffle=False,
                drop_last=True
            )

    def _load_snapshot(self):
        try:
            snapshot = Snapshot.load_from(self.config.snapshot_path)
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return 
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch, path):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            model_config=self.config.model_config,
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        
        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, os.path.join(self.config.log_path, path))
            
        print(f"-> Snapshot saved at epoch {epoch}: {path}")
    
    def _save_training_history(self):
        df_loss = pd.DataFrame(self.losses)
        df_loss.to_csv(os.path.join(self.config.log_path, f'{self.criterion._get_name()}_history.csv'), index=False) 

        plt.figure()
        plt.plot(self.losses['train'], label='train')
        plt.plot(self.losses['val'], label='val')
        plt.xlabel('epoch')
        plt.ylabel(f'{self.criterion._get_name()}')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.config.log_path, f'{self.criterion._get_name()}.png'))
        
        plt.figure()
        plt.plot(np.log(np.array(self.losses['train'])), label='train')
        plt.plot(np.log(np.array(self.losses['val'])), label='val')
        plt.xlabel('epoch')
        plt.ylabel(f'Log_({self.criterion._get_name()})')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.config.log_path, f'log_{self.criterion._get_name()}.png'))

    def _format_epoch_log(self, epoch, time_start, time_end, train=True):
        if train:
            return f"EPOCH {epoch:03d}/{self.config.max_epochs:03d} - TRAIN -- loss: {self.losses['train'][epoch-1]:.4f} -- {time_end - time_start:.4f}s/epoch"
        else:
            return f"\n|____________ - VALID -- loss: {self.losses['val'][epoch-1]:.4f} -- {time_end- time_start:.4f}s/epoch\n"

    def _sync_metric_across_processes(self, metric):
        # Sync across processes
        metric_tensor = torch.tensor(metric, device=self.local_rank)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        metric_tensor /= dist.get_world_size()
        return metric_tensor.item()
    
    def _run_batch(self, batch_data, train: bool = True) -> float:
        inputs, targets = batch_data
        inputs = inputs.to(self.local_rank)
        targets = targets.to(self.local_rank)

        with torch.set_grad_enabled(train):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
                
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if train:
            dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        for _, batch in enumerate(dataloader):
            batch_loss = self._run_batch(batch, train)
            epoch_loss += batch_loss

        avg_epoch_loss = epoch_loss / len(dataloader)
        self.losses['train' if train else 'val'].append(
            self._sync_metric_across_processes(avg_epoch_loss)
        )
   
    def run(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            time_start_epoch = time.time()
            
            epoch += 1
            self.model.train()
            self._run_epoch(epoch, self.train_loader, train=True)
            time_end_epoch = time.time()
            epoch_log = self._format_epoch_log(epoch, time_start_epoch, time_end_epoch, train=True)

            # eval run
            if self.valid_loader:
                time_start_val = time.time()
                self.model.eval()
                self._run_epoch(epoch, self.valid_loader, train=False)
                time_end_val = time.time()
                epoch_log +=  self._format_epoch_log(epoch, time_start_val, time_end_val, train=False)
              
            # log and snapshot saving
            if self.global_rank == 0:
                if self.losses['val'][epoch-1] < self.best_loss_so_far:
                    self.best_loss_so_far = self.losses['val'][epoch-1]
                    self._save_snapshot(epoch, path='model_best_so_far.pt')
                        
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch, path= self.config.snapshot_path)
                print(epoch_log)
                
        if self.global_rank == 0:
            self._save_snapshot(epoch, path='model_trained.pt') 
            self._save_training_history()


class TrainerMLP(Trainer):
    def __init__(
        self, 
        trainer_config: TrainerConfig, 
        model, 
        optimizer, 
        criterion, 
        train_dataset, 
        one_hot_embeds=False,
        valid_dataset=None
    ):
        super().__init__(
            trainer_config=trainer_config, 
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            train_dataset=train_dataset, 
            valid_dataset=valid_dataset
        )
        self.one_hot_embeds = one_hot_embeds
        self.metrics = {
            'acc': {'train':[], 'val':[]}, 
        }
    
    def _prepare_dataloader(self, dataset: Dataset, train=True):
        if train:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
                sampler=DistributedSampler(dataset),
                drop_last=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
                drop_last=True
            )
    
    def _save_training_history(self):
        super()._save_training_history()

        df_skel_acc = pd.DataFrame(self.metrics['acc'])
        df_skel_acc.to_csv(os.path.join(self.config.log_path, f'Accuracy_history.csv'), index=False) 
        plt.figure()
        plt.plot(self.metrics['acc']['train'], label='train')
        plt.plot(self.metrics['acc']['val'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy [%]')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.config.log_path, f'Accuracy.png'))

    def _format_epoch_log(self, epoch, time_start, time_end, train=True):
        if train:
            return f"EPOCH {epoch:03d}/{self.config.max_epochs:03d} - TRAIN -- loss: {self.losses['train'][epoch-1]:.4f} -- acc: {self.metrics['acc']['train'][epoch-1]:.4f} -- {time_end - time_start:.4f}s/epoch"
        else:
            return f"\n|____________ - VALID -- loss: {self.losses['val'][epoch-1]:.4f} -- acc: {self.metrics['acc']['val'][epoch-1]:.4f} -- {time_end- time_start:.4f}s/epoch\n"
    
    def _one_hot_embeds(self, preds):
        if self.one_hot_embeds:
            return  F.one_hot(preds.long(), num_classes=6).float()
        else:
            return preds
        
    def _run_batch(self, batch_data, train: bool = True) -> float:    
        inputs_1, inputs_2, targets = batch_data
        
        inputs_1 = inputs_1.to(self.local_rank)
        inputs_2 = inputs_2.to(self.local_rank)
        if self.config.model_config['output_dim'] == 1:
            targets = torch.where(targets < 4, 1, 0).float().to(self.local_rank)
        elif self.config.model_config['output_dim'] == 4:
            targets = (targets % 4).long().to(self.local_rank)
        else:
            targets = torch.where(targets == 4, targets, targets % 4).long().to(self.local_rank)

        with torch.set_grad_enabled(train):
            inputs = torch.cat((
                self._one_hot_embeds(inputs_1).view(self.config.batch_size, -1), 
                self._one_hot_embeds(inputs_2).view(self.config.batch_size, -1)
                ), 
                dim=-1)

            outputs = self.model(inputs)       
            
            # accuracy metric:
            if self.config.model_config['output_dim'] == 1:
                outputs = outputs.squeeze()
                preds = (torch.sigmoid(outputs).squeeze() >= 0.5).float()
            else:
                preds = outputs.argmax(dim=-1)
                
            acc =  (preds == targets).float().mean() # can use .mean because drop_last = True

            loss = self.criterion(outputs, targets)    

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
           
        return loss.item(), acc.item()
    
    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if train:
            dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_acc = 0.0
        for _, batch in enumerate(dataloader):
            batch_loss, batch_acc = self._run_batch(batch, train)            
            epoch_loss += batch_loss
            epoch_acc += batch_acc

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        
        self.losses['train' if train else 'val'].append(
            self._sync_metric_across_processes(epoch_loss)
        )

        self.metrics['acc']['train' if train else 'val'].append(
            self._sync_metric_across_processes(epoch_acc)
        )
     
             
class TrainerViT(Trainer):
    def __init__(
        self, 
        trainer_config: TrainerConfig, 
        model: ViT3d, 
        encoder_0: SpatialRep,
        scheduler_handler: SchedulerHandler, 
        optimizer, 
        criterion,
        train_dataset, 
        valid_dataset=None
    ):
        super().__init__(
            trainer_config=trainer_config, 
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            train_dataset=train_dataset, 
            valid_dataset=valid_dataset
        )
        
        self.encoder_0 = encoder_0
        self.scheduler_handler = scheduler_handler

        self.metrics = {
            'skel_acc': {'train':[], 'val':[]}, 
            'char_acc': {'train':[], 'val':[]}
        }

    def _save_training_history(self):
        super()._save_training_history()

        df_skel_acc = pd.DataFrame(self.metrics['skel_acc'])
        df_skel_acc.to_csv(os.path.join(self.config.log_path, f'SkelAccuracy_history.csv'), index=False) 
        plt.figure()
        plt.plot(self.metrics['skel_acc']['train'], label='train')
        plt.plot(self.metrics['skel_acc']['val'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('Skeleton Accuracy [%]')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.config.log_path, f'SkelAccuracy.png'))
        

        df_char_acc = pd.DataFrame(self.metrics['char_acc'])
        df_char_acc.to_csv(os.path.join(self.config.log_path, f'CharAccuracy_history.csv'), index=False) 
        plt.figure()
        plt.plot(self.metrics['char_acc']['train'], label='train')
        plt.plot(self.metrics['char_acc']['val'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('Char Accuracy [%]')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.config.log_path, f'CharAccuracy.png'))

    def _format_epoch_log(self, epoch, time_start, time_end, train=True):
        if train:
            return f"EPOCH {epoch:03d}/{self.config.max_epochs:03d} - TRAIN -- loss: {self.losses['train'][epoch-1]:.4f} -- acc: {self.metrics['skel_acc']['train'][epoch-1]:.4f} -- {time_end - time_start:.4f}s/epoch"
        else:
            return f"\n|____________ - VALID -- loss: {self.losses['val'][epoch-1]:.4f} -- acc: {self.metrics['skel_acc']['val'][epoch-1]:.4f} -- {time_end- time_start:.4f}s/epoch\n"

    def _run_batch(self, batch_data, train: bool = True) -> float:    
        inputs, _, targets, azimuth_source, _ = batch_data
        inputs = inputs.to(self.local_rank)
        targets = targets.to(self.local_rank)
        azimuth_source = azimuth_source.to(self.local_rank)
        
        with torch.set_grad_enabled(train):
            outputs = self.model(
                self.encoder_0(
                    inputs, 
                    azimuth= azimuth_source if train else None
                )
            ) 
            loss = self.criterion(outputs.reshape(-1, 6), targets.reshape(-1))            
            
            # accuracy metric:
            preds = outputs.argmax(dim=-1)
            skel_acc = preds.eq(targets).all(dim=1).float().mean() # can use .mean because drop_last = True
            char_acc = preds.eq(targets).float().mean()

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
           
        return loss.item(), skel_acc.item(), char_acc.item()
    
    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if train:
            dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_skel_acc = 0.0
        epoch_char_acc = 0.0
        for _, batch in enumerate(dataloader):
            batch_loss, batch_skel_acc, batch_char_acc = self._run_batch(batch, train)
            if train:
                self.scheduler_handler.step()
            
            epoch_loss += batch_loss
            epoch_skel_acc += batch_skel_acc
            epoch_char_acc += batch_char_acc

        epoch_loss /= len(dataloader)
        epoch_skel_acc /= len(dataloader)
        epoch_char_acc /= len(dataloader)
        
        self.losses['train' if train else 'val'].append(
            self._sync_metric_across_processes(epoch_loss)
        )

        self.metrics['skel_acc']['train' if train else 'val'].append(
            self._sync_metric_across_processes(epoch_skel_acc)
        )

        self.metrics['char_acc']['train' if train else 'val'].append(
            self._sync_metric_across_processes(epoch_char_acc)
        )

class TrainerVSM(TrainerViT):
    def __init__(
        self, 
        trainer_config: TrainerConfig, 
        model: VisionSymbolicModel, 
        encoder_0: SpatialRep,
        scheduler_handler: SchedulerHandler, 
        optimizer, 
        criterion,
        train_dataset, 
        valid_dataset=None
    ):
        super().__init__(
            trainer_config=trainer_config, 
            model=model, 
            encoder_0= encoder_0,
            scheduler_handler= scheduler_handler, 
            optimizer=optimizer, 
            criterion=criterion, 
            train_dataset=train_dataset, 
            valid_dataset=valid_dataset
        )
        
    def _run_batch(self, batch_data, train: bool = True) -> float:    
        inputs, _, targets, azimuth_source, _ = batch_data
        inputs = inputs.to(self.local_rank)
        targets = targets.to(self.local_rank)
        azimuth_source = azimuth_source.to(self.local_rank)
        
        with torch.set_grad_enabled(train):
            outputs = self.model(
                self.encoder_0(
                    inputs, 
                    azimuth= azimuth_source if train else None
                ),
                tgt=targets 
            ) 
            loss = self.criterion(outputs.reshape(-1, self.model.module.decoder.vocab_size), targets.reshape(-1))            
            
            # accuracy metric:
            preds = outputs.argmax(dim=-1)
            skel_acc = preds.eq(targets).all(dim=1).float().mean() # can use .mean because drop_last = True
            char_acc = preds.eq(targets).float().mean()

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
           
        return loss.item(), skel_acc.item(), char_acc.item()
