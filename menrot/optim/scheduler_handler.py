from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

__all__ = [
    "SchedulerHandler",
]

class SchedulerHandler:
    def __init__(self, optimizer, strategy, total_epochs, base_lr=1e-3, steps_per_epoch=None, gamma=0.97):
        self.optimizer = optimizer
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.base_lr = base_lr

        self.steps_per_epoch = steps_per_epoch  # for CosineAnnealingLR
        self.gamma = gamma                      # for ExponentialLR
        
        self.scheduler = None
        if strategy == "cosine":
            self.scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs * steps_per_epoch)
        elif strategy == "exponential":
            self.scheduler = ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def get_lr(self):
       return self.optimizer.param_groups[0]["lr"]

    def step(self):
        self.scheduler.step()