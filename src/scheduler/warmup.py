from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, min_lr, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [self._get_lr(base_lr, step) for base_lr in self.base_lrs]

    def _get_lr(self, base_lr, step):
        factor = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        lr = base_lr * factor
        if step > self.warmup_steps:
            lr = max(lr, self.min_lr)
        return lr