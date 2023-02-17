from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import math


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler. This is a wrapper around the PyTorch _LRScheduler class.
    We reimplement a CosineScheduler from scratch here.
    """

    def __init__(
        self,
        optimizer,
        max_update,
        base_lr=0.1,
        final_lr=0,
        warmup_steps=5,
        warmup_begin_lr=0,
        last_epoch=-1,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        Adapted from: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html

        """
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.base_lr = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return the current learning rate. This is called internally by Torch.
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        if self.last_epoch < self.warmup_steps:
            warmup_lr = (
                self.warmup_begin_lr
                + (self.base_lr - self.warmup_begin_lr)
                * self.last_epoch
                / self.warmup_steps
            )
            return [warmup_lr]
        else:
            cosine_lr = (
                self.final_lr
                + (self.base_lr - self.final_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.last_epoch - self.warmup_steps) / self.max_steps
                    )
                )
                / 2
            )
            return [cosine_lr]

        # Here's our dumb baseline implementation:
        return [i for i in self.base_lrs]
