from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler. This is a wrapper around the PyTorch _LRScheduler class.
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        Adapted from: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html

        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return the current learning rate. This is called internally by Torch.
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        return [
            base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]
        # Here's our dumb baseline implementation:
        return [i for i in self.base_lrs]
