from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    # RandomHorizontalFlip,
    # RandomRotation,
)


class CONFIG:
    """
    This class contains all the configuration parameters for the assignment.
    """

    batch_size = 64
    num_epochs = 2
    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # "lr_lambda": lambda epoch: 0.95**epoch,
        "max_update": 20,
        "base_lr": 0.3,
        "final_lr": 0.01,
        "warmup_steps": 5,
        "warmup_begin_lr": 0,
        # You can pass arguments to the learning rate scheduler
        # constructor here.
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            # RandomRotation(10),
            # RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
    )
