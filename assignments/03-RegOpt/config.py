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
    initial_learning_rate = 1e-4
    initial_weight_decay = 0

    lrs_kwargs = {
        "max_update": 1600,
        "final_lr": initial_learning_rate,
        "warmup_steps": 200,
        "warmup_begin_lr": initial_learning_rate,
        "base_lr": 1e-3,
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
