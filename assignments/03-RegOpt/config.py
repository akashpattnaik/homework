from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomRotation,
)


class CONFIG:
    """
    This class contains all the configuration parameters for the assignment.
    """

    batch_size = 32
    num_epochs = 5
    initial_learning_rate = 1e-3
    initial_weight_decay = 1e-4

    lrs_kwargs = {
        # "max_lr": 1e-3,
        # "min_lr": 1e-5,
        # "T_0": 50,
        "T_0": int(1563 * num_epochs * 1.1),
        # "T_mult": 2,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    # optimizer_factory: Callable[
    #     [nn.Module], torch.optim.Optimizer
    # ] = lambda model: torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.1,
    #     weight_decay=CONFIG.initial_weight_decay,
    # )

    # optimizer_factory: Callable[
    #     [nn.Module], torch.optim.Optimizer
    # ] = lambda model: torch.optim.Adadelta(
    #     model.parameters(),
    #     lr=1,
    #     weight_decay=CONFIG.initial_weight_decay,
    # )

    transforms = Compose(
        [
            RandomRotation(5),
            ToTensor(),
            # Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2]),
        ]
    )
