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

    batch_size = 16
    num_epochs = 5
    initial_learning_rate = 1e-3
    initial_weight_decay = 1e-4

    lrs_kwargs = {
        "T_0": int(3125 * num_epochs * 1.1),
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
            RandomRotation(5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2]),
        ]
    )
