from typing import Callable
import torch


class MLP(torch.nn.Module):
    """A simple MLP"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            hidden_count: The number of hidden layers.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__()

        self.activation = activation()

        # initialize layers of MLP
        self.layers = torch.nn.ModuleList()

        n_input = input_size
        # Loop over layers and create each one
        for _ in range(hidden_count):
            layer = torch.nn.Linear(n_input, hidden_size)
            initializer(layer.weight)
            self.layers += [layer]
            n_input = hidden_size

        self.out = torch.nn.Linear(n_input, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        x = x.view(x.shape[0], -1)

        # Get activations of each layer
        for layer in self.layers:
            x = self.activation(layer(x))

        # Get outputs
        x = self.out(x)

        return x
