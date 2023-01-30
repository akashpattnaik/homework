import numpy as np
import torch


class LinearRegression:
    """A linear regression model that uses the closed form solution to fit the model."""

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit function

        Args:
            X (np.ndarray): features
            y (np.ndarray): targets
        """
        X_bar = np.hstack((X, np.ones((X.shape[0], 1))))
        wb = np.linalg.pinv(X_bar.T @ X_bar) @ X_bar.T @ y

        self.w = wb[:-1]
        self.b = wb[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict function

        Args:
            X (np.ndarray): features

        Returns:
            np.ndarray: predicted targets
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def _gradient_descent(self, w, b, lr):
        """
        gradient_descent algorithm.

        Parameters
        ----------
        w : torch.tensor
            weights.
        b : torch.tensor
            bias.
        lr : FLOAT
            learning rate.

        Returns
        -------
        w, b.

        """
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            # Set gradient to zero to flush the cache
            w.grad.zero_()
            b.grad.zero_()

        return (w, b)

    def _squared_error(self, y_hat, y):
        """
        Squared error loss function.

        Parameters
        ----------
        y_hat : torch.tensor
            predicted values.
        y : torch.tensor
            true values.

        Returns
        -------
        err: FLOAT
            the squared error (loss)

        """
        err = (y_hat - y) ** 2
        return err

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """Fit function

        Args:
            X (np.ndarray): targets
            y (np.ndarray): _description_
            lr (float, optional): learning rate. Defaults to 0.01.
            epochs (int, optional): number of epochs for training. Defaults to 1000.
        """
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y.squeeze())  # to avoid wrong broadcasting

        w = torch.zeros(X.shape[-1], 1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        for _ in range(epochs):
            y_hat = X_train @ w + b
            l = self._squared_error(y_hat, y_train).mean()  # Loss in `X` and `y`

            # Compute gradient on `l` with respect to `w`, `b`
            l.backward()
            w, b = self._gradient_descent(
                w, b, lr
            )  # Update parameters using their gradient

        self.w = w
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        y_train = torch.tensor(X, dtype=torch.float32) @ self.w + self.b
        return y_train.detach().numpy()
