import numpy as np


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
        X_bar = np.hstack((X, np.ones((X.shape[0], 1))))
        wb = np.zeros((X.shape[-1] + 1, 1))

        for _ in range(epochs):
            # update gradient
            wb -= (-2 * X_bar.T @ (y[:, None] - X_bar @ wb)) * lr

        self.w = wb[:-1]
        self.b = wb[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b