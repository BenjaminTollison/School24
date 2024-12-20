"""
Minimizing the loss function to measure how good the predictions are
"""
from practice.tensor import Tensor
import numpy as np


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error but in this case we ar going to really be using a total square error
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
