"""
Our netural net is going to be made up of layers

inputs -> Linear -> Tanh -> Linear -> output
"""
import numpy as np
from typing import Dict, Callable

from practice.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        back propogates error through the previous layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes Ax=b
    with outputs = inputs @ weights + batch
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # inputs will be (batch_size, output_size)
        super().__init__()
        self.params["weights"] = np.random.randn(input_size, output_size)
        self.params["batch"] = np.random.randn(output_size)

        def forward(self, inputs: Tensor) -> Tensor:
            self.inputs = inputs
            return self.inputs @ self.params["weights"] + self.params["batch"]

        def backward(self, grad: Tensor) -> Tensor:
            """
            if y = f(x) and x = A @ B + c
            then dy/dA = f'(x) @ B.T
            dy/dB = A.T @ f'(x)
            dy/dc = f'(x)
            """
            self.grads["batch"] = np.sum(grad, axis=0)
            self.grads["weights"] = self.inputs.T @ grad
            return grad @ self.params["weights"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    applying a function elementwise to its inputs
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    return 1 - tanh(x) ** 2


class Tanh(Activation):
    def __init__(self, f: F, f_prime: F) -> None:
        raise NotImplementedError
