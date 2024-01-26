"""
using an optimizer to adjust parameters during
backpropgation
"""
from practice.neural_net import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.lr = learning_rate

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
