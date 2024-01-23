"""
A neural net is just a collection of layers, but this will be a simple example
"""
from practice.tensor import Tensor
from practice.layers import Layer
from typing import Sequence


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
