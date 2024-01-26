"""
The first example to work through is the XOR, because it isnt a simple linear model
"""
import numpy as np

from practice.train import train
from practice.neural_net import NeuralNet
from practice.layers import Linear, Tanh

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

net = NeuralNet([Linear(input_size=2, output_size=2)])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
