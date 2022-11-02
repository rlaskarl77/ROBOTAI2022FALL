from __future__ import annotations
import cupy as cp
from typing import Optional, Sequence, Union
from easydict import EasyDict as edict

import utils.activation as ac
from utils.common import *
import utils.module as m

class NeuralNet(object):
    def __init__(self, modules: Sequence[m.Layer]) -> None:
        self.modules = modules
    
    def forward(self, x: cp.ndarray, learning=False):
        for layer in self.modules:
            x = layer.forward(x, learning=learning)
        return x
    
    def backward(self, grad: cp.ndarray):
        for layer in self.modules[::-1]:
            # print(layer)
            grad = layer.backward(grad)
    
    def explain(self):
        print(f'model structure')
        print(f'*'*20)
        for layer in self.modules:
            print(layer)
        print(f'*'*20)

class CNN(NeuralNet):
    def __init__(self) -> None:
        module_list = []
        # (B, 1, 28, 28)
        module_list.append(m.Conv2d(1, 16, 1, 1, 3))
        module_list.append(m.Conv2d(16, 16, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 16, 14, 14)
        module_list.append(m.Conv2d(16, 32, 1, 1, 3))
        module_list.append(m.Conv2d(32, 32, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 32, 7, 7)
        module_list.append(m.Conv2d(32, 64, 1, 1, 3))
        module_list.append(m.Conv2d(64, 64, 1, 1, 3))
        module_list.append(m.MaxPool2d(3, 2, 1))
        # (B, 64, 4, 4)
        module_list.append(m.Conv2d(64, 128, 1, 1, 3))
        module_list.append(m.Conv2d(128, 128, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 128, 2, 2)
        module_list.append(m.Conv2d(128, 256, 1, 1, 3))
        module_list.append(m.Conv2d(256, 256, 1, 1, 3))
        # (B, 256, 2, 2)
        module_list.append(m.GlobalAveragePool())
        # (B, 256)
        module_list.append(m.Perceptron(256, 1000))
        module_list.append(m.Dropout(0.3))
        module_list.append(m.Perceptron(1000, 10, activation=ac.Identical))
        super().__init__(module_list)


class CNN2(NeuralNet):
    def __init__(self) -> None:
        module_list = []
        # (B, 1, 28, 28)
        module_list.append(m.Conv2d(1, 16, 1, 1, 3))
        module_list.append(m.Conv2d(16, 16, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 16, 14, 14)
        module_list.append(m.Conv2d(16, 32, 1, 1, 3))
        module_list.append(m.Conv2d(32, 32, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 32, 7, 7)
        module_list.append(m.Conv2d(32, 64, 1, 1, 3))
        module_list.append(m.Conv2d(64, 64, 1, 1, 3))
        # (B, 64, 7, 7)
        module_list.append(m.Flatten())
        # (B, 64*7*7)
        module_list.append(m.Dropout(0.5))
        module_list.append(m.Perceptron(64*7*7, 1000))
        # module_list.append(m.Dropout(0.2))
        module_list.append(m.Perceptron(1000, 10, activation=ac.Identical))
        super().__init__(module_list)


class CNN3(NeuralNet):
    def __init__(self) -> None:
        module_list = []
        # (B, 1, 28, 28)
        module_list.append(m.Conv2d(1, 16, 2, 2, 5))
        module_list.append(m.Conv2d(16, 32, 2, 2, 5))
        module_list.append(m.Flatten())
        module_list.append(m.Dropout(0.25))
        module_list.append(m.Perceptron(32*7*7, 1000))
        module_list.append(m.Dropout(0.1))
        module_list.append(m.Perceptron(1000, 10, activation=ac.Identical))
        super().__init__(module_list)


class CNN4(NeuralNet):
    def __init__(self) -> None:
        module_list = []
        # (B, 1, 28, 28)
        module_list.append(m.Conv2d(1, 16, 1, 1, 3))
        module_list.append(m.Conv2d(16, 16, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 16, 14, 14)
        module_list.append(m.Conv2d(16, 32, 1, 1, 3))
        module_list.append(m.Conv2d(32, 32, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 32, 7, 7)
        module_list.append(m.Conv2d(32, 64, 1, 1, 3))
        module_list.append(m.Conv2d(64, 64, 1, 1, 3))
        module_list.append(m.MaxPool2d(3, 2, 1))
        # (B, 64, 4, 4)
        module_list.append(m.Conv2d(64, 128, 1, 1, 3))
        module_list.append(m.Conv2d(128, 128, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 128, 2, 2)
        module_list.append(m.Conv2d(128, 256, 1, 1, 3))
        module_list.append(m.Conv2d(256, 256, 1, 1, 3))
        module_list.append(m.MaxPool2d(2, 2))
        # (B, 256, 2, 2)
        module_list.append(m.GlobalAveragePool())
        # (B, 64*7*7)
        module_list.append(m.Dropout(0.5))
        module_list.append(m.Perceptron(64*7*7, 1000))
        # module_list.append(m.Dropout(0.2))
        module_list.append(m.Perceptron(1000, 10, activation=ac.Identical))
        super().__init__(module_list)