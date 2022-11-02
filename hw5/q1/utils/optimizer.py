from __future__ import annotations
import cupy as cp
from typing import Optional, Sequence, Union, overload
from easydict import EasyDict as edict

import utils.activation as ac
from utils.common import *
import utils.module as m
from utils.model import NeuralNet
import utils.loss as ls

class Optimizer(object):
    def __init__(self, model: NeuralNet, loss: ls.Loss, lr: float=0.1) -> None:
        self.model = model
        self.loss = loss
        self.lr = lr

    def zero_grad(self):
        for layer in self.model.modules:
            layer.zero_grad()
    
    def step(self):
        for layer in self.model.modules:
            update = {k: self._calculate_update(layer, k, grad) \
                for k, grad in layer.grad.items() if k in layer.weight.keys()}
            layer.update(update, self.lr)
    
    def _calculate_update(self, layer: m.Layer, k: str, v: cp.ndarray):
        raise NotImplementedError()

class SGD(Optimizer):
    def _calculate_update(self, layer: m.Layer, k: str, grad: cp.ndarray):
        return -grad

class Adam(Optimizer):
    def __init__(self, model: NeuralNet, loss: ls.Loss, lr: float = 0.1, beta1: float=0.0, beta2: float = 0.999) -> None:
        super().__init__(model, loss, lr)
        self.m = edict()
        self.v = edict()

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        
        self.t = 0

    def _calculate_update(self, layer: m.Layer, k: str, grad: cp.ndarray):

        m = self.m[str(layer)][k]
        v = self.v[str(layer)][k]
        
        m = self.beta1*m + (1-self.beta1) * grad
        v = self.beta2*v + (1-self.beta2) * cp.power(grad, 2)

        self.m[str(layer)][k] = m
        self.v[str(layer)][k] = v

        m /= (1-self.beta1**self.t)
        v /= (1-self.beta2**self.t)

        return - m / (self.eps + cp.power(v, 0.5))
    
    def initialize_adam(self):
        for layer in self.model.modules:
            self.m[str(layer)] = edict()
            self.v[str(layer)] = edict()
            for k, grad in layer.grad.items():
                if k not in layer.weight.keys():
                    continue
                self.m[str(layer)][k] = cp.zeros_like(grad)
                self.v[str(layer)][k] = cp.zeros_like(grad)
   
    def step(self):
        if self.t==0:
            self.initialize_adam()
        self.t += 1
        super().step()