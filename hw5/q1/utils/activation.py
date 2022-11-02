from __future__ import annotations
import cupy as cp
from typing import Any, Optional, Sequence, Tuple, Union
from easydict import EasyDict as edict

class Activation(object):
    def __init__(self) -> None:
        self.weight = edict()
        self.grad = edict()
        self.act = edict()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()
    
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        raise NotImplementedError()

class Identical(Activation):
    def __call__(self, x: cp.ndarray, learning: bool=False) -> Tuple[cp.ndarray, cp.ndarray]:
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return grad

class ReLU(Activation):
    def __call__(self, x: cp.ndarray, learning: bool=False) -> Tuple[cp.ndarray, cp.ndarray]:
        mask = x>0
        x[~mask] = 0
        self.weight.mask = mask if learning else None
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        if self.weight.get('mask', None) is None:
            raise Exception('At least one forward pass needed before backprop.')
        grad[~self.weight.mask] = 0
        self.grad = grad
        return grad

class LeakyReLU(Activation):
    def __init__(self, alpha: float=0.1) -> None:
        super().__init__()
        self.weight.alpha= alpha
    def __call__(self, x: cp.ndarray, learning: bool=False) -> Tuple[cp.ndarray, cp.ndarray]:
        mask = x>0
        x[~mask] *= self.weight.alpha
        self.weight.mask = mask if learning else None
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        if self.weight.get('mask', None) is None:
            raise Exception('At least one forward pass needed before backprop.')
        grad[~self.weight.mask] *= self.weight.alpha
        self.grad = grad
        return grad

class Softmax(Activation):
    def __call__(self, x: cp.ndarray, learning: bool=False) -> Tuple[cp.ndarray, cp.ndarray]:
        x = cp.exp(x-cp.max(x, axis=-1, keepdims=True))
        x /= cp.sum(x, axis=-1, keepdims=True)
        self.act['x'] = x if learning else None
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        grad *= self.act.x * (1-self.act.x)
        return grad