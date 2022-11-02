from __future__ import annotations
import cupy as cp
from typing import Any, Optional, Sequence, Union
from pprint import pprint
from easydict import EasyDict as edict

import utils.activation as ac
from utils.common import *
import utils.module as m

class Loss(object):
    def __init__(self) -> None:
        self.act = edict()
    
    def __call__(self, output: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        raise NotImplementedError()
    
    def backprop(self, output: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        raise NotImplementedError()

class MSE(Loss):
    def __call__(self, output: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        n_b = output.shape[0]
        return cp.sum((output-y)**2)/n_b

    def backprop(self, output: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        n_b = output.shape[0]
        return (output-y)/n_b

class CrossEntropyWithLogits(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = ac.Softmax()

    def __call__(self, output: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        n_b = output.shape[0]
        output = self.softmax(output, learning=True)
        self.act['output'] = output
        eps = 1e-9
        output = - cp.sum(cp.log(output+eps) * y)/n_b
        return output

    def backprop(self, output: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
        n_b = output.shape[0]
        return (self.act.output-y)/n_b