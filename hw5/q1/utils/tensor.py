from __future__ import annotations
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import IPython.display as ipd
import os
import shutil
from tqdm import tqdm
from typing import Optional, Sequence, Union
from pprint import pprint

class Tensor(object):
    def __init__(
        self, 
        x: Union[np.ndarray, cp.ndarray], 
        require_grad: bool = True,
        dtype: Union[np.dtype, cp.dtype] = cp.float32,
        grad: Optional[cp.ndarray] = None
    ):
        self.dtype = dtype
        self.value = cp.array(x, dtype=dtype)
        self.require_grad = require_grad or grad is not None
        self.grad = None if not require_grad \
                    else grad if grad is not None \
                    else cp.zeros_like(x)

        self.shape = self.value.shape
    
    def __len__(self):
        return self.value.__len__()
    
    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def __le__(self, other):
        return self.value.__le__(other.value)

    def __eq__(self, other):
        return self.value.__eq__(other.value)

    def __ne__(self, other):
        return self.value.__ne__(other.value)

    def __gt__(self, other):
        return self.value.__gt__(other.value)

    def __ge__(self, other):
        return self.value.__ge__(other.value)

    def __bool__(self):
        return self.value.__bool__()
        
    def __iter__(self):
        for value in self.value:
            yield value
    
    def __delitem__(self, key):
        self.value.__delattr__(key)

    def __getitem__(self, key):
        return self.value.__getattribute__(key)

    def __setitem__(self, key, value):
        self.value.__setattr__(key, value)
    
    def __reversed__(self):
        return Tensor(self.value.__reversed__(), self.require_grad, self.dtype)
    
    def __contains__(self, item):
        return self.value.__contains__(item)

    def __str__(self) -> str:
        return self.value.__str__()
    
    # operators
    def __add__(self, other):
        value = self.value.__add__(other.value)
        return Tensor(value)
    def __sub__(self, other):
        value = self.value.__sub__(other.value)
        return Tensor(value)
    def __mul__(self, other):
        value = self.value.__mul__(other.value)
        return Tensor(value)
    def __matmul__(self, other):
        value = self.value.__matmul__(other.value)
        return Tensor(value)
    def __truediv__(self, other):
        value = self.value.__truediv__(other.value)
        return Tensor(value)
    def __floordiv__(self, other):
        value = self.value.__floordiv__(other.value)
        return Tensor(value)
    def __mod__(self, other):
        value = self.value.__mod__(other.value)
        return Tensor(value)
    def __divmod__(self, other):
        value = self.value.__divmod__(other.value)
        return Tensor(value)
    def __pow__(self, other):
        value = self.value.__pow__(other.value)
        return Tensor(value)
    def __lshift__(self, other):
        value = self.value.__lshift__(other.value)
        return Tensor(value)
    def __rshift__(self, other):
        value = self.value.__rshift__(other.value)
        return Tensor(value)
    def __and__(self, other):
        value = self.value.__and__(other.value)
        return Tensor(value)
    def __xor__(self, other):
        value = self.value.__xor__(other.value)
        return Tensor(value)
    def __or__(self, other):
        value = self.value.__or__(other.value)
        return Tensor(value)
    
    def __radd__(self, other):
        value = self.value.__radd__(other.value)
        return Tensor(value)
    def __rsub__(self, other):
        value = self.value.__rsub__(other.value)
        return Tensor(value)
    def __rmul__(self, other):
        value = self.value.__rmul__(other.value)
        return Tensor(value)
    def __rmatmul__(self, other):
        value = self.value.__rmatmul__(other.value)
        return Tensor(value)
    def __rtruediv__(self, other):
        value = self.value.__rtruediv__(other.value)
        return Tensor(value)
    def __rfloordiv__(self, other):
        value = self.value.__rfloordiv__(other.value)
        return Tensor(value)
    def __rmod__(self, other):
        value = self.value.__rmod__(other.value)
        return Tensor(value)
    def __rdivmod__(self, other):
        value = self.value.__rdivmod__(other.value)
        return Tensor(value)
    def __rpow__(self, other):
        value = self.value.__rpow__(other.value)
        return Tensor(value)
    def __rlshift__(self, other):
        value = self.value.__rlshift__(other.value)
        return Tensor(value)
    def __rrshift__(self, other):
        value = self.value.__rrshift__(other.value)
        return Tensor(value)
    def __rand__(self, other):
        value = self.value.__rand__(other.value)
        return Tensor(value)
    def __rxor__(self, other):
        value = self.value.__rxor__(other.value)
        return Tensor(value)
    def __ror__(self, other):
        value = self.value.__ror__(other.value)
        return Tensor(value)

    def __iadd__(self, other):
        value = self.value.__iadd__(other.value)
        return Tensor(value)
    def __isub__(self, other):
        value = self.value.__isub__(other.value)
        return Tensor(value)
    def __imul__(self, other):
        value = self.value.__imul__(other.value)
        return Tensor(value)
    def __imatmul__(self, other):
        value = self.value.__imatmul__(other.value)
        return Tensor(value)
    def __itruediv__(self, other):
        value = self.value.__itruediv__(other.value)
        return Tensor(value)
    def __ifloordiv__(self, other):
        value = self.value.__ifloordiv__(other.value)
        return Tensor(value)
    def __imod__(self, other):
        value = self.value.__imod__(other.value)
        return Tensor(value)
    def __ipow__(self, other):
        value = self.value.__ipow__(other.value)
        return Tensor(value)
    def __ilshift__(self, other):
        value = self.value.__ilshift__(other.value)
        return Tensor(value)
    def __irshift__(self, other):
        value = self.value.__irshift__(other.value)
        return Tensor(value)
    def __iand__(self, other):
        value = self.value.__iand__(other.value)
        return Tensor(value)
    def __ixor__(self, other):
        value = self.value.__ixor__(other.value)
        return Tensor(value)
    def __ior__(self, other):
        value = self.value.__ior__(other.value)
        return Tensor(value)

    def __neg__(self):
        value = self.value.__neg__()
        return Tensor(value)
    def __pos__(self):
        value = self.value.__pos__()
        return Tensor(value)
    def __abs__(self):
        value = self.value.__abs__()
        return Tensor(value)

def zeros(
    shape: Union[Sequence, int],  
    require_grad: bool = False,
    dtype: Union[np.dtype, cp.dtype] = cp.float32
    ) -> Tensor:
    return Tensor(cp.zeros(shape), require_grad, dtype)

def ones(
    shape: Union[Sequence, int],  
    require_grad: bool = False,
    dtype: Union[np.dtype, cp.dtype] = cp.float32
    ) -> Tensor:
    return Tensor(cp.ones(shape), require_grad, dtype)

def zeros_like(
    x: Union[np.ndarray, cp.ndarray], 
    require_grad: bool = False,
    dtype: Union[np.dtype, cp.dtype] = cp.float32
    ) -> Tensor:
    return Tensor(cp.zeros_like(x), require_grad, dtype)

def ones_like(
    x: Union[np.ndarray, cp.ndarray], 
    require_grad: bool = False,
    dtype: Union[np.dtype, cp.dtype] = cp.float32
    ) -> Tensor:
    return Tensor(cp.ones_like(x), require_grad, dtype)

def add_bias(x: Tensor):
    shape = x.shape
    bias = cp.ones((shape[:-1], 1))
    value = x.value.concatenate((x.value, bias), axis=-1)
    if x.grad is not None:
        grad_bias = cp.zeros((shape[:-1], 1))
        grad = x.grad.concatenate((x.grad, grad_bias), axis=-1)
    return Tensor(value, grad=grad)
