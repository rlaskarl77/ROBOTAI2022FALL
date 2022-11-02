from __future__ import annotations
import cupy as cp
from typing import Optional, Sequence, Union
from easydict import EasyDict as edict

def autopad(padding: Optional[int]=None, k: Optional[int]=3) -> int:
    if padding is None:
        padding = (k-1)//2
    return padding

def padding(x: cp.ndarray, p: int, pad_value: int=0) -> cp.ndarray:
    b, c, h, w = x.shape
    pad_width = [(0,0), (0,0), (p,p), (p,p)]    
    return cp.pad(x, pad_width, 'constant', constant_values=pad_value)

def convolution(x: cp.ndarray, kernel: cp.ndarray, p: float, s: float) -> cp.ndarray:
    b, c, h, w = x.shape
    c_out, c, k, k = kernel.shape

    h_out = (h-k+2*p)//s+1
    w_out = (w-k+2*p)//s+1

    x = padding(x, p=p, pad_value=0)

    output = cp.zeros((b, c_out, h_out, w_out))

    # print(x.shape)
    # print(kernel.shape)
    # print(output.shape)

    for i in range(h_out):
        for j in range(w_out):
            output[:, :, i, j] = cp.sum(x[:, cp.newaxis, :, i*s:i*s+k, j*s:j*s+k] \
                * kernel[cp.newaxis, :, :, :, :], axis=(2, 3, 4))

    return output

def softmax(x: cp.ndarray) -> cp.ndarray:
    x = cp.exp(x-cp.max(x, axis=-1, keepdims=True))
    x /= cp.sum(x, axis=-1, keepdims=True)
    return x

def one_hot(x: int, size: int) -> cp.ndarray:
    assert x<size, 'one_hot vector size is smaller than index.'
    out = cp.zeros((size,), dtype=cp.float)
    out[x] = 1.
    return out

def sigmoid(x: cp.ndarray) -> cp.ndarray:
    return 1/(1+cp.exp(-x))