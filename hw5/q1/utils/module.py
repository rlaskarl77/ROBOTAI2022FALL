from __future__ import annotations
import cupy as cp
from typing import Optional, Sequence, Union
from easydict import EasyDict as edict

import utils.activation as ac
from utils.common import *



class Layer(object):
    NUM_LAYERS = 0
    def __init__(self):
        self.weight = edict()
        self.grad = edict()
        self.act = edict()

        self.id = Layer.NUM_LAYERS
        Layer.NUM_LAYERS += 1
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        '''
        x.shape = B, ...
        return: B, ...
        '''
        raise NotImplementedError()
        
    def backward(self):
        raise NotImplementedError()
    
    def update(self, delta: edict, lr: float):
        for k in delta.keys():
            self.weight[k] += lr * delta[k]
    
    def zero_grad(self):
        for k in self.grad.keys():
            self.grad[k] = cp.zeros_like(self.grad[k])
    
    def _calculate_out_dim(self, in_dim:Union[Sequence, int]):
        x = cp.zeros((1, *in_dim)) if isinstance(in_dim, Sequence) else cp.zeros((1, in_dim))
        x = self.forward(x)
        out_dim = x.shape[1:]
        return out_dim
    
    def initialize_He(self, weight: cp.ndarray, out_dim: int=1):
        n_out = weight.shape[out_dim]
        weight = cp.random.randn(*weight.shape) * cp.sqrt(2/n_out)
        return weight

    def initialize_Xavier(self, weight: cp.ndarray, in_dim: int=0, out_dim: int=1):
        n_in, n_out = weight.shape[in_dim, out_dim]
        weight = cp.random.randn(*weight.shape) * cp.sqrt(2/(n_in+n_out))
        return weight


class Perceptron(Layer):
    def __init__(self, 
            in_dim: int,
            out_dim: int,
            activation: ac.Activation = ac.ReLU,
        ):
        super().__init__()
        self.out_dim = out_dim

        self.weight['w'] = cp.ones((in_dim, out_dim))
        self.weight['w'] = self.initialize_He(self.weight['w'])

        self.weight['b'] = cp.ones((1, out_dim))
        self.weight['b'] = self.initialize_He(self.weight['b'])

        self.activation = activation()
    
    def forward(self, x: cp.ndarray, learning: bool=False):
        if learning:
            self.act['x'] = x
        out = self.activation(x @ self.weight.w + self.weight.b, learning=learning)
        return out
        
    def backward(self, grad: cp.ndarray):
        grad = self.activation.backward(grad)

        n_b = grad.shape[0]
        self.grad['w'] = self.act.x.T @ grad / n_b
        self.grad['b'] = cp.sum(grad, axis=0, keepdims=True) / n_b
        grad_out = grad @ self.weight.w.T

        return grad_out

    def add_bias(self, x: cp.ndarray):
        shape = x.shape
        bias = cp.ones((*shape[:-1], 1))
        x = cp.concatenate((x, bias), axis=-1)
        return x
    
    def __str__(self) -> str:
        return f'Perceptron_{self.weight.w.shape}_{self.id}'

class Conv2d(Layer):
    def __init__(self, 
            in_channel: int,
            out_channel: int,
            padding: Optional[int]=None,
            stride: int=1,
            kernel: int=3,
            activation: ac.Activation=ac.ReLU
        ):
        super().__init__()
        self.c_in = in_channel
        self.c_out = out_channel
        self.s = stride
        self.k = kernel
        self.p = autopad(padding, kernel)

        self.weight['kernel'] = cp.random.randn(out_channel, in_channel, kernel, kernel) * 0.1
        self.weight['bias'] = cp.random.randn(1, out_channel, 1, 1) * 0.1

        self.activation = activation()
    
    def forward(self, x:cp.ndarray, learning=False) -> cp.ndarray:
        if learning:
            self.act['x'] = x
        out = convolution(x, self.weight.kernel, self.p, self.s)
        out = self.activation(out + self.weight.bias, learning=learning)
        self.out_shape = out.shape
        return out
        
    def backward(self, grad: cp.ndarray):
        # print(grad.shape)
        # print(self.act.x.shape)
        # print(self.out_shape)
        # print(self.activation.weight.mask.shape)
        grad = self.activation.backward(grad)

        n_b = grad.shape[0]
        b, c, h, w = self.act.x.shape
        c_out, c, k, k = self.weight.kernel.shape

        h_out = (h-k+2*self.p)//self.s+1
        w_out = (w-k+2*self.p)//self.s+1

        self.act.x = padding(self.act.x, self.p, pad_value=0)

        self.grad['kernel'] = cp.zeros_like(self.weight.kernel)
        self.grad['bias'] = cp.zeros_like(self.weight.bias)
        grad_out = cp.zeros_like(self.act.x)

        for i in range(h_out):
            for j in range(w_out):
                h_from, h_to = i*self.s, i*self.s + k
                w_from, w_to = j*self.s, j*self.s + k

                self.grad.kernel += cp.sum(grad[:, :, cp.newaxis, i:i+1, j:j+1] \
                        * self.act.x[:, cp.newaxis, :, h_from:h_to, w_from:w_to], axis=(0,))
                
                grad_out[:, :, h_from:h_to, w_from:w_to] += \
                    cp.sum(grad[:, :, cp.newaxis, i:i+1, j:j+1] \
                        * self.weight.kernel[cp.newaxis, :, :, :, :], axis=(1,))
        
        self.grad.kernel /= n_b
        self.grad.bias += cp.sum(grad[:, :, :, :], axis=(0, 2, 3), keepdims=True) / n_b

        return grad_out[:, :, self.p:self.p+h, self.p:self.p+w]

    def __str__(self) -> str:
        return f'Convolution_{self.weight.kernel.shape}_{self.id}'

class MaxPool2d(Layer):
    def __init__(self,
            kernel: int=2,  
            stride: int=2,
            padding: int=0,      
        ):
        super().__init__()
        self.k = kernel
        self.s = stride
        self.p = padding

    def forward(self, x: cp.ndarray, learning: bool=False):
        b, c, h, w = x.shape
        x = padding(x, self.p, pad_value=-1e9)
        self.weight['mask'] = cp.zeros_like(x, dtype=cp.int)

        _, _, h_, w_ = x.shape

        h_start = w_start = (self.k-1)//2
        h_end = w_end = h_-(self.k-1)//2-1

        h_out = w_out = (h_end-1-h_start)//self.s+1


        out = cp.zeros((b, c, h_out, w_out))

        self.x_shape = x.shape
        self.out_shape = out.shape

        for i in range(h_out):
            for j in range(w_out):

                h_from, h_to = i*self.s, min(h_, i*self.s+self.k)
                w_from, w_to = j*self.s, min(w_, j*self.s+self.k)

                out[:, :, i:i+1, j:j+1] = \
                        cp.amax(x[:, :, h_from:h_to, w_from:w_to], axis=(2, 3), keepdims=True)
                
                if not learning:
                    continue

                self.weight.mask[:, :, h_from:h_to, w_from:w_to] = \
                        x[:, :, h_from:h_to, w_from:w_to] \
                        == out[:, :, i:i+1, j:j+1]

        self.weight.mask = self.weight.mask[:, :, self.p:h_-self.p, self.p:w_-self.p]

        return out

    def __str__(self) -> str:
        return f'MaxPool_{self.k}_{self.s}_{self.p}_{self.id}'
        
    def backward(self, grad: cp.ndarray):
        b, c, h_, w_ = self.x_shape
        _, _, h_out, w_out = self.out_shape

        grad_out = cp.zeros((b, c, h_, w_))

        for i in range(h_out):
            for j in range(w_out):

                h_from, h_to = i*self.s, min(h_, i*self.s+self.k)
                w_from, w_to = j*self.s, min(w_, j*self.s+self.k)

                grad_out[:, :, h_from:h_to, w_from:w_to] = grad[:, :, i:i+1, j:j+1]

        grad_out = grad_out[:, :, self.p:h_-self.p, self.p:w_-self.p]
        grad_out *= self.weight.mask
        return grad_out

class GlobalAveragePool(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: cp.ndarray, learning: bool=False):
        b, c, h, w = x.shape
        self.x_shape = x.shape
        out = cp.sum(x, axis=(2, 3)) / (h*w)
        return out
        
    def backward(self, grad: cp.ndarray):
        b, c, h, w = self.x_shape
        grad = cp.broadcast_to(grad[:, :, cp.newaxis, cp.newaxis], self.x_shape)
        return grad / (h*w)

    def __str__(self) -> str:
        return f'GAP_{self.id}'

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: cp.ndarray, learning=False):
        self.x_shape = x.shape
        n_b = x.shape[0]
        return x.reshape(n_b, -1)

    def backward(self, grad: cp.ndarray):
        return grad.reshape(*self.x_shape)

    def __str__(self) -> str:
        return f'Flatten_{self.id}'

class Dropout(Layer):
    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: cp.ndarray, learning=False):
        if learning:
            self.weight['mask'] = cp.random.random(x.shape) < self.p
            x[self.weight.mask] = 0
        else:
            x *= 1-self.p
        return x
    
    def backward(self, grad: cp.ndarray):
        grad[self.weight.mask] = 0
        return grad

    def __str__(self) -> str:
        return f'Dropout_{self.p}_{self.id}'

class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        self.weight['lmbda'] = None
        self.weight['beta'] = None
        self.eps = 1e-8
    
    def forward(self, x: cp.ndarray, learning=False):
        if learning:
            if self.weight.lmbda is None:
                self.weight.lmbda = cp.ones_like(x)
                self.weight.beta = cp.zeros_like(x)
            mu = cp.mean(x, axis=0, keepdims=True)
            sigma = cp.var(x, axis=0, keepdims=True)

            x = (x-mu)/cp.sqrt(sigma+self.eps)

            y = self.weight.lmbda*x + self.weight.beta

        else:
            x *= 1-self.p
        return x
    
    def backward(self, grad: cp.ndarray):
        grad[self.weight.mask] = 0
        return grad

    def __str__(self) -> str:
        return f'Dropout_{self.p}_{self.id}'