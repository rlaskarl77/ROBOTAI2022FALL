from typing import Optional
import torch
import torch.nn as nn
from tqdm import tqdm

def autopad(p, k=3):
    if p is None:
        p = (k-1)//2
    return p

class ConvBlock(nn.Module):
    def __init__(self,
            c_in: int,
            c_out: int,
            k: int=3,
            s: int=1,
            p: Optional[int]=None,
        ) -> None:
        super().__init__()

        p = autopad(p, k)

        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.body = nn.Sequential(
            # (B, 1, 32, 32)
            ConvBlock(1, 16, 3, 1, 1),
            # (B, 16, 32, 32)
            nn.MaxPool2d(2, 2),
            # (B, 16, 16, 16)
            ConvBlock(16, 32, 3, 1, 1),
            # (B, 32, 16, 16)
            nn.MaxPool2d(2, 2),
            # (B, 32, 8, 8)
            ConvBlock(32, 64, 3, 1, 1),
            # (B, 64, 8, 8)
            nn.MaxPool2d(2, 2),
            # (B, 64, 4, 4)
            ConvBlock(64, 128, 3, 1, 1),
            # (B, 128, 4, 4)
            nn.MaxPool2d(2, 2),
            # (B, 128, 2, 2)
            nn.AdaptiveAvgPool2d(1),
            # (B, 128, 1, 1)
            nn.Flatten(),
            # (B, 128)
            nn.Linear(128, 1000),
            # (B, 1000)
            nn.Dropout(0.5),
            # (B, 1000)
            nn.Linear(1000, 10),
            # (B, 10)
        )
    
    def forward(self, x):
        return self.body(x)