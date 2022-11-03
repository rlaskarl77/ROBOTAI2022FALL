from cProfile import label
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self,
            data_path: str,
            training: bool=False,
            out_dim: int=10
        ) -> None:
        super().__init__()

        with open(data_path, 'r') as f:
            self.datas = f.readlines()
        
        self.out_dim = out_dim
        self.training = training

        self.n = len(self.datas)
        self.indexes = np.arange(self.n)

        self.mixup = True
        self.mixup_alpha = 1.

        self.label_smoothing = True
        self.label_smoothing_alpha = 0.1

        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(0.4720751017922801, 0.23932276261069446),
        ])

        self.augmentations = tf.Compose([
            tf.RandomHorizontalFlip(0.5),
            tf.RandomVerticalFlip(0.5),
            tf.RandomCrop((32, 32), padding=2),
        ])
        
    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        image, label = self.getImageAndLabel(idx)

        if self.training:
            image = self.augmentations(image)

            if self.mixup:
                image, label = self._mixup(image, label, self.mixup_alpha)
            
            if self.label_smoothing:
                label = (1-self.label_smoothing_alpha) * label \
                    + self.label_smoothing_alpha / self.out_dim

        return image, label
    
    def getImageAndLabel(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.datas[idx].rstrip().split(',')
        
        label = torch.from_numpy(np.array(values[0], dtype=int))
        label = F.one_hot(label, num_classes=self.out_dim).float()

        image = np.asfarray(values[1:1025], dtype=np.float32).reshape(32, 32)
        image = self.transforms(image)

        return image, label
    
    def _mixup(self, image: torch.Tensor, label: torch.Tensor, alpha: float=1.) -> Tuple[torch.Tensor, torch.Tensor]:
        idx2 = np.random.choice(self.indexes)
        image2, label2 = self.getImageAndLabel(idx2)

        assert alpha > 0, 'alpha should be greater than 0.'
        lmbda = np.random.beta(alpha, alpha)
        
        mixed_image = lmbda*image + (1-lmbda)*image2
        mixed_label = lmbda*label + (1-lmbda)*label2

        return mixed_image, mixed_label
