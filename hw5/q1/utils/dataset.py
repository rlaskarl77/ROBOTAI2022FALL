from __future__ import annotations
import os
import cupy as cp
from typing import Any, List, Optional, Sequence, Union
from pprint import pprint
from easydict import EasyDict as edict
import random

from utils.common import one_hot, padding

class Dataset(object):
    def __init__(self,
            file_name: str,
            file_length: Optional[int]=None,
        ) -> None:

        self.training = False

        file_nme = os.path.join(os.getcwd(), file_name)

        file = open(file_name, 'r')
        data_list = file.readlines()
        if file_length is not None:
            try:
                data_list = data_list[:file_length]
            except IndexError:
                print('file is shorter than expected.')
        
        self.labels = []
        self.images = []

        for data in data_list:
            data = data.split(',')
            self.labels.append(int(data[0]))
            self.images.append(list(map(float, data[1:])))

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        image = self.images[idx]

        label = one_hot(label, 10)
        image = cp.array(image, dtype=cp.float).reshape(1, 28, 28)

        if self.training:
            # image = self.augment_image(image)
            label = self.augment_label(label)
            pass

        # print(image)
        # print(label)

        return (image, label)
    
    def __len__(self):
        return len(self.labels)

    def augment_image(self, image):

        c, h, w = image.shape

        # random translation
        # new_image = cp.zeros_like(image)

        p1 = random.randint(-2, 3)
        p2 = random.randint(-2, 3)

        p1 = random.randint(-2, 3)
        # p11 = (int(abs(p1))-p1)//2
        # p12 = (int(abs(p1))+p1)//2
        p11 = p12 = abs(p1)

        p2 = random.randint(-2, 3)
        # p21 = (int(abs(p2))-p2)//2
        # p22 = (int(abs(p2))+p2)//2
        p21 = p22 = abs(p2)

        # print(p11, p12, p21, p22, p1, p2)

        pad_width = [(0,0), (p11,p12), (p21,p22)]    
        image = cp.pad(image, pad_width, 'constant', constant_values=0)
        image = image[:, 0:h, 0:w]

        # for i in range(h):
        #     for j in range(w):
        #         if (i+p1)<0 or (j+p2)<0 or (i+p1)>=h or (j+p2)>=w:
        #             continue
        #         # print(new_image[:, i:i+1, j:j+1].shape)
        #         # print(image[:, i+p1:i+p1+1, j+p2:j+p2+1].shape)
        #         # print(i, j, p1, p2)
        #         new_image[:, i, j] = image[:, i+p1, j+p2]

        return image
    
    def augment_label(self, label):
        alpha = 0.01
        label = (1-alpha)*label + alpha*cp.ones_like(label)
        # label /= cp.sum(label)
        return label

class DataLoader(object):
    def __init__(self,
            dataset: Dataset,
            training: bool=False,
            batch_size: int=200,
        ) -> None:
        self.dataset = dataset
        self.dataset.training = training
        self.training = training
        self.n_data = len(dataset)

        self.batch_size = batch_size
        self.iter_size = cp.ceil(self.n_data/self.batch_size)

        self.batch_idx = 0
        self.idxes = cp.arange(self.n_data)

        # print(self.iter_size)
        # print(self.n_data)
    
    def __iter__(self):
        self.batch_idx = 0
        if self.training:
            self.idxes = cp.random.permutation(cp.arange(self.n_data))
        else:
            self.idxes = cp.arange(self.n_data)
        return self
    
    def __next__(self):
        if self.batch_idx >= self.iter_size:
            raise StopIteration

        batch_from = self.batch_idx * self.batch_size
        batch_to = min(self.n_data, (self.batch_idx+1) * self.batch_size)
        idx_list = self.idxes[batch_from:batch_to]

        # print(idx_list)
        self.batch_idx += 1

        return self.get_batch(idx_list)
    
    def get_batch(self, idx_list: cp.ndarray):
        labels = []
        images = []

        for idx in idx_list:
            image, label = self.dataset[int(idx)]
            labels.append(label)
            images.append(image)
        
        labels = cp.stack(labels, axis=0)
        images = cp.stack(images, axis=0)

        # print(labels.shape)
        # print(images.shape)

        return images, labels
