import copy
import sys
import os

import numpy as np
import torch
import torchvision
import logging

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


# due to pytorch + numpy bug
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# TODO setup train/val/test and use this in place of the orig dataset function
class Cifar10(torch.utils.data.Dataset):
    TRANSFORM_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32 * 0.125),
                                          padding_mode='reflect'),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    TRANSFORM_TEST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    def __init__(self, transforms=None, train:bool=True, lcl_fldr:str='./data'):
        self.lcl_fldr = lcl_fldr
        self._transforms = transforms
        if train:
            self._dataset = torchvision.datasets.CIFAR10(self.lcl_fldr, train=True, download=True, transform=self._transforms)
        else:
            self._dataset = torchvision.datasets.CIFAR10(self.lcl_fldr, train=False, download=True, transform=self._transforms)

    def train_val_split(self, val_fraction: float = 0.2):
        train_fraction = 1.0 - val_fraction
        if train_fraction <= 0.0:
            raise RuntimeError("Train fraction too small. {}% of the data was allocated to training split, {}% to validation split.".format(int(train_fraction * 100), int(val_fraction * 100)))
        logging.info("Train data fraction: {}, Validation data fraction: {}".format(train_fraction, val_fraction))

        # use 90% of the train for train, and 10% for val
        val_size = int(val_fraction * len(self._dataset))
        train_size = len(self._dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(self._dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def __getitem__(self, index):
        return self._dataset.__getitem__(index)

    def __len__(self):
        return self._dataset.__len__()



class Cifar100(torch.utils.data.Dataset):
    TRANSFORM_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32 * 0.125),
                                          padding_mode='reflect'),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    TRANSFORM_TEST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    def __init__(self, transforms=None, train:bool=True, lcl_fldr:str='./data'):
        self.lcl_fldr = lcl_fldr
        self._transforms = transforms
        if train:
            self._dataset = torchvision.datasets.CIFAR100(self.lcl_fldr, train=True, download=True, transform=self._transforms)
        else:
            self._dataset = torchvision.datasets.CIFAR100(self.lcl_fldr, train=False, download=True, transform=self._transforms)

    def train_val_split(self, val_fraction: float = 0.2):
        train_fraction = 1.0 - val_fraction
        if train_fraction <= 0.0:
            raise RuntimeError("Train fraction too small. {}% of the data was allocated to training split, {}% to validation split.".format(int(train_fraction * 100), int(val_fraction * 100)))
        logging.info("Train data fraction: {}, Validation data fraction: {}".format(train_fraction, val_fraction))

        # use 90% of the train for train, and 10% for val
        val_size = int(val_fraction * len(self._dataset))
        train_size = len(self._dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(self._dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def __getitem__(self, index):
        return self._dataset.__getitem__(index)

    def __len__(self):
        return self._dataset.__len__()