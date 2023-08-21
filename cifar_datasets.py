import copy
import sys
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torchvision
import torch.utils.data
import logging
import random
import torchvision.transforms

import fixmatch_augmentation


logger = logging.getLogger()

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class Cifar10(torch.utils.data.Dataset):
    TRANSFORM_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32*0.125),
                                          padding_mode='reflect'),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    TRANSFORM_TEST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    # TRANSFORM_STRONG_TRAIN = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.RandomCrop(size=32,
    #                                       padding=int(32 * 0.125),
    #                                       padding_mode='reflect'),
    #     torchvision.transforms.RandAugment(num_ops=2, magnitude=10),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    # ])
    # TRANSFORM_STRONG_AUGMIX_TRAIN = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.RandomCrop(size=32,
    #                                       padding=int(32 * 0.125),
    #                                       padding_mode='reflect'),
    #     torchvision.transforms.AugMix(),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    # ])

    TRANSFORM_FIXMATCH = fixmatch_augmentation.TransformFixMatch(mean=cifar10_mean, std=cifar10_std)

    def __init__(self, transform=None, train:bool=True, subset=False, lcl_fldr:str='./data', empty=False):

        self.lcl_fldr = lcl_fldr
        self.transform = transform
        self.numel = None
        if empty:
            self.data = list()
            self.targets = list()
        else:
            if train:
                _dataset = torchvision.datasets.CIFAR10(self.lcl_fldr, train=True, download=True)
            else:
                _dataset = torchvision.datasets.CIFAR10(self.lcl_fldr, train=False, download=True)

            self.targets = _dataset.targets
            # break the data up into a list instead of a single numpy block to allow deleting and addition
            self.data = list()
            data_len = _dataset.data.shape[0]
            if subset:
                # for debugging, keep just 10% of the data to accelerate things
                data_len = int(0.1 * data_len)

            for i in range(data_len):
                self.data.append(_dataset.data[i, :, :, :])

            # cleanup the tmp CIFAR object
            del _dataset

    def set_epoch_size(self, epoch_size):
        self.numel = epoch_size
        if epoch_size < len(self.data):
            logging.warning("Requested dataset length = {} is less than actual data length = {}, some elements will be unused.".format(epoch_size, len(self.data)))

    def __len__(self) -> int:
        if self.numel is not None:
            return self.numel
        else:
            return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # account for nb_reps
        index = index % len(self.data)
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target#, index

    def get(self, index:int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target

    def get_raw_datapoint(self, index):
        img = copy.deepcopy(self.data[index])
        target = copy.deepcopy(self.targets[index])
        return img, target

    def set_transforms(self, transforms):
        self.transform = transforms

    def data_split_class_balanced(self, subset_count: int = 400):

        idx = torch.randperm(len(self.data)).detach().cpu().numpy()

        subset_dataset = copy.deepcopy(self)
        remainder_dataset = copy.deepcopy(self)
        subset_dataset.data = list()
        subset_dataset.targets = list()
        remainder_dataset.data = list()
        remainder_dataset.targets = list()

        nb_classes = len(set(self.targets))
        per_class_subset_count = subset_count / nb_classes
        # if per_class_subset_count - int(per_class_subset_count) != 0:
        #     raise RuntimeError("Invalid subset_count = {}, resulted in a non-integer number of examples per class={}".format(subset_count, per_class_subset_count))
        per_class_subset_count = int(per_class_subset_count)

        a_class_instance_count = np.zeros(nb_classes)
        for i in idx:
            t = self.targets[i]
            d = self.data[i]
            if a_class_instance_count[t] < per_class_subset_count:
                subset_dataset.data.append(d)
                subset_dataset.targets.append(t)
                a_class_instance_count[t] += 1
            else:
                remainder_dataset.data.append(d)
                remainder_dataset.targets.append(t)

        return subset_dataset, remainder_dataset

    def train_val_split(self, val_fraction: float = 0.1):
        if val_fraction < 0.0 or val_fraction > 1.0:
            raise RuntimeError("Impossible validation fraction {}.".format(val_fraction))

        val_size = int(val_fraction * len(self.data))

        idx = list(range(len(self.data)))
        random.shuffle(idx)
        v_idx = idx[0:val_size]
        t_idx = idx[val_size:]
        t_idx.sort()
        v_idx.sort()

        train_dataset = copy.deepcopy(self)
        val_dataset = copy.deepcopy(self)
        train_dataset.data = list()
        for i in t_idx:
            train_dataset.data.append(self.data[i])
        train_dataset.targets = list()
        for i in t_idx:
            train_dataset.targets.append(self.targets[i])

        val_dataset.data = list()
        for i in v_idx:
            val_dataset.data.append(self.data[i])
        val_dataset.targets = list()
        for i in v_idx:
            val_dataset.targets.append(self.targets[i])

        return train_dataset, val_dataset

    def append_dataset(self, pytorch_dataset):
        for i in range(len(pytorch_dataset)):
            data = pytorch_dataset.data[i]
            target = pytorch_dataset.targets[i]

            self.data.append(data)
            self.targets.append(target)

    def add(self, data, target):
        self.data.append(data)
        self.targets.append(target)



class Cifar100(Cifar10):

    TRANSFORM_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32 * 0.125),
                                          padding_mode='reflect'),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    TRANSFORM_TEST = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    TRANSFORM_STRONG_TRAIN = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32 * 0.125),
                                          padding_mode='reflect'),
        # fixmatch_augmentation.RandAugmentMC(n=2, m=10),
        torchvision.transforms.RandAugment(num_ops=2, magnitude=10),  # TODO confirm this works as expected
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    def __init__(self, transform=None, train:bool=True, subset=False, lcl_fldr:str='./data'):
        self.lcl_fldr = lcl_fldr
        self.transform = transform
        if train:
            _dataset = torchvision.datasets.CIFAR100(self.lcl_fldr, train=True, download=True)
        else:
            _dataset = torchvision.datasets.CIFAR100(self.lcl_fldr, train=False, download=True)

        self.targets = _dataset.targets
        # break the data up into a list instead of a single numpy block to allow deleting and addition
        self.data = list()
        data_len = _dataset.data.shape[0]
        if subset:
            # for debugging, keep just 10% of the data to accelerate things
            data_len = int(0.1 * data_len)

        for i in range(data_len):
            self.data.append(_dataset.data[i, :, :, :])

        # cleanup the tmp CIFAR object
        del _dataset


class Cifar10plus100(Cifar10):

    def __init__(self, transform=None, train:bool=True, subset=False, lcl_fldr:str='./data'):
        super(Cifar10plus100, self).__init__(transform, train, subset, lcl_fldr)

    def add_cifar100_ood_data(self, p=0.1):
        # get some classes from the cifar100 training dataset
        # this will only be used to contaminate the unlabeled data
        _dataset = torchvision.datasets.CIFAR100(self.lcl_fldr, train=True, download=True)

        # break the data up into a list instead of a single numpy block to allow deleting and addition
        self.ood_targets = list()
        self.ood_data = list()
        data_len = _dataset.data.shape[0]
        idx = list(range(data_len))
        random.shuffle(idx)
        idx = idx[0:int(p * len(self.data))]
        for i in idx:
            self.ood_data.append(_dataset.data[i, :, :, :])
            self.ood_targets.append(_dataset.targets[i] + 100)  # offset by 100 to indicate its coming from cifar100
        # cleanup the tmp CIFAR object
        del _dataset

        for i in range(len(self.ood_data)):
            data = self.ood_data[i]
            target = self.ood_targets[i]

            self.data.append(data)
            self.targets.append(target)

