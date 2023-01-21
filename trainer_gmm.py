import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional
import time
import logging
import psutil

import cifar_datasets
import utils
import trainer


class FixMatchTrainer_gmm(trainer.SupervisedTrainer):

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, nb_reps=1, unlabeled_dataset=None):
        pass
