import os
import numpy as np
import torch
import torch.utils.data
import time
import logging
import psutil

import cifar_datasets
import utils


MAX_EPOCHS = 10000


class SupervisedTrainer:

    # allow default collate function to work
    collate_fn = None

    def __init__(self, args):
        self.args = args

    def get_optimizer(self, model):
        # Setup optimizer
        if self.args.weight_decay is not None:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=False)
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            else:
                raise RuntimeError("Invalid optimizer: {}".format(self.args.optimizer))
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=False)
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)
            else:
                raise RuntimeError("Invalid optimizer: {}".format(self.args.optimizer))
        return optimizer

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, unlabeled_dataset=None, ema_model=None):

        model.train()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = len(dataloader)

        if self.args.cycle_factor is None or self.args.cycle_factor == 0:
            cyclic_lr_scheduler = None
        else:
            epoch_init_lr = optimizer.param_groups[0]['lr']
            train_stats.add(epoch, 'learning_rate', epoch_init_lr)
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(epoch_init_lr / self.args.cycle_factor), max_lr=(epoch_init_lr * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        start_time = time.time()
        loss_nan_count = 0

        for batch_idx, tensor_dict in enumerate(dataloader):

            optimizer.zero_grad()

            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            # outputs = model(inputs)
            resp_gmm, resp_cmm, cluster_dist = model(inputs)
            outputs = resp_gmm

            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            optimizer.step()
            if cyclic_lr_scheduler is not None:
                cyclic_lr_scheduler.step()

            pred = torch.argmax(outputs, dim=-1)
            accuracy = torch.sum(pred == labels) / len(pred)

            # nan loss values are ignored when using AMP, so ignore them for the average
            if not np.isnan(batch_loss.item()):
                train_stats.append_accumulate('train_loss', batch_loss.item())
                train_stats.append_accumulate('train_accuracy', accuracy.item())
            else:
                loss_nan_count += 1

            if loss_nan_count > int(0.5 * batch_count):
                raise RuntimeError("Loss is consistently nan (>50% of epoch), terminating train.")

            if ema_model is not None:
                ema_model.update(model)

            if batch_idx % 100 == 0:
                # log loss and current GPU utilization
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        wall_time = time.time() - start_time

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.add(epoch, 'train_wall_time', wall_time)
        train_stats.close_accumulate(epoch, 'train_loss', method='avg')
        train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')

        if cyclic_lr_scheduler is not None:
            # reset any leftover changes to the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = epoch_init_lr


    def eval_model(self, model, pytorch_dataset, criterion, train_stats, split_name, epoch, args):
        if pytorch_dataset is None or len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = len(dataloader)
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                # outputs = model(inputs)
                # resp_gmm, resp_cmm, cluster_dist = model(inputs)
                # outputs = resp_gmm
                outputs = model(inputs)

                batch_loss = criterion(outputs, labels)
                train_stats.append_accumulate('{}_loss'.format(split_name), batch_loss.item())
                pred = torch.argmax(outputs, dim=-1)
                accuracy = torch.sum(pred == labels) / len(pred)
                train_stats.append_accumulate('{}_accuracy'.format(split_name), accuracy.item())

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
        train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_accuracy'.format(split_name), method='avg')
