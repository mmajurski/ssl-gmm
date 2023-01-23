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
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=True)
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            else:
                raise RuntimeError("Invalid optimizer: {}".format(self.args.optimizer))
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True)
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)
            else:
                raise RuntimeError("Invalid optimizer: {}".format(self.args.optimizer))
        return optimizer

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, nb_reps=1, unlabeled_dataset=None):

        loss_list = list()
        accuracy_list = list()
        model.train()
        scaler = torch.cuda.amp.GradScaler()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = nb_reps * len(dataloader)

        factor = 4.0
        cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(self.args.learning_rate / factor), max_lr=(self.args.learning_rate * factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        start_time = time.time()
        loss_nan_count = 0

        for rep_count in range(nb_reps):
            for batch_idx, tensor_dict in enumerate(dataloader):
                # adjust for the rep offset
                batch_idx = rep_count * len(dataloader) + batch_idx

                optimizer.zero_grad()

                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                # FP16 training
                if self.args.amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)
                        scaler.scale(batch_loss).backward()
                        # scaler.step() first unscales the gradients of the optimizer's assigned params.
                        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                        # otherwise, optimizer.step() is skipped.
                        scaler.step(optimizer)
                        # Updates the scale for next iteration.
                        scaler.update()
                else:
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    batch_loss.backward()
                    optimizer.step()

                pred = torch.argmax(outputs, dim=-1)
                if self.args.soft_labels:
                    # convert soft labels into hard for accuracy
                    labels = torch.argmax(labels, dim=-1)
                accuracy = torch.sum(pred == labels) / len(pred)

                # nan loss values are ignored when using AMP, so ignore them for the average
                if not np.isnan(batch_loss.detach().cpu().numpy()):
                    loss_list.append(batch_loss.item())
                    accuracy_list.append(accuracy.item())
                    cyclic_lr_scheduler.step()
                else:
                    loss_nan_count += 1

                    if loss_nan_count > 500:
                        raise RuntimeError("Loss is consistently nan (>100x per epoch), terminating train.")

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        avg_loss = np.mean(loss_list)
        avg_accuracy = np.mean(accuracy_list)
        wall_time = time.time() - start_time

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.add(epoch, 'learning_rate', optimizer.param_groups[0]['lr'])
        train_stats.add(epoch, 'train_wall_time', wall_time)
        train_stats.add(epoch, 'train_loss', avg_loss)
        train_stats.add(epoch, 'train_accuracy', avg_accuracy)

    def eval_model(self, model, pytorch_dataset, criterion, train_stats, split_name, epoch):
        if pytorch_dataset is None or len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = len(dataloader)
        model.eval()
        start_time = time.time()

        loss_list = list()
        gt_labels = list()
        preds = list()

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                labels_cpu = labels.detach().cpu().numpy()
                gt_labels.extend(labels_cpu)

                if self.args.amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                batch_loss = criterion(outputs, labels)
                loss_list.append(batch_loss.item())
                pred = torch.argmax(outputs, dim=-1).detach().cpu().numpy()
                preds.extend(pred)

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        avg_loss = np.nanmean(loss_list)
        gt_labels = np.asarray(gt_labels)
        softmax_preds = np.asarray(preds)
        softmax_accuracy = np.mean((softmax_preds == gt_labels).astype(float))

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
        train_stats.add(epoch, '{}_loss'.format(split_name), avg_loss)
        train_stats.add(epoch, '{}_accuracy'.format(split_name), softmax_accuracy)

