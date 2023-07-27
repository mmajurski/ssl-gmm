import os
import numpy as np
import torch
import torch.utils.data
import time
import logging
import psutil
import inspect

import cifar_datasets
import utils
import embedding_constraints


MAX_EPOCHS = 10000


class SupervisedTrainer:

    # allow default collate function to work
    collate_fn = None

    def __init__(self, args):
        self.args = args

    # adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py#L270
    def configure_optimizer(self, model):

        if self.args.weight_decay is None:
            weight_decay = 0.0
        else:
            weight_decay = self.args.weight_decay

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info("num decayed parameter tensors: {}, with {} parameters".format(len(decay_params), num_decay_params))
        logging.info("num non-decayed parameter tensors: {}, with {} parameters".format(len(nodecay_params), num_nodecay_params))

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optim_groups, lr=self.args.learning_rate, momentum=0.9, nesterov=False)
            logging.info("Using SGD")

        elif self.args.optimizer == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            extra_args = dict(fused=True) if fused_available else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, **extra_args)
            logging.info("Using fused AdamW: {}".format(fused_available))
        else:
            raise RuntimeError("Invalid optimizer: {}".format(self.args.optimizer))
        return optimizer


    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, unlabeled_dataset=None, ema_model=None):

        model.train()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        batch_count = len(dataloader)

        if self.args.cycle_factor is None or self.args.cycle_factor == 0:
            cyclic_lr_scheduler = None
        else:
            epoch_init_lr = optimizer.param_groups[0]['lr']
            train_stats.add(epoch, 'learning_rate', epoch_init_lr)
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(epoch_init_lr / self.args.cycle_factor), max_lr=(epoch_init_lr * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        start_time = time.time()
        loss_nan_count = 0

        embedding_criterion = torch.nn.MSELoss()
        if self.args.embedding_constraint is None or self.args.embedding_constraint.lower() == 'none':
            emb_constraint = None
        elif self.args.embedding_constraint == 'mean_covar':
            emb_constraint = embedding_constraints.MeanCovar()
        elif self.args.embedding_constraint == 'mean_covar2':
            emb_constraint = embedding_constraints.MeanCovar2(embedding_dim=self.args.embedding_dim, num_classes=self.args.num_classes)
        elif self.args.embedding_constraint == 'gauss_moment':
            emb_constraint = embedding_constraints.GaussianMoments(embedding_dim=self.args.embedding_dim, num_classes=self.args.num_classes)
        elif self.args.embedding_constraint == 'l2':
            emb_constraint = embedding_constraints.L2ClusterCentroid()
        else:
            raise RuntimeError("Invalid embedding constraint type: {}".format(self.args.embedding_constraint))

        for batch_idx, tensor_dict in enumerate(dataloader):

            optimizer.zero_grad()

            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            embedding, logits = model(inputs)
            # resp_gmm, resp_cmm, cluster_dist = model(inputs)
            # outputs = resp_gmm

            batch_loss = criterion(logits, labels)
            if emb_constraint is not None:
                emb_constraint_vals = emb_constraint(embedding, model.last_layer.centers, logits)
                emb_constraint_loss = embedding_criterion(emb_constraint_vals, torch.zeros_like(emb_constraint_vals))
                train_stats.append_accumulate('train_embedding_constraint_loss', emb_constraint_loss.item())
                batch_loss += emb_constraint_loss

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if cyclic_lr_scheduler is not None:
                cyclic_lr_scheduler.step()

            pred = torch.argmax(logits, dim=-1)
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

        if emb_constraint is not None:
            train_stats.close_accumulate(epoch, 'train_embedding_constraint_loss', method='avg', default_value=0.0)  # default value in case no data
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

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        batch_count = len(dataloader)
        model.eval()
        start_time = time.time()

        cluster_criterion = torch.nn.MSELoss()
        if self.args.embedding_constraint is None or self.args.embedding_constraint.lower() == 'none':
            emb_constraint = None
        elif self.args.embedding_constraint == 'mean_covar':
            emb_constraint = embedding_constraints.MeanCovar()
        elif self.args.embedding_constraint == 'mean_covar2':
            emb_constraint = embedding_constraints.MeanCovar2(embedding_dim=self.args.embedding_dim, num_classes=self.args.num_classes)
        elif self.args.embedding_constraint == 'gauss_moment':
            emb_constraint = embedding_constraints.GaussianMoments(embedding_dim=self.args.embedding_dim, num_classes=self.args.num_classes)
        elif self.args.embedding_constraint == 'l2':
            emb_constraint = embedding_constraints.L2ClusterCentroid()
        else:
            raise RuntimeError("Invalid embedding constraint type: {}".format(self.args.embedding_constraint))

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                embedding, logits = model(inputs)
                # resp_gmm, resp_cmm, cluster_dist = model(inputs)
                # outputs = resp_gmm

                batch_loss = criterion(logits, labels)
                train_stats.append_accumulate('{}_loss'.format(split_name), batch_loss.item())
                pred = torch.argmax(logits, dim=-1)
                accuracy = torch.sum(pred == labels) / len(pred)
                train_stats.append_accumulate('{}_accuracy'.format(split_name), accuracy.item())

                if emb_constraint is not None:
                    # only include a "logit" loss, when there are other terms
                    train_stats.append_accumulate('{}_logit_loss'.format(split_name), batch_loss.item())
                    emb_constraint_l = emb_constraint(embedding, model.last_layer.centers, logits)
                    emb_constraint_loss = cluster_criterion(emb_constraint_l, torch.zeros_like(emb_constraint_l))
                    train_stats.append_accumulate('{}_emb_constraint_loss'.format(split_name), emb_constraint_loss.item())
                    batch_loss += emb_constraint_loss

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
        if emb_constraint is not None:
            train_stats.close_accumulate(epoch, '{}_logit_loss'.format(split_name), method='avg')
            train_stats.close_accumulate(epoch, '{}_emb_constraint_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_accuracy'.format(split_name), method='avg')
