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

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, emb_constraint, epoch, train_stats, unlabeled_dataset=None, ema_model=None, save_embedding=False, output_dirpath="./model"):

        model.train()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        batch_count = len(dataloader)

        cyclic_lr_scheduler = None
        # cyclic learning rate with one up/down cycle per epoch.
        if self.args.cycle_factor is not None and self.args.cycle_factor > 0:
            epoch_init_lr = optimizer.param_groups[0]['lr']
            train_stats.add(epoch, 'learning_rate', epoch_init_lr)
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(epoch_init_lr / self.args.cycle_factor), max_lr=(epoch_init_lr * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        start_time = time.time()
        loss_nan_count = 0

        embedding_criterion = torch.nn.MSELoss()

        embedding_output = []
        labels_output    = []

        for batch_idx, tensor_dict in enumerate(dataloader):

            optimizer.zero_grad()

            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            if hasattr(model, "last_layer") and hasattr(model.last_layer, 'centers'):
                model.last_layer.centers = model.last_layer.centers.cuda()

            embedding, logits = model(inputs)
            # resp_gmm, resp_cmm, cluster_dist = model(inputs)
            # outputs = resp_gmm
            #print("embedding",embedding.shape)

            batch_loss = criterion(logits, labels)
            if emb_constraint is not None:
                emb_constraint_vals = emb_constraint(embedding, model.last_layer.centers, logits)
                emb_constraint_loss = embedding_criterion(emb_constraint_vals, torch.zeros_like(emb_constraint_vals))
                train_stats.append_accumulate('train_embedding_constraint_loss', emb_constraint_loss.item())
                batch_loss += emb_constraint_loss

            batch_loss.backward()
            if self.args.clip_grad:
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

            if save_embedding:
                embedding_output.append( embedding.detach().cpu().numpy() )
                labels_output.append(  labels.detach().cpu().numpy() )

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

        if save_embedding:
            embedding_output = utils.multiconcat_numpy(embedding_output)
            labels_output    = utils.multiconcat_numpy(labels_output)
            outpath = output_dirpath + "/train_embedding.npy"
            logging.info("save " + outpath)
            np.save(outpath, embedding_output)
            outpath = output_dirpath + "/train_labels.npy"
            logging.info("save " + outpath)
            np.save(outpath, labels_output)


    def eval_model(self, model, pytorch_dataset, criterion, train_stats, split_name, emb_constraint, epoch, args, output_dirpath="./model"):
        if pytorch_dataset is None or len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        batch_count = len(dataloader)
        model.eval()
        start_time = time.time()

        embedding_criterion = torch.nn.MSELoss()
        
        embedding_output_test = []
        labels_output_test    = []

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
                    train_stats.append_accumulate('{}_emb_constraint_loss'.format(split_name), emb_constraint_loss.item())
                    emb_constraint_loss = embedding_criterion(emb_constraint_l, torch.zeros_like(emb_constraint_l))
                    batch_loss += emb_constraint_loss

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))
                    
                if args.save_embedding:
                    embedding_output_test.append( embedding.detach().cpu().numpy() )
                    labels_output_test.append(  labels.detach().cpu().numpy() )


        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
        if emb_constraint is not None:
            train_stats.close_accumulate(epoch, '{}_logit_loss'.format(split_name), method='avg')
            train_stats.close_accumulate(epoch, '{}_emb_constraint_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_accuracy'.format(split_name), method='avg')
        
        
        if args.save_embedding:
            embedding_output_test = utils.multiconcat_numpy(embedding_output_test)
            outpath = output_dirpath + "/test_embedding.npy"
            logging.info("save " + outpath)
            np.save(outpath, embedding_output_test)
            
            labels_output_test    = utils.multiconcat_numpy(labels_output_test)
            outpath = output_dirpath + "/test_labels.npy"
            logging.info("save " + outpath)
            np.save(outpath, labels_output_test)
            
            
