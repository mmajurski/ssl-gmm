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
import embedding_constraints


def sharpen_mixmatch(x:torch.Tensor, T:float):
    # from https://arxiv.org/pdf/1905.02249.pdf equation 7
    p = 1.0 / float(T)
    neum = torch.pow(x, p)
    denom = torch.sum(neum, dim=-1, keepdim=True)
    return neum / denom


class FixMatchTrainer_gmm(trainer.SupervisedTrainer):

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, unlabeled_dataset=None, ema_model=None):

        if unlabeled_dataset is None:
            raise RuntimeError("Unlabeled dataset missing. Cannot use FixMatch train_epoch function without an unlabeled_dataset.")

        model.train()
        loss_nan_count = 0
        start_time = time.time()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = len(dataloader)

        # cyclic learning rate with one up/down cycle per epoch.
        if self.args.cycle_factor is None or self.args.cycle_factor == 0:
            cyclic_lr_scheduler = None
        else:
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(self.args.learning_rate / self.args.cycle_factor), max_lr=(self.args.learning_rate * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        unlabeled_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH)
        dataloader_ul = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=self.args.mu*self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)
        iter_ul = iter(dataloader_ul)

        # loss_list = list()
        # accuracy_gmm_list = list()
        # accuracy_cmm_list = list()
        # pl_loss_list = list()
        # pl_count_list = list()
        # pl_acc_list = list()
        pl_acc_per_class = list()
        pl_count_per_class = list()
        pl_gt_count_per_class = list()
        for i in range(self.args.num_classes):
            pl_acc_per_class.append(list())
            pl_count_per_class.append(0)
            pl_gt_count_per_class.append(0)

        cluster_criterion = torch.nn.MSELoss()
        if self.args.last_layer == 'kmeans_layer':
            emb_constraint = embedding_constraints.mean_covar
        else:
            emb_constraint = embedding_constraints.l2_cluster_centroid

        for batch_idx, tensor_dict_l in enumerate(dataloader):
            optimizer.zero_grad()

            inputs_l = tensor_dict_l[0]
            targets_l = tensor_dict_l[1]

            try:
                tensor_dict_ul = next(iter_ul)
            except:
                # recreate the iterator
                iter_ul = iter(dataloader_ul)
                tensor_dict_ul = next(iter_ul)

            inputs_ul = tensor_dict_ul[0]
            targets_ul = tensor_dict_ul[1]
            inputs_ul_weak, inputs_ul_strong = inputs_ul

            # interleave not required for single GPU training
            # inputs, l_idx = utils.interleave(torch.cat((inputs_l, inputs_ul)))
            # inputs = torch.cat((inputs_l, inputs_ul_weak, inputs_ul_strong))
            # inputs = inputs.cuda()

            inputs = torch.cat((inputs_l, inputs_ul_weak, inputs_ul_strong))

            inputs = inputs.cuda()
            targets_l = targets_l.cuda()
            targets_ul = targets_ul.cuda()
            embedding, logits = model(inputs)

            # split the logits back into labeled and unlabeled
            logits_l = logits[:inputs_l.shape[0]]
            logits_ul = logits[inputs_l.shape[0]:]
            logits_ul_weak = logits_ul[:inputs_ul_weak.shape[0]]
            logits_ul_strong = logits_ul[inputs_ul_weak.shape[0]:]

            embedding_l = embedding[:inputs_l.shape[0]]
            embedding_ul = embedding[inputs_l.shape[0]:]
            embedding_ul_weak = embedding_ul[:inputs_ul_weak.shape[0]]
            embedding_ul_strong = embedding_ul[inputs_ul_weak.shape[0]:]

            cluster_dist_l = emb_constraint(embedding_l, model.last_layer.centers, logits_l)

            max_l_score = torch.max(torch.nn.functional.softmax(logits_l, dim=-1))
            max_l_score = max_l_score.item()

            softmax_ul_weak = torch.nn.functional.softmax(logits_ul_weak, dim=-1)

            if self.args.tau < 1.0:
                if self.args.tau_method == 'fixmatch':
                    # sharpen the logits with tau
                    softmax_ul_weak = softmax_ul_weak / self.args.tau
                elif self.args.tau_method == 'mixmatch':
                    # sharpen the logits with tau, but in a manner which preserves sum to 1
                    softmax_ul_weak = sharpen_mixmatch(x=softmax_ul_weak, T=self.args.tau)
                else:
                    raise RuntimeError("invalid tau method = {}".format(self.args.tau_method))

            sw, _ = torch.max(softmax_ul_weak, dim=-1)
            max_pl_score, _ = torch.max(sw, dim=0)
            max_pl_score = max_pl_score.item()

            score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
            targets_weak_ul = pred_weak

            valid_pl = score_weak >= torch.tensor(self.args.pseudo_label_threshold)

            # capture the number of PL for this batch
            pl_count = torch.sum(valid_pl).item()
            train_stats.append_accumulate('train_pseudo_label_count', pl_count)

            if pl_count > 0:
                # capture the confusion matrix of the PL
                preds = pred_weak[valid_pl]
                tgts = targets_ul[valid_pl]
                acc_vec = preds == tgts
                acc = torch.mean(acc_vec.detach().cpu().type(torch.FloatTensor))
                train_stats.append_accumulate('train_pseudo_label_accuracy', acc.item())
                for c in range(self.args.num_classes):
                    pl_acc_per_class[c].extend(acc_vec[tgts == c].detach().cpu().tolist())
                    pl_count_per_class[c] += torch.sum(preds == c).item()
                    pl_gt_count_per_class[c] += torch.sum(tgts == c).item()

            loss_logits = criterion(logits_l, targets_l)
            cluster_loss = cluster_criterion(cluster_dist_l, torch.zeros_like(cluster_dist_l))
            loss_l = loss_logits + cluster_loss

            # keep just those labels which are valid PL
            logits_ul_strong = logits_ul_strong[valid_pl]
            logits_ul_weak = logits_ul_weak[valid_pl]
            targets_weak_ul = targets_weak_ul[valid_pl]
            embedding_ul_strong = embedding_ul_strong[valid_pl]
            embedding_ul_weak = embedding_ul_weak[valid_pl]

            if pl_count > 0:
                loss_ul = criterion(logits_ul_strong, targets_weak_ul)
                train_stats.append_accumulate('train_pseudo_label_loss', loss_ul.item())


                cluster_dist_ul_strong = emb_constraint(embedding_ul_strong, model.last_layer.centers, logits_ul_weak)
                cluster_dist_ul_weak = emb_constraint(embedding_ul_weak, model.last_layer.centers, logits_ul_weak)

                cluster_loss_ul_strong = cluster_criterion(cluster_dist_ul_strong, torch.zeros_like(cluster_dist_ul_strong))
                cluster_loss_ul_weak = cluster_criterion(cluster_dist_ul_weak, torch.zeros_like(cluster_dist_ul_weak))
                cluster_loss_ul = cluster_loss_ul_strong + cluster_loss_ul_weak
                train_stats.append_accumulate('train_pseudo_label_cluster_loss', cluster_loss_ul.item())
            else:
                loss_ul = torch.tensor(torch.nan).to(loss_l.device)
                cluster_loss_ul = torch.tensor(torch.nan).to(loss_l.device)
            batch_loss = loss_l
            if not torch.isnan(loss_ul):
                batch_loss += loss_ul
            if not torch.isnan(cluster_loss_ul):
                batch_loss += cluster_loss_ul

            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 50)
            optimizer.step()
            if cyclic_lr_scheduler is not None:
                cyclic_lr_scheduler.step()

            if ema_model is not None:
                ema_model.update(model)

            # nan loss values are ignored when using AMP, so ignore them for the average
            if not torch.isnan(batch_loss):
                accuracy = torch.mean((torch.argmax(logits_l, dim=-1) == targets_l).type(torch.float))
                train_stats.append_accumulate('train_loss', batch_loss.item())
                train_stats.append_accumulate('train_accuracy', accuracy.item())
            else:
                loss_nan_count += 1

                if loss_nan_count > int(0.5 * batch_count):
                    raise RuntimeError("Loss is consistently nan (>50% of epoch), terminating train.")

            if batch_idx % 100 == 0:
                # log loss and current GPU utilization
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))
                logging.info('    max resp score: {:4.4g}'.format(max_l_score))
                logging.info('    max resp PL score: {:4.4g}'.format(max_pl_score))

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.add(epoch, 'learning_rate', optimizer.param_groups[0]['lr'])
        train_stats.add(epoch, 'train_wall_time', time.time() - start_time)

        train_stats.close_accumulate(epoch, 'train_pseudo_label_count', method='sum', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_pseudo_label_accuracy', method='avg', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_pseudo_label_loss', method='avg', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_pseudo_label_cluster_loss', method='avg', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_loss', method='avg')
        train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')

        for c in range(len(pl_acc_per_class)):
            pl_acc_per_class[c] = float(np.mean(pl_acc_per_class[c]))
            pl_count_per_class[c] = int(np.sum(pl_count_per_class[c]))
            pl_gt_count_per_class[c] = int(np.sum(pl_gt_count_per_class[c]))

        # get the average accuracy of the pseudo-labels (this data is not available in real SSL applications, since the unlabeled population would truly be unlabeled
        train_stats.add(epoch, 'pseudo_label_counts_per_class', pl_count_per_class)
        train_stats.add(epoch, 'pseudo_label_gt_counts_per_class', pl_gt_count_per_class)

        # update the training metadata
        train_stats.add(epoch, 'pseudo_label_accuracy_per_class', pl_acc_per_class)



    def eval_model(self, model, pytorch_dataset, criterion, train_stats, split_name, epoch, args):
        if pytorch_dataset is None or len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = len(dataloader)
        model.eval()
        start_time = time.time()

        cluster_criterion = torch.nn.MSELoss()

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                embedding, logits = model(inputs)
                cluster_dist = embedding_constraints.l2_cluster_centroid(embedding, model.last_layer.centers, logits)

                cluster_loss = cluster_criterion(cluster_dist, torch.zeros_like(cluster_dist))
                loss_logits = criterion(logits, labels)
                loss = loss_logits + cluster_loss

                train_stats.append_accumulate('{}_loss'.format(split_name), loss.item())
                train_stats.append_accumulate('{}_cluster_loss'.format(split_name), cluster_loss.item())
                train_stats.append_accumulate('{}_logit_loss'.format(split_name), loss_logits.item())

                acc = torch.argmax(logits, dim=-1) == labels
                train_stats.append_accumulate('{}_accuracy'.format(split_name), torch.mean(acc, dtype=torch.float32).item())

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_cluster_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_logit_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_accuracy'.format(split_name), method='avg')

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
