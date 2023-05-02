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
            resp_gmm, resp_cmm, cluster_dist = model(inputs)

            # logits_l = logits[:inputs_l.shape[0]]
            # logits_ul = logits[inputs_l.shape[0]:]
            # logits_ul_weak = logits_ul[:inputs_ul_weak.shape[0]]
            # logits_ul_strong = logits_ul[inputs_ul_weak.shape[0]:]

            # split the logits back into labeled and unlabeled
            resp_gmm_l = resp_gmm[:inputs_l.shape[0]]
            resp_cmm_l = resp_cmm[:inputs_l.shape[0]]
            cluster_dist_l = cluster_dist[:inputs_l.shape[0]]

            resp_gmm_ul = resp_gmm[inputs_l.shape[0]:]
            resp_cmm_ul = resp_cmm[inputs_l.shape[0]:]

            resp_gmm_ul_weak = resp_gmm_ul[:inputs_ul_weak.shape[0]]
            resp_cmm_ul_weak = resp_cmm_ul[:inputs_ul_weak.shape[0]]

            resp_gmm_ul_strong = resp_gmm_ul[inputs_ul_weak.shape[0]:]
            resp_cmm_ul_strong = resp_cmm_ul[inputs_ul_weak.shape[0]:]

            max_l_cmm_score = torch.max(torch.nn.functional.softmax(resp_cmm_l, dim=-1))
            max_l_cmm_score = max_l_cmm_score.item()
            max_l_gmm_score = torch.max(torch.nn.functional.softmax(resp_gmm_l, dim=-1))
            max_l_gmm_score = max_l_gmm_score.item()

            if self.args.val_acc_term == 'gmm':
                logits_ul_weak = resp_gmm_ul_weak
                logits_ul_strong = resp_gmm_ul_strong
            elif self.args.val_acc_term == 'cmm':
                logits_ul_weak = resp_cmm_ul_weak
                logits_ul_strong = resp_cmm_ul_strong
            elif self.args.val_acc_term == 'gmmcmm':
                a = torch.nn.functional.softmax(resp_gmm_ul_weak, dim=-1)
                b = torch.nn.functional.softmax(resp_cmm_ul_weak, dim=-1)
                logits_ul_weak = torch.log((a + b) / 2.0)

                a = torch.nn.functional.softmax(resp_gmm_ul_strong, dim=-1)
                b = torch.nn.functional.softmax(resp_cmm_ul_strong, dim=-1)
                logits_ul_strong = torch.log((a + b) / 2.0)
            else:
                raise RuntimeError("Invalid val_acc_term = {}".format(self.args.val_acc_term))

            softmax_ul_weak = torch.nn.functional.softmax(logits_ul_weak, dim=-1)
            softmax_gmm_ul_weak = torch.nn.functional.softmax(resp_gmm_ul_weak, dim=-1)
            softmax_cmm_ul_weak = torch.nn.functional.softmax(resp_cmm_ul_weak, dim=-1)

            if self.args.tau < 1.0:
                if self.args.tau_method == 'fixmatch':
                    # sharpen the logits with tau
                    softmax_ul_weak = softmax_ul_weak / self.args.tau
                    softmax_gmm_ul_weak = softmax_gmm_ul_weak / self.args.tau
                    softmax_cmm_ul_weak = softmax_cmm_ul_weak / self.args.tau
                elif self.args.tau_method == 'mixmatch':
                    # sharpen the logits with tau, but in a manner which preserves sum to 1
                    softmax_ul_weak = sharpen_mixmatch(x=softmax_ul_weak, T=self.args.tau)
                    softmax_gmm_ul_weak = sharpen_mixmatch(x=softmax_gmm_ul_weak, T=self.args.tau)
                    softmax_cmm_ul_weak = sharpen_mixmatch(x=softmax_cmm_ul_weak, T=self.args.tau)
                else:
                    raise RuntimeError("invalid tau method = {}".format(self.args.tau_method))

            sw, _ = torch.max(softmax_gmm_ul_weak, dim=-1)
            max_gmm_pl_score, _ = torch.max(sw, dim=0)
            max_gmm_pl_score = max_gmm_pl_score.item()

            sw, _ = torch.max(softmax_cmm_ul_weak, dim=-1)
            max_cmm_pl_score, _ = torch.max(sw, dim=0)
            max_cmm_pl_score = max_cmm_pl_score.item()

            if self.args.soft_labels:
                # convert hard labels in the fully labeled dataset into soft labels (i.e. one hot)
                targets_l = torch.nn.functional.one_hot(targets_l, num_classes=self.args.num_classes).type(torch.float)
                targets_l = targets_l.cuda()

                score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
                targets_weak_ul = softmax_ul_weak
            else:
                score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
                targets_weak_ul = pred_weak

            # pseudo_label_threshold = torch.quantile(score_weak.detach(), 0.99)
            # pseudo_label_threshold = torch.clip(pseudo_label_threshold, 0.5, 0.9)
            # valid_pl = score_weak >= pseudo_label_threshold
            valid_pl = score_weak >= torch.tensor(self.args.pseudo_label_threshold)

            # capture the number of PL for this batch
            pl_count = torch.sum(valid_pl).item()

            # if pl_count == 0:
            #     # grab the best one if there are zero PL
            #     val, idx = torch.max(score_weak, dim=0)
            #     valid_pl = score_weak >= val
            #     pl_count = torch.sum(valid_pl).item()

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

            loss_l = None
            if 'gmm' in self.args.val_acc_term:
                batch_loss_gmm = criterion(resp_gmm_l, targets_l)
                if loss_l is None:
                    loss_l = batch_loss_gmm
                else:
                    loss_l += batch_loss_gmm
            if 'cmm' in self.args.val_acc_term:
                batch_loss_cmm = criterion(resp_cmm_l, targets_l)
                if loss_l is None:
                    loss_l = batch_loss_cmm
                else:
                    loss_l += batch_loss_cmm

            cluster_loss = cluster_criterion(cluster_dist_l, torch.zeros_like(cluster_dist_l))
            if loss_l is None:
                loss_l = cluster_loss
            else:
                loss_l += cluster_loss

            if self.args.soft_labels:
                # remove the invalid elements before the loss is calculated
                logits_ul_strong = logits_ul_strong[valid_pl, :]
                targets_weak_ul = targets_weak_ul[valid_pl, :]
            else:
                # use CE = -100 to invalidate certain labels
                # targets_weak_ul[torch.logical_not(valid_pl)] = -100
                logits_ul_strong = logits_ul_strong[valid_pl]
                targets_weak_ul = targets_weak_ul[valid_pl]

            if pl_count > 0:
                loss_ul = criterion(logits_ul_strong, targets_weak_ul)
                train_stats.append_accumulate('train_pseudo_label_loss', loss_ul.item())
            else:
                loss_ul = torch.tensor(torch.nan).to(loss_l.device)
            if not torch.isnan(loss_ul):
                batch_loss = loss_l + loss_ul
            else:
                batch_loss = loss_l

            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 50)
            optimizer.step()
            if cyclic_lr_scheduler is not None:
                cyclic_lr_scheduler.step()

            if ema_model is not None:
                ema_model.update(model)

            # nan loss values are ignored when using AMP, so ignore them for the average
            if not torch.isnan(batch_loss):
                if self.args.soft_labels:
                    targets_l = torch.argmax(targets_l, dim=-1)
                accuracy_gmm = torch.mean((torch.argmax(resp_gmm_l, dim=-1) == targets_l).type(torch.float))
                accuracy_cmm = torch.mean((torch.argmax(resp_cmm_l, dim=-1) == targets_l).type(torch.float))
                train_stats.append_accumulate('train_loss', batch_loss.item())
                train_stats.append_accumulate('train_gmm_accuracy', accuracy_gmm.item())
                train_stats.append_accumulate('train_cmm_accuracy', accuracy_cmm.item())
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
                logging.info('    max GMM score: {:4.4g}  max CMM score: {:4.4g}'.format(max_l_gmm_score, max_l_cmm_score))
                logging.info('    max GMM PL score: {:4.4g}  max CMM PL score: {:4.4g}'.format(max_gmm_pl_score, max_cmm_pl_score))

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.add(epoch, 'learning_rate', optimizer.param_groups[0]['lr'])
        train_stats.add(epoch, 'train_wall_time', time.time() - start_time)

        train_stats.close_accumulate(epoch, 'train_pseudo_label_count', method='sum', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_pseudo_label_accuracy', method='avg', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_pseudo_label_loss', method='avg', default_value=0.0)  # default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_loss', method='avg')
        train_stats.close_accumulate(epoch, 'train_gmm_accuracy', method='avg')
        train_stats.close_accumulate(epoch, 'train_cmm_accuracy', method='avg')

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

                resp_gmm, resp_cmm, cluster_dist = model(inputs)

                cluster_loss = cluster_criterion(cluster_dist, torch.zeros_like(cluster_dist))
                batch_loss_gmm = criterion(resp_gmm, labels)
                batch_loss_cmm = criterion(resp_cmm, labels)

                loss = None
                if 'gmm' in self.args.val_acc_term:
                    if loss is None:
                        loss = batch_loss_gmm
                    else:
                        loss += batch_loss_gmm
                if 'cmm' in self.args.val_acc_term:
                    if loss is None:
                        loss = batch_loss_cmm
                    else:
                        loss += batch_loss_cmm
                if loss is None:
                    loss = cluster_loss
                else:
                    loss += cluster_loss

                # loss = (batch_loss_gmm + batch_loss_cmm) / 2.0
                # loss = criterion(output, target)
                train_stats.append_accumulate('{}_loss'.format(split_name), loss.item())
                train_stats.append_accumulate('{}_cluster_loss'.format(split_name), cluster_loss.item())
                train_stats.append_accumulate('{}_gmm_loss'.format(split_name), batch_loss_gmm.item())
                train_stats.append_accumulate('{}_cmm_loss'.format(split_name), batch_loss_cmm.item())

                # acc = torch.argmax(output, dim=-1) == target
                acc_gmm = torch.argmax(resp_gmm, dim=-1) == labels
                acc_cmm = torch.argmax(resp_cmm, dim=-1) == labels
                resp_gmm_sm = resp_gmm.softmax(dim=-1)
                resp_cmm_sm = resp_cmm.softmax(dim=-1)
                gmm_cmm_sm = (resp_gmm_sm + resp_cmm_sm) / 2.0
                acc_gmm_cmm = torch.argmax(gmm_cmm_sm, dim=-1) == labels
                train_stats.append_accumulate('{}_gmm_accuracy'.format(split_name), torch.mean(acc_gmm, dtype=torch.float32).item())
                train_stats.append_accumulate('{}_cmm_accuracy'.format(split_name), torch.mean(acc_cmm, dtype=torch.float32).item())
                train_stats.append_accumulate('{}_gmmcmm_accuracy'.format(split_name), torch.mean(acc_gmm_cmm, dtype=torch.float32).item())

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_cluster_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_gmm_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_cmm_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_gmm_accuracy'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_cmm_accuracy'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_gmmcmm_accuracy'.format(split_name), method='avg')

        if args.val_acc_term == 'gmm':
            # use just gmm for val acc
            acc = train_stats.get_epoch('{}_gmm_accuracy'.format(split_name), epoch)
            train_stats.add(epoch, '{}_accuracy'.format(split_name), acc)
        elif args.val_acc_term == 'cmm':
            # use just cmm for val acc
            acc = train_stats.get_epoch('{}_cmm_accuracy'.format(split_name), epoch)
            train_stats.add(epoch, '{}_accuracy'.format(split_name), acc)
        elif args.val_acc_term == 'gmmcmm':
            # use just average of gmm and cmm for val acc
            acc = train_stats.get_epoch('{}_gmmcmm_accuracy'.format(split_name), epoch)
            train_stats.add(epoch, '{}_accuracy'.format(split_name), acc)
        else:
            raise RuntimeError("Unknown args.val_acc_term={} codebase doesn't know which accuracy to return as val_accuracy for early stopping".format(args.val_acc_term))

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
