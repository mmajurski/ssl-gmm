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


class FixMatchTrainer(trainer.SupervisedTrainer):

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, nb_reps=1, unlabeled_dataset=None, ema_model=None):

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
            epoch_init_lr = optimizer.param_groups[0]['lr']
            train_stats.add(epoch, 'learning_rate', epoch_init_lr)
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(epoch_init_lr / self.args.cycle_factor), max_lr=(epoch_init_lr * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        unlabeled_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH)
        dataloader_ul = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=self.args.mu*self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)
        iter_ul = iter(dataloader_ul)

        pl_acc_per_class = list()
        pl_count_per_class = list()
        pl_gt_count_per_class = list()
        for i in range(self.args.num_classes):
            pl_acc_per_class.append(list())
            pl_count_per_class.append(0)
            pl_gt_count_per_class.append(0)

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
            inputs = torch.cat((inputs_l, inputs_ul_weak, inputs_ul_strong))
            inputs = inputs.cuda()
            targets_l = targets_l.cuda()
            targets_ul = targets_ul.cuda()

            logits = model(inputs)

            # split the logits back into labeled and unlabeled
            logits_l = logits[:inputs_l.shape[0]]
            logits_ul = logits[inputs_l.shape[0]:]
            logits_ul_weak = logits_ul[:inputs_ul_weak.shape[0]]
            logits_ul_strong = logits_ul[inputs_ul_weak.shape[0]:]

            softmax_ul_weak = torch.nn.functional.softmax(logits_ul_weak, dim=-1)
            # sharpen the logits with tau, but in a manner which preserves sum to 1
            if self.args.tau < 1.0:
                if self.args.tau_method == 'fixmatch':
                    softmax_ul_weak = softmax_ul_weak / self.args.tau
                elif self.args.tau_method == 'mixmatch':
                    softmax_ul_weak = sharpen_mixmatch(x=softmax_ul_weak, T=self.args.tau)
                else:
                    raise RuntimeError("invalid tau method = {}".format(self.args.tau_method))

            score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
            targets_weak_ul = pred_weak

            valid_pl = score_weak >= torch.tensor(self.args.pseudo_label_threshold)

            # capture the number of PL for this batch
            pl_count = torch.sum(valid_pl).item()
            train_stats.append_accumulate('train_pseudo_label_count', pl_count)

            if pl_count > 0:
                # capture the accuracy of the PL
                # capture the confusion matrix of the PL
                preds = pred_weak[valid_pl]
                tgts = targets_ul[valid_pl]
                acc_vec = preds == tgts
                for a in acc_vec.detach().cpu().tolist():
                    train_stats.append_accumulate('train_pseudo_label_accuracy', float(a))

                for c in range(self.args.num_classes):
                    pl_acc_per_class[c].extend(acc_vec[tgts == c].detach().cpu().tolist())
                    pl_count_per_class[c] += torch.sum(preds == c).item()
                    pl_gt_count_per_class[c] += torch.sum(tgts == c).item()

            loss_l = criterion(logits_l, targets_l)

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

            # nan loss values are ignored when using AMP, so ignore them for the average
            if not torch.isnan(batch_loss):
                accuracy = torch.mean((torch.argmax(logits_l, dim=-1) == targets_l).type(torch.float))
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

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.close_accumulate(epoch, 'train_pseudo_label_count', method='sum')
        train_stats.close_accumulate(epoch, 'train_pseudo_label_accuracy', method='avg', default_value=0.0)  #default value in case no data was collected
        train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')
        train_stats.close_accumulate(epoch, 'train_pseudo_label_loss', method='avg')
        train_stats.close_accumulate(epoch, 'train_loss', method='avg')

        train_stats.add(epoch, 'train_wall_time', time.time() - start_time)

        try:
            pl_accuracy = train_stats.get_epoch('train_pseudo_label_accuracy', epoch)
        except:
            pl_accuracy = np.nan
        if np.isnan(pl_accuracy):
            train_stats.set_epoch('train_pseudo_label_accuracy', epoch, 0.0)

        for c in range(len(pl_acc_per_class)):
            pl_acc_per_class[c] = float(np.mean(pl_acc_per_class[c]))
            pl_count_per_class[c] = int(np.sum(pl_count_per_class[c]))
            pl_gt_count_per_class[c] = int(np.sum(pl_gt_count_per_class[c]))

        train_stats.add(epoch, 'pseudo_label_counts_per_class', pl_count_per_class)
        train_stats.add(epoch, 'pseudo_label_gt_counts_per_class', pl_gt_count_per_class)

        # update the training metadata
        train_stats.add(epoch, 'pseudo_label_accuracy_per_class', pl_acc_per_class)

        if cyclic_lr_scheduler is not None:
            # reset any leftover changes to the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = epoch_init_lr