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

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, nb_reps=1, unlabeled_dataset=None):

        if unlabeled_dataset is None:
            raise RuntimeError("Unlabeled dataset missing. Cannot use FixMatch train_epoch function without an unlabeled_dataset.")

        model.train()
        loss_nan_count = 0
        scaler = torch.cuda.amp.GradScaler()
        start_time = time.time()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = nb_reps * len(dataloader)

        # cyclic learning rate with one up/down cycle per epoch.
        if self.args.cycle_factor is None or self.args.cycle_factor == 0:
            cyclic_lr_scheduler = None
        else:
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(self.args.learning_rate / self.args.cycle_factor), max_lr=(self.args.learning_rate * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        unlabeled_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH)
        dataloader_ul = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=self.args.mu*self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)
        iter_ul = iter(dataloader_ul)

        loss_list = list()
        accuracy_list = list()
        pl_loss_list = list()
        pl_accuracy_list = list()
        pl_count_list = list()
        pl_counts_per_class = list()
        gt_counts_per_class = list()
        tp_counter_per_class = list()
        pl_accuracy_per_class = list()
        for i in range(self.args.num_classes):
            pl_counts_per_class.append(0)
            gt_counts_per_class.append(0)
            tp_counter_per_class.append(0)
            pl_accuracy_per_class.append(0)

        cluster_criterion = torch.nn.MSELoss()

        for rep_count in range(nb_reps):
            for batch_idx, tensor_dict_l in enumerate(dataloader):
                optimizer.zero_grad()
                # adjust for the rep offset
                batch_idx = rep_count * len(dataloader) + batch_idx

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

                inputs_l = inputs_l.cuda()
                inputs_ul_weak = inputs_ul_weak.cuda()
                inputs_ul_strong = inputs_ul_strong.cuda()

                targets_l = targets_l.cuda()
                targets_ul = targets_ul.cuda()

                # if self.args.amp:
                #     with torch.cuda.amp.autocast():
                #         resp_gmm, resp_cmm, cluster_dist = model(inputs)
                # else:
                resp_gmm_l, resp_cmm_l, cluster_dist_l = model(inputs_l)
                logits_l = resp_cmm_l

                resp_gmm_ul_weak, resp_cmm_ul_weak, _ = model(inputs_ul_weak)
                logits_ul_weak = resp_cmm_ul_weak

                resp_gmm_ul_strong, resp_cmm_ul_strong, _ = model(inputs_ul_strong)
                logits_ul_strong = resp_cmm_ul_strong

                # split the logits back into labeled and unlabeled
                # logits_l = resp_cmm[:inputs_l.shape[0]]
                cluster_loss = cluster_criterion(cluster_dist_l, torch.zeros_like(cluster_dist_l))

                # logits_ul = resp_cmm[inputs_l.shape[0]:]
                # logits_ul_weak = logits_ul[:inputs_ul_weak.shape[0]]
                # logits_ul_strong = logits_ul[inputs_ul_weak.shape[0]:]

                softmax_ul_weak = torch.nn.functional.softmax(logits_ul_weak, dim=-1)

                # sharpen the logits with tau, but in a manner which preserves sum to 1
                if self.args.tau < 1.0:
                    softmax_ul_weak = sharpen_mixmatch(x=softmax_ul_weak, T=self.args.tau)

                if self.args.soft_labels:
                    # convert hard labels in the fully labeled dataset into soft labels (i.e. one hot)
                    targets_l = torch.nn.functional.one_hot(targets_l, num_classes=self.args.num_classes).type(torch.float)
                    targets_l = targets_l.cuda()

                    score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
                    targets_weak_ul = softmax_ul_weak
                else:
                    score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
                    targets_weak_ul = pred_weak

                valid_pl = score_weak >= torch.tensor(self.args.pseudo_label_threshold)

                # capture the number of PL for this batch
                pl_count = torch.sum(valid_pl).item()
                pl_count_list.append(pl_count)

                if pl_count > 0:
                    # capture the accuracy of the PL
                    acc_vec = torch.logical_and(valid_pl, targets_ul == pred_weak)
                    acc = torch.mean(acc_vec.type(torch.float)).item()
                    pl_accuracy_list.append(acc)
                    for c in range(self.args.num_classes):
                        idx = torch.logical_and(valid_pl, pred_weak == c)
                        val = torch.sum(idx).item()
                        pl_counts_per_class[c] += val

                        idx = torch.logical_and(valid_pl, targets_ul == c)
                        val = torch.sum(idx).item()
                        gt_counts_per_class[c] += val

                        idx = torch.logical_and(targets_ul == c, pred_weak == c)
                        idx = torch.logical_and(valid_pl, idx)
                        val = torch.sum(idx).item()
                        tp_counter_per_class[c] += val

                        idx = torch.logical_and(acc_vec, targets_ul == c)
                        val = torch.sum(idx).item()
                        pl_accuracy_per_class[c] += val

                loss_l = criterion(logits_l, targets_l) + cluster_loss

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
                    pl_loss_list.append(loss_ul.item())
                else:
                    loss_ul = torch.tensor(torch.nan).to(loss_l.device)
                if not torch.isnan(loss_ul):
                    batch_loss = loss_l + loss_ul
                else:
                    batch_loss = loss_l

                if self.args.amp:
                    scaler.scale(batch_loss).backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 50)
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()
                else:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 50)
                    optimizer.step()
                if cyclic_lr_scheduler is not None:
                    cyclic_lr_scheduler.step()

                # nan loss values are ignored when using AMP, so ignore them for the average
                if not torch.isnan(batch_loss):
                    if self.args.soft_labels:
                        targets_l = torch.argmax(targets_l, dim=-1)
                    accuracy = torch.mean((torch.argmax(logits_l, dim=-1) == targets_l).type(torch.float))
                    loss_list.append(batch_loss.item())
                    accuracy_list.append(accuracy.item())
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

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.add(epoch, 'learning_rate', optimizer.param_groups[0]['lr'])
        train_stats.add(epoch, 'train_wall_time', time.time() - start_time)
        train_stats.add(epoch, 'train_loss', np.mean(loss_list))
        train_stats.add(epoch, 'train_accuracy', np.mean(accuracy_list))

        train_stats.add(epoch, 'train_pseudo_label_loss', np.mean(pl_loss_list))

        pl_accuracy = np.mean(pl_accuracy_list)
        if np.isnan(pl_accuracy):
            pl_accuracy = 0.0
        pl_accuracy_per_class = np.asarray(tp_counter_per_class) / np.asarray(gt_counts_per_class)
        pl_accuracy_per_class[np.isnan(pl_accuracy_per_class)] = 0.0
        pl_accuracy_per_class = pl_accuracy_per_class.tolist()



        # get the average accuracy of the pseudo-labels (this data is not available in real SSL applications, since the unlabeled population would truly be unlabeled
        train_stats.add(epoch, 'pseudo_label_accuracy', float(pl_accuracy))
        train_stats.add(epoch, 'num_pseudo_labels', int(np.sum(pl_count_list)))

        # update the training metadata
        train_stats.add(epoch, 'pseudo_label_counts_per_class', pl_counts_per_class)
        train_stats.add(epoch, 'pseudo_label_accuracy_per_class', pl_accuracy_per_class)

        vals = np.asarray(pl_counts_per_class) / np.sum(pl_counts_per_class)
        train_stats.add(epoch, 'pseudo_label_percentage_per_class', vals.tolist())
        vals = np.asarray(tp_counter_per_class) / np.sum(tp_counter_per_class)
        train_stats.add(epoch, 'pseudo_label_gt_percentage_per_class', vals.tolist())



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
        loss_cluster_list = list()
        loss_gmm_list = list()
        loss_cmm_list = list()
        accuracy_gmm_list = list()
        accuracy_cmm_list = list()

        cluster_criterion = torch.nn.MSELoss()

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                labels_cpu = labels.detach().cpu().numpy()
                gt_labels.extend(labels_cpu)

                resp_gmm, resp_cmm, cluster_dist = model(inputs)

                cluster_loss = cluster_criterion(cluster_dist, torch.zeros_like(cluster_dist))
                loss_cluster_list.append(cluster_loss.item())

                batch_loss_gmm = criterion(resp_gmm, labels)
                batch_loss_cmm = criterion(resp_cmm, labels)

                loss = batch_loss_cmm + cluster_loss

                # loss = (batch_loss_gmm + batch_loss_cmm) / 2.0
                # loss = criterion(output, target)
                loss_list.append(loss.item())
                loss_gmm_list.append(batch_loss_gmm.item())
                loss_cmm_list.append(batch_loss_cmm.item())

                # acc = torch.argmax(output, dim=-1) == target
                acc_gmm = torch.argmax(resp_gmm, dim=-1) == labels
                acc_cmm = torch.argmax(resp_cmm, dim=-1) == labels
                accuracy_gmm_list.append(torch.mean(acc_gmm, dtype=torch.float32).item())
                accuracy_cmm_list.append(torch.mean(acc_cmm, dtype=torch.float32).item())

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))


        test_loss = np.nanmean(loss_list)
        train_stats.add(epoch, '{}_loss'.format(split_name), test_loss)
        logging.info('Test set: Average loss: {:.4f}'.format(test_loss))

        test_loss = np.nanmean(loss_gmm_list)
        test_acc = np.nanmean(accuracy_gmm_list)
        train_stats.add(epoch, '{}_gmm_loss'.format(split_name), test_loss)
        train_stats.add(epoch, '{}_gmm_accuracy'.format(split_name), test_acc)
        logging.info('Test set (GMM): Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))

        test_loss = np.nanmean(loss_cmm_list)
        test_acc = np.nanmean(accuracy_cmm_list)
        train_stats.add(epoch, '{}_cmm_loss'.format(split_name), test_loss)
        # train_stats.add(epoch, '{}_cmm_accuracy'.format(split_name), test_acc)
        train_stats.add(epoch, '{}_accuracy'.format(split_name), test_acc)
        logging.info('Test set (CMM): Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))

        test_loss = np.nanmean(loss_cluster_list)
        train_stats.add(epoch, '{}_cluster_loss'.format(split_name), test_loss)
        logging.info('Test set (GMM): Cluster loss: {:.4f}'.format(test_loss))

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)