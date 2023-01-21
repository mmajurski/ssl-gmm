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
from gmm_module import GMM


class FixMatchTrainer_gmm(trainer.SupervisedTrainer):

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, epoch, train_stats, nb_reps=1, unlabeled_dataset=None):

        if unlabeled_dataset is None:
            raise RuntimeError("Unlabeled dataset missing. Cannot use FixMatch train_epoch function without an unlabeled_dataset.")

        model.train()
        loss_nan_count = 0
        scaler = torch.cuda.amp.GradScaler()
        mu = 5  #7  # the unlabeled data is 7x the size of the labeled
        start_time = time.time()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)

        batch_count = nb_reps * len(dataloader)

        unlabeled_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH)
        dataloader_ul = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=mu*self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, worker_init_fn=utils.worker_init_fn)
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

        for rep_count in range(nb_reps):
            for batch_idx, tensor_dict_l in enumerate(dataloader):
                optimizer.zero_grad()
                # adjust for the rep offset
                batch_idx = rep_count * len(dataloader) + batch_idx

                inputs_l, targets_l, _ = tensor_dict_l

                try:
                    tensor_dict_ul = next(iter_ul)
                except:
                    # recreate the iterator
                    iter_ul = iter(dataloader_ul)
                    tensor_dict_ul = next(iter_ul)

                inputs_ul, targets_ul, _ = tensor_dict_ul
                inputs_ul_weak, inputs_ul_strong = inputs_ul

                # interleave not required for single GPU training
                # inputs, l_idx = utils.interleave(torch.cat((inputs_l, inputs_ul)))
                inputs = torch.cat((inputs_l, inputs_ul_weak, inputs_ul_strong))
                inputs = inputs.cuda()
                targets_l = targets_l.cuda()
                targets_ul = targets_ul.cuda()

                if self.args.amp:
                    with torch.cuda.amp.autocast():
                        logits = model(inputs)
                else:
                    logits = model(inputs)

                # split the logits back into labeled and unlabeled
                logits_l = logits[:inputs_l.shape[0]]
                logits_ul = logits[inputs_l.shape[0]:]
                logits_ul_weak = logits_ul[:inputs_ul_weak.shape[0]]
                logits_ul_strong = logits_ul[inputs_ul_weak.shape[0]:]

                # creating gmm/cmm and bucketing data for training the same

                #TODO: trasnfer this process into a new function
                unique_targets_l = torch.unique(targets_l)
                bucketed_logits_l = list()
                gmm_list = list()
                pred_weak = list()
                score_weak = list()

                for i in range(len(unique_targets_l)):
                    c = unique_targets_l[i]
                    class_c_logits = logits_l[logits_l == c]
                    bucketed_logits_l.append(class_c_logits)

                    gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=self.args.cluster_per_class, tolerance=1e-4, max_iter=50, isCauchy=(self.args.inference_method == 'cauchy'))
                    gmm.fit(class_c_logits)
                    while np.any(np.isnan(gmm.get("sigma").detach().cpu().numpy())):
                        gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=self.args.cluster_per_class, tolerance=1e-4, max_iter=50, isCauchy=(self.args.inference_method == 'cauchy'))
                        gmm.fit(class_c_logits)
                    gmm_list.append(gmm)

                class_preval = utils.compute_class_prevalance(dataloader)
                # logging.info(class_preval)
                class_preval = torch.tensor(list(class_preval.values()))

                # generate weights, mus and sigmas
                weights = class_preval.repeat_interleave(self.args.cluster_per_class)
                pi = torch.cat([torch.FloatTensor(gmm.weights_) for gmm in gmm_list])
                weights *= pi
                mus = torch.cat([torch.FloatTensor(gmm.means_) for gmm in gmm_list])
                sigmas = torch.cat([torch.FloatTensor(gmm.covariances_) for gmm in gmm_list])
                gmm = GMM(n_features=self.args.num_classes, n_clusters=weights.shape[0], weights=weights, mu=mus, sigma=sigmas)

                logits_ul_weak = logits_ul_weak / self.args.tau
                # softmax_ul_weak = torch.softmax(logits_ul_weak, dim=-1)

                gmm_inputs = logits_ul_weak.detach().cpu().numpy()

                # TODO: check if we are supposed to train gmm or not
                if True:
                    gmm_resp = gmm.predict_proba(gmm_inputs)  # N*1, N*K
                    if not isinstance(gmm_resp, np.ndarray):
                        gmm_resp = gmm_resp.detach().cpu().numpy()
                    # TODO: change the value
                    gmm_pred = np.argmax(gmm_resp, axis=-1)
                    gmm_score = np.max(gmm_resp, axis=-1)
                    # gmm_pred = np.argmax(gmm_resp, axis=-1) // args.cluster_per_class
                    pred_weak.extend(gmm_pred)
                    score_weak.extend(gmm_score)

                # TODO: change this condition to check if we are supposed to create gmm or cmm
                if hasattr(gmm, 'predict_cauchy_probability'):
                    cauchy_unnorm_resp, _, cauchy_resp = gmm.predict_cauchy_probability(gmm_inputs)
                    if not isinstance(cauchy_resp, np.ndarray):
                        cauchy_resp = cauchy_resp.detach().cpu().numpy()
                    cauchy_pred = np.argmax(cauchy_resp, axis=-1)
                    cauchy_score = np.max(cauchy_resp, axis=-1)
                    pred_weak.extend(cauchy_pred)
                    score_weak.extend(gmm_score)



                # TODO: discuss if soft_labeling is to be incorporated and make changes accordingly
                # if self.args.soft_labels:
                #     # convert hard labels in the fully labeled dataset into soft labels (i.e. one hot)
                #     targets_l = torch.nn.functional.one_hot(targets_l, num_classes=self.args.num_classes).type(torch.float)
                #     targets_l = targets_l.cuda()
                #
                #     score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
                #     targets_weak_ul = softmax_ul_weak
                # else:
                #     score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
                #     targets_weak_ul = pred_weak

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


                loss_l = criterion(logits_l, targets_l)

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
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimizer.step()

                # nan loss values are ignored when using AMP, so ignore them for the average
                if not np.isnan(batch_loss.detach().cpu().numpy()):
                    if self.args.soft_labels:
                        targets_l = torch.argmax(targets_l, dim=-1)
                    accuracy = torch.mean((torch.argmax(logits_l, dim=-1) == targets_l).type(torch.float))
                    loss_list.append(batch_loss.item())
                    accuracy_list.append(accuracy.item())
                else:
                    loss_nan_count += 1

                    if loss_nan_count > 100:
                        raise RuntimeError("Loss is consistently nan (>100x per epoch), terminating train.")

                if batch_idx % 100 == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

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