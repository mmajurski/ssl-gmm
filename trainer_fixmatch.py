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


class FixMatchTrainer(trainer.SupervisedTrainer):

    def train_epoch(self, model, pytorch_dataset, optimizer, criterion, emb_constraint, epoch, train_stats, unlabeled_dataset=None, ema_model=None):

        if unlabeled_dataset is None:
            raise RuntimeError("Unlabeled dataset missing. Cannot use FixMatch train_epoch function without an unlabeled_dataset.")

        model.train()
        loss_nan_count = 0
        start_time = time.time()

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)

        batch_count = len(dataloader)

        cyclic_lr_scheduler = None
        # cyclic learning rate with one up/down cycle per epoch.
        if self.args.cycle_factor is not None and self.args.cycle_factor > 0:
            epoch_init_lr = optimizer.param_groups[0]['lr']
            train_stats.add(epoch, 'learning_rate', epoch_init_lr)
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(epoch_init_lr / self.args.cycle_factor), max_lr=(epoch_init_lr * self.args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

        if self.args.dataset.upper() == 'CIFAR10':
            unlabeled_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH)
        elif self.args.dataset.upper() == 'STL10':
            unlabeled_dataset.set_transforms(cifar_datasets.STL10.TRANSFORM_FIXMATCH)

        dataloader_ul = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=self.args.mu*self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)

        if len(dataloader) != len(dataloader_ul):
            raise RuntimeError("Mismatch is dataloader lengths")

        iter_ul = iter(dataloader_ul)

        pl_acc_per_class = list()
        l_acc_per_class = list()
        pl_count_per_class = list()
        pl_gt_count_per_class = list()
        for i in range(self.args.num_classes):
            pl_acc_per_class.append(list())
            l_acc_per_class.append(list())
            pl_count_per_class.append(0)
            pl_gt_count_per_class.append(0)
        nb_high_loss = 0

        embedding_criterion = torch.nn.MSELoss()

        # if hasattr(model, 'last_layer') and hasattr(model.last_layer, 'centers'):
        #     model.last_layer.centers = model.last_layer.centers.cuda()
        # if hasattr(model, 'module') and hasattr(model.module.last_layer, 'centers'):
        #     model.module.last_layer.centers = model.module.last_layer.centers.cuda()

        for batch_idx, tensor_dict_l in enumerate(dataloader):
            inputs_l = tensor_dict_l[0]
            targets_l = tensor_dict_l[1]

            tensor_dict_ul = next(iter_ul)

            inputs_ul = tensor_dict_ul[0]
            targets_ul = tensor_dict_ul[1]
            inputs_ul_weak, inputs_ul_strong = inputs_ul

            inputs = torch.cat((inputs_l, inputs_ul_weak, inputs_ul_strong))
            inputs = inputs.cuda()


            # interleave not required for single GPU training
            inputs = utils.interleave(inputs, 2 * self.args.mu + 1)
            embedding, logits, latent_ce = model(inputs)

            logits = utils.de_interleave(logits, 2 * self.args.mu + 1)
            embedding = utils.de_interleave(embedding, 2 * self.args.mu + 1)

            targets_l = targets_l.cuda()
            targets_ul = targets_ul.cuda()
            # embedding, logits = model(inputs)

            # split the logits back into labeled and unlabeled
            logits_l = logits[:inputs_l.shape[0]]
            logits_ul = logits[inputs_l.shape[0]:]
            logits_ul_weak = logits_ul[:inputs_ul_weak.shape[0]]
            logits_ul_strong = logits_ul[inputs_ul_weak.shape[0]:]

            embedding_l = embedding[:inputs_l.shape[0]]
            embedding_ul = embedding[inputs_l.shape[0]:]
            embedding_ul_weak = embedding_ul[:inputs_ul_weak.shape[0]]
            embedding_ul_strong = embedding_ul[inputs_ul_weak.shape[0]:]


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

            score_weak, pred_weak = torch.max(softmax_ul_weak, dim=-1)
            targets_weak_ul = pred_weak

            valid_pl = score_weak >= torch.tensor(self.args.pseudo_label_threshold)
            if self.args.cosine_sim_pl_threshold > 0:
                a = embedding_l.repeat(embedding_ul_weak.shape[0], 1, 1).permute(1, 0, 2)
                b = embedding_ul_weak.unsqueeze(0)
                cosine_sim_weak = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                cosine_sim_weak, _ = torch.topk(cosine_sim_weak, k=self.args.cosine_sim_topk, dim=0)
                cosine_sim_weak, _ = torch.min(cosine_sim_weak, dim=0)

                valid_pl2 = cosine_sim_weak >= torch.tensor(self.args.cosine_sim_pl_threshold)
                valid_pl = torch.logical_and(valid_pl, valid_pl2)

            # capture the number of PL for this batch
            pl_count = torch.sum(valid_pl).item()
            train_stats.append_accumulate('train_pseudo_label_count', pl_count)
            train_stats.append_accumulate('train_pseudo_label_mask_rate', (pl_count / (self.args.batch_size * self.args.mu)) )

            if pl_count > 0:
                # capture the confusion matrix of the PL
                preds = pred_weak[valid_pl]
                tgts = targets_ul[valid_pl]
                acc_vec = preds == tgts
                acc = torch.mean(acc_vec.detach().cpu().type(torch.FloatTensor))
                train_stats.append_accumulate('train_pseudo_label_accuracy', acc.item())
                train_stats.append_accumulate('train_pseudo_label_impurity', (1.0 - acc.item()))

                for c in range(self.args.num_classes):
                    pl_acc_per_class[c].extend(acc_vec[tgts == c].detach().cpu().tolist())
                    pl_count_per_class[c] += torch.sum(preds == c).item()
                    pl_gt_count_per_class[c] += torch.sum(tgts == c).item()

            loss_l = criterion(logits_l, targets_l)
            score_l, pred_l = torch.max(logits_l, dim=-1)
            acc_vec = pred_l == targets_l
            for c in range(self.args.num_classes):
                l_acc_per_class[c].extend(acc_vec[targets_l == c].detach().cpu().tolist())

            # # handle negative samples (any logit < 0.1) has its CE calculated and added to the loss
            # invalid_pl_logits_mask = softmax_ul_weak < torch.tensor(self.args.pseudo_label_negative_threshold)
            # if torch.any(invalid_pl_logits_mask):
            #     # calculate y and yhat, subsetting to only those logits which are invalid PL
            #     y = torch.nn.functional.one_hot(targets_weak_ul, self.args.num_classes)[invalid_pl_logits_mask]
            #     softmax_ul_strong = torch.nn.functional.softmax(logits_ul_strong, dim=-1)
            #     yhat = softmax_ul_strong[invalid_pl_logits_mask]
            #
            #     # if there are any true class predictions among the invalid PL, return that as impurity.
            #     train_stats.append_accumulate('train_invalid_pl_impurity', torch.mean(y>0, dtype=yhat.dtype).item())
            #
            #     # implement log_loss by hand
            #     minus_one_over_N = (-1.0 / torch.numel(yhat))
            #
            #     log_yhat = torch.log(torch.clamp(yhat, min=0.0001))
            #     log_one_minus_yhat = torch.log(torch.clamp(1.0 - yhat, min=0.0001))
            #     presum = (y * log_yhat + (1.0 - y) * log_one_minus_yhat) * minus_one_over_N
            #     loss_invalid_pl = torch.sum(presum)
            #     # scale the loss to equal the contribution of the valid PL
            #     loss_invalid_pl = loss_invalid_pl * (1.0 / self.args.num_classes)
            #     loss_l += loss_invalid_pl
            #     train_stats.append_accumulate('train_invalid_pl_loss', loss_invalid_pl.item())

            if emb_constraint is not None:
                if hasattr(model, 'module'):
                   emb_constraint_l = emb_constraint(embedding_l, model.module.last_layer.centers, logits_l)
                else:
                   emb_constraint_l = emb_constraint(embedding_l, model.last_layer.centers, logits_l)
                emb_constraint_loss_l = embedding_criterion(emb_constraint_l, torch.zeros_like(emb_constraint_l))

                # emb_constraint_loss_l = 0.01 * latent_ce

                #print('emb_constraint_loss_l', emb_constraint_loss_l.detach().cpu().numpy(), 'loss_l', loss_l)
                train_stats.append_accumulate('train_embedding_constraint_loss', emb_constraint_loss_l.item())
                loss_l += emb_constraint_loss_l

            # keep just those labels which are valid PL
            logits_ul_strong = logits_ul_strong[valid_pl]
            logits_ul_weak = logits_ul_weak[valid_pl]
            targets_weak_ul = targets_weak_ul[valid_pl]
            embedding_ul_strong = embedding_ul_strong[valid_pl]
            embedding_ul_weak = embedding_ul_weak[valid_pl]

            loss_ul = torch.tensor(torch.nan).to(loss_l.device)
            emb_constraint_loss_ul = torch.tensor(torch.nan).to(loss_l.device)

            if pl_count > 0:
                loss_ul = criterion(logits_ul_strong, targets_weak_ul)
                train_stats.append_accumulate('train_pseudo_label_loss', loss_ul.item())

                if emb_constraint is not None:
                    if hasattr(model, 'module'):
                        emb_constraint_ul_strong = emb_constraint(embedding_ul_strong, model.module.last_layer.centers, logits_ul_weak)
                        emb_constraint_ul_weak = emb_constraint(embedding_ul_weak, model.module.last_layer.centers, logits_ul_weak)
                    else:
                        emb_constraint_ul_strong = emb_constraint(embedding_ul_strong, model.last_layer.centers, logits_ul_weak)
                        emb_constraint_ul_weak = emb_constraint(embedding_ul_weak, model.last_layer.centers, logits_ul_weak)

                    emb_constraint_loss_ul_strong = embedding_criterion(emb_constraint_ul_strong, torch.zeros_like(emb_constraint_ul_strong))
                    emb_constraint_loss_ul_weak = embedding_criterion(emb_constraint_ul_weak, torch.zeros_like(emb_constraint_ul_weak))
                    emb_constraint_loss_ul = emb_constraint_loss_ul_strong + emb_constraint_loss_ul_weak
                    train_stats.append_accumulate('train_pseudo_label_embedding_constraint_loss', emb_constraint_loss_ul.item())

            batch_loss = loss_l
            if not torch.isnan(loss_ul):
                batch_loss += loss_ul
            if not torch.isnan(emb_constraint_loss_ul):
                batch_loss += emb_constraint_loss_ul

            batch_loss.backward()
            if batch_loss > 1.0:
                nb_high_loss += 1
                # logging.info("Loss {} is high clipping grad norm 1.0".format(batch_loss))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # if self.args.clip_grad:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #     # torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            optimizer.zero_grad()  # set_to_none=True)
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

            if batch_idx % self.args.log_interval == 0:
                # log loss and current GPU utilization
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        if loss_nan_count > 0:
            logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

        train_stats.add(epoch, 'train_wall_time', time.time() - start_time)

        train_stats.close_accumulate(epoch, 'train_pseudo_label_count', method='sum', default_value=0.0)
        train_stats.close_accumulate(epoch, 'train_pseudo_label_ood_count', method='sum', default_value=0.0)
        # close the rest with avg
        train_stats.close_all_accumulate(epoch, method='avg', default_value=0.0)

        nb_high_loss = nb_high_loss / len(dataloader)
        train_stats.add(epoch, 'train_grad_clip_rate', nb_high_loss)

        for c in range(len(pl_acc_per_class)):
            pl_acc_per_class[c] = float(np.mean(pl_acc_per_class[c]))
            l_acc_per_class[c] = float(np.mean(l_acc_per_class[c]))
            pl_count_per_class[c] = int(np.sum(pl_count_per_class[c]))
            pl_gt_count_per_class[c] = int(np.sum(pl_gt_count_per_class[c]))

        # get the average accuracy of the pseudo-labels (this data is not available in real SSL applications, since the unlabeled population would truly be unlabeled
        train_stats.add(epoch, 'pseudo_label_counts_per_class', pl_count_per_class)
        train_stats.add(epoch, 'pseudo_label_gt_counts_per_class', pl_gt_count_per_class)
        train_stats.add(epoch, 'train_accuracy_per_class', l_acc_per_class)


        # update the training metadata
        train_stats.add(epoch, 'pseudo_label_accuracy_per_class', pl_acc_per_class)

        if cyclic_lr_scheduler is not None:
            # reset any leftover changes to the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = epoch_init_lr


    def eval_model(self, model, pytorch_dataset, criterion, train_stats, split_name, emb_constraint, epoch, args, return_embedding=False):
        if pytorch_dataset is None or len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        batch_count = len(dataloader)
        model.eval()
        start_time = time.time()

        embedding_criterion = torch.nn.MSELoss()
        
        embedding_output_test = list()
        labels_output_test    = list()

        acc_per_class = list()
        for i in range(self.args.num_classes):
            acc_per_class.append(list())

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                inputs = tensor_dict[0].cuda()
                labels = tensor_dict[1].cuda()

                embedding, logits, latent_ce = model(inputs)
                loss = criterion(logits, labels)
                if emb_constraint is not None:
                    # only include a "logit" loss, when there are other terms
                    train_stats.append_accumulate('{}_logit_loss'.format(split_name), loss.item())
                    if hasattr(model, 'module'):
                        emb_constraint_l = emb_constraint(embedding, model.module.last_layer.centers, logits)
                    else:
                        emb_constraint_l = emb_constraint(embedding, model.last_layer.centers, logits)
                    emb_constraint_loss = embedding_criterion(emb_constraint_l, torch.zeros_like(emb_constraint_l))

                    # emb_constraint_loss = latent_ce

                    train_stats.append_accumulate('{}_emb_constraint_loss'.format(split_name), emb_constraint_loss.item())
                    loss += emb_constraint_loss

                train_stats.append_accumulate('{}_loss'.format(split_name), loss.item())

                acc = torch.argmax(logits, dim=-1) == labels
                train_stats.append_accumulate('{}_accuracy'.format(split_name), torch.mean(acc, dtype=torch.float32).item())

                for c in range(self.args.num_classes):
                    acc_per_class[c].extend(acc[labels == c].detach().cpu().tolist())

                if batch_idx % self.args.log_interval == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))
                
                if return_embedding or args.save_embedding:
                    embedding_output_test.append( embedding.detach().cpu().numpy() )
                    labels_output_test.append(  labels.detach().cpu().numpy() )

        train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
        if emb_constraint is not None:
            train_stats.close_accumulate(epoch, '{}_logit_loss'.format(split_name), method='avg')
            train_stats.close_accumulate(epoch, '{}_emb_constraint_loss'.format(split_name), method='avg')
        train_stats.close_accumulate(epoch, '{}_accuracy'.format(split_name), method='avg')

        for c in range(len(acc_per_class)):
            acc_per_class[c] = float(np.mean(acc_per_class[c]))

        train_stats.add(epoch, '{}_accuracy_per_class'.format(split_name), acc_per_class)

        wall_time = time.time() - start_time
        train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)

        if return_embedding:
            # merge embeddings together
            embedding_output_test = utils.multiconcat_numpy(embedding_output_test)
            labels_output_test = utils.multiconcat_numpy(labels_output_test)
            return embedding_output_test, labels_output_test

        if args.save_embedding:
            embedding_output_test = utils.multiconcat_numpy(embedding_output_test)
            outpath = args.output_dirpath + "/test_embedding.npy"
            logging.info("save " + outpath)
            np.save(outpath, embedding_output_test)

            labels_output_test    = utils.multiconcat_numpy(labels_output_test)
            outpath = args.output_dirpath + "/test_labels.npy"
            logging.info("save " + outpath)
            np.save(outpath, labels_output_test)


