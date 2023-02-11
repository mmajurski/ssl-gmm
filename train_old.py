import os
import time
import copy
import numpy as np
import torch
import torchvision
import json
import logging
import sklearn.mixture
import psutil
from matplotlib import pyplot as plt

import cifar_datasets
import metadata
from gmm_module import GMM
from skl_cauchy_mm import CMM
import lr_scheduler
import flavored_resnet18
import flavored_wideresnet
import fixmatch_augmentation
import utils
import trainer



def plot_loss_acc(train_stats, args, best_epoch):
    plt.figure(figsize=(8, 6))
    y = train_stats.get('train_loss')
    if len(y) == 0 or y[0] is None:
        # skip if data is not valid
        return

    plt.plot(y)
    y = train_stats.get('val_loss')
    plt.plot(y)

    y = train_stats.get('train_loss')[best_epoch]
    plt.plot(best_epoch, y, marker='*')
    y = train_stats.get('val_loss')[best_epoch]
    plt.plot(best_epoch, y, marker='*')

    plt.title("Train and Val Loss")
    plt.legend(['train', 'val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(args.output_dirpath, 'loss.png'))
    plt.clf()


    y = train_stats.get('train_accuracy')
    plt.plot(y)
    y = train_stats.get('val_softmax_accuracy')
    plt.plot(y)

    y = train_stats.get('train_accuracy')[best_epoch]
    plt.plot(best_epoch, y, marker='*')
    y = train_stats.get('val_softmax_accuracy')[best_epoch]
    plt.plot(best_epoch, y, marker='*')

    plt.title("Train and Val Softmax Accuracy")
    plt.legend(['train','val'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(args.output_dirpath, 'accuracy.png'))
    plt.clf()


    y = train_stats.get('pseudo_labeling_accuracy')
    if len(y) > 0:
        plt.plot(y)

        plt.title("PL Accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(args.output_dirpath, 'pl_accuracy.png'))
        plt.clf()

    y = train_stats.get('num_pseudo_labels')
    if len(y) > 0:
        plt.plot(y)

        plt.title("PL Count")
        plt.ylabel('Count')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(args.output_dirpath, 'num_pseudo_labels.png'))
        plt.clf()


    plt.close()


def get_optimizer(args, model):
    # Setup optimizer
    if args.weight_decay is not None:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, nesterov=args.nesterov)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise RuntimeError("Invalid optimizer: {}".format(args.optimizer))
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=args.nesterov)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        else:
            raise RuntimeError("Invalid optimizer: {}".format(args.optimizer))
    return optimizer


def setup(args):
    # load stock models from https://pytorch.org/vision/stable/models.html
    model = None
    if args.starting_model is not None:
        # warning, this over rides the args.arch selection
        logging.info("Loading requested starting model from '{}'".format(args.starting_model))
        model = torch.load(args.starting_model)
    else:
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
            #model = flavored_resnets.ResNet18(num_classes=args.num_classes)
        if args.arch == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)
        if args.arch == 'resnext50_32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=args.num_classes)
        if args.arch == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=args.num_classes)
        if args.arch == 'wide_resnet':
            model = flavored_wideresnet.build_wideresnet(num_classes=args.num_classes)

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {}.".format(args.arch))

    logging.info("Total Model params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # setup and load CIFAR10
    if args.num_classes == 10:
        if args.strong_augmentation:
            train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_STRONG_TRAIN, train=True, subset=args.debug)
        else:
            train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_WEAK_TRAIN, train=True, subset=args.debug)
    else:
        raise RuntimeError("unsupported CIFAR class count: {}".format(args.num_classes))

    # split the data class balanced based on a total count.
    # returns subset, remainder
    val_dataset, train_dataset = train_dataset.data_split_class_balanced(subset_count=args.num_labeled_datapoints)
    # set the validation augmentation to just normalize (.dataset since val_dataset is a Subset, not a full dataset)
    val_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_TEST)

    train_dataset_labeled, train_dataset_unlabeled = train_dataset.data_split_class_balanced(subset_count=args.num_labeled_datapoints)

    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    return model, train_dataset_labeled, train_dataset_unlabeled, val_dataset, test_dataset



def build_gmm(model, pytorch_dataset, epoch, train_stats, args):
    model.eval()

    dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    start_time = time.time()
    dataset_logits = list()
    dataset_labels = list()

    for batch_idx, tensor_dict in enumerate(dataloader):
        inputs = tensor_dict[0].cuda()
        labels = tensor_dict[1].cuda()

        # FP16 training
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        dataset_logits.append(outputs.detach().cpu())
        dataset_labels.append(labels.detach().cpu())

    bucketed_dataset_logits = list()
    # join together the individual batches of numpy logit data
    dataset_logits = torch.cat(dataset_logits)
    dataset_labels = torch.cat(dataset_labels)
    unique_class_labels = torch.unique(dataset_labels)

    for i in range(len(unique_class_labels)):
        c = unique_class_labels[i]
        bucketed_dataset_logits.append(dataset_logits[dataset_labels == c])

    # gmm_list = list()
    gmm_list_skl = list()
    for i in range(len(unique_class_labels)):
        class_c_logits = bucketed_dataset_logits[i]
        # gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=args.cluster_per_class, tolerance=1e-4, max_iter=50, isCauchy= (args.inference_method == 'cauchy'))
        # gmm.fit(class_c_logits)
        # while np.any(np.isnan(gmm.get("sigma").detach().cpu().numpy())):
        #     gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=args.cluster_per_class, tolerance=1e-4, max_iter=50, isCauchy= (args.inference_method == 'cauchy'))
        #     gmm.fit(class_c_logits)
        # gmm_list.append(gmm)
        gmm_skl = sklearn.mixture.GaussianMixture(n_components=args.cluster_per_class)
        gmm_skl.fit(class_c_logits.numpy())
        gmm_list_skl.append(gmm_skl)

    class_preval = utils.compute_class_prevalance(dataloader)
    logging.info(class_preval)
    class_preval = torch.tensor(list(class_preval.values()))

    # generate weights, mus and sigmas
    weights = class_preval.repeat_interleave(args.cluster_per_class)
    pi = torch.cat([torch.FloatTensor(gmm.weights_) for gmm in gmm_list_skl])
    weights *= pi
    # mus = torch.cat([gmm.get("mu") for gmm in gmm_list])
    mus = torch.cat([torch.FloatTensor(gmm.means_) for gmm in gmm_list_skl])
    # sigmas = torch.cat([gmm.get("sigma") for gmm in gmm_list])
    sigmas = torch.cat([torch.FloatTensor(gmm.covariances_) for gmm in gmm_list_skl])
    gmm = GMM(n_features=args.num_classes, n_clusters=weights.shape[0], weights=weights, mu=mus, sigma=sigmas)

    # merge the individual sklearn GMMs
    # means = np.concatenate([gmm.means_ for gmm in gmm_list_skl])
    # covariances = np.concatenate([gmm.covariances_ for gmm in gmm_list_skl])
    # precisions = np.concatenate([gmm.precisions_ for gmm in gmm_list_skl])
    # precisions_chol = np.concatenate([gmm.precisions_cholesky_ for gmm in gmm_list_skl])
    # gmm_skl = sklearn.mixture.GaussianMixture(n_components=args.num_classes * 2)
    # gmm_skl.weights_ = weights.numpy()
    # gmm_skl.means_ = means
    # gmm_skl.covariances_ = covariances
    # gmm_skl.precisions_ = precisions
    # gmm_skl.precisions_cholesky_ = precisions_chol

    wall_time = time.time() - start_time
    train_stats.add(epoch, 'gmm_build_wall_time', wall_time)

    return gmm
    # return gmm_skl


def train_epoch(model, pytorch_dataset_labeled, optimizer, criterion, epoch, train_stats, args, scheduler=None, nb_reps=1):

    loss_list = list()
    accuracy_list = list()
    model.train()
    scaler = torch.cuda.amp.GradScaler()


    # use the effective dataset and shuffle the data to interleave the labeled and psudo-labeled together
    dataloader = torch.utils.data.DataLoader(pytorch_dataset_labeled, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    batch_count = nb_reps * len(dataloader)
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
            if args.amp:
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
            if args.soft_labels:
                # convert soft labels into hard for accuracy
                labels = torch.argmax(labels, dim=-1)
            accuracy = torch.sum(pred == labels) / len(pred)

            # nan loss values are ignored when using AMP, so ignore them for the average
            if not np.isnan(batch_loss.detach().cpu().numpy()):
                loss_list.append(batch_loss.item())
                accuracy_list.append(accuracy.item())
                if scheduler is not None:
                    scheduler.step()
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

    avg_loss = np.mean(loss_list)
    avg_accuracy = np.mean(accuracy_list)
    wall_time = time.time() - start_time

    if loss_nan_count > 0:
        logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_loss)
    train_stats.add(epoch, 'train_accuracy', avg_accuracy)




def train_epoch_ul(model, pytorch_dataset_labeled, dataset_pseudo_labeled, optimizer, criterion, epoch, train_stats, args, scheduler=None, nb_reps=1):
    loss_list = list()
    accuracy_list = list()
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    mu = 7  # the unlabeled data is 7x the size of the labeled

    # if len(dataset_pseudo_labeled) < mu * args.batch_size:
    #     logging.info("Not enough valid pseudo-labels for epoch {}. Skipping Semi-Supervised training for this epoch.".format(epoch))
    #     return

    # create a copy of the labeled dataset to append the pseudolabels to
    effective_dataset = copy.deepcopy(pytorch_dataset_labeled)
    if args.soft_labels:
        for i in range(len(effective_dataset.targets)):
            v = np.zeros((args.num_classes), dtype=float)
            v[effective_dataset.targets[i]] = 1
            effective_dataset.targets[i] = v

    # effective_dataset.append_dataset(dataset_pseudo_labeled)
    # include only the normalization and to tensor transforms
    dataset_pseudo_labeled.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH.normalize)

    # use the effective dataset and shuffle the data to interleave the labeled and psudo-labeled together
    dataloader_ul = torch.utils.data.DataLoader(dataset_pseudo_labeled, batch_size=(args.batch_size * mu), shuffle=True, num_workers=args.num_workers/2, worker_init_fn=cifar_datasets.worker_init_fn, drop_last=True)


    # use the effective dataset and shuffle the data to interleave the labeled and psudo-labeled together
    dataloader = torch.utils.data.DataLoader(effective_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers/2, worker_init_fn=cifar_datasets.worker_init_fn, drop_last=True)


    batch_count = nb_reps * min(len(dataloader), len(dataloader_ul))
    start_time = time.time()
    loss_nan_count = 0

    iter_ul = iter(dataloader_ul)

    for rep_count in range(nb_reps):

        for batch_idx, tensor_dict_l in enumerate(dataloader):
            # adjust for the rep offset
            batch_idx = rep_count * len(dataloader) + batch_idx
            optimizer.zero_grad()

            inputs_l = tensor_dict_l[0]
            targets_l = tensor_dict_l[1]

            try:
                tensor_dict_ul = next(iter_ul)
                inputs_ul = tensor_dict_ul[0]
                targets_ul = tensor_dict_ul[1]

                inputs = utils.interleave(torch.cat((inputs_l, inputs_ul)), mu + 1)
                inputs = inputs.cuda()
                targets_l = targets_l.cuda()
                targets_ul = targets_ul.cuda()

            except:
                inputs_ul = None
                targets_ul = None

                inputs = inputs_l
                inputs = inputs.cuda()
                targets_l = targets_l.cuda()


            # FP16 training
            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)

                    if inputs_ul is not None:
                        logits = utils.de_interleave(logits, mu + 1)
                        logits_l = logits[:args.batch_size]
                        logits_u = logits[args.batch_size:]

                        loss_l = criterion(logits_l, targets_l)
                        loss_ul = criterion(logits_u, targets_ul)
                        batch_loss = loss_l + loss_ul
                    else:
                        logits_l = logits
                        batch_loss = criterion(logits_l, targets_l)

                    scaler.scale(batch_loss).backward()
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()
            else:
                logits = model(inputs)
                if inputs_ul is not None:
                    logits = utils.de_interleave(logits, mu + 1)
                    logits_l = logits[:args.batch_size]
                    logits_u = logits[args.batch_size:]

                    loss_l = criterion(logits_l, targets_l)
                    loss_ul = criterion(logits_u, targets_ul)
                    batch_loss = loss_l + loss_ul
                else:
                    logits_l = logits
                    batch_loss = criterion(logits_l, targets_l)

                batch_loss.backward()
                optimizer.step()

            pred_l = torch.argmax(logits_l, dim=-1)
            if inputs_ul is not None:
                pred_ul = torch.argmax(logits_u, dim=-1)

            if args.soft_labels:
                # convert soft labels into hard for accuracy
                targets_l = torch.argmax(targets_l, dim=-1)
                if inputs_ul is not None:
                    targets_ul = torch.argmax(targets_ul, dim=-1)

            acc_l = pred_l == targets_l
            if inputs_ul is not None:
                acc_ul = pred_ul == targets_ul
                acc = torch.cat((acc_l, acc_ul)).type(torch.float64)
            else:
                acc = acc_l.type(torch.float64)

            accuracy = torch.mean(acc)

            # nan loss values are ignored when using AMP, so ignore them for the average
            if not np.isnan(batch_loss.detach().cpu().numpy()):
                loss_list.append(batch_loss.item())
                accuracy_list.append(accuracy.item())
                if scheduler is not None:
                    scheduler.step()
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

    avg_loss = np.mean(loss_list)
    avg_accuracy = np.mean(accuracy_list)
    wall_time = time.time() - start_time

    if loss_nan_count > 0:
        logging.info("epoch has {} batches with nan loss.".format(loss_nan_count))

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_loss)
    train_stats.add(epoch, 'train_accuracy', avg_accuracy)


def eval_model(model, pytorch_dataset, criterion, train_stats, split_name, epoch, gmm, args):
    if pytorch_dataset is None or len(pytorch_dataset) == 0:
        return

    dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    batch_count = len(dataloader)
    model.eval()
    start_time = time.time()

    loss_list = list()
    gt_labels = list()
    softmax_preds = list()

    if gmm is not None:
        gmm_preds = list()
        cauchy_preds = list()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            labels_cpu = labels.detach().cpu().numpy()
            gt_labels.extend(labels_cpu)

            if args.amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            batch_loss = criterion(outputs, labels)
            loss_list.append(batch_loss.item())
            pred = torch.argmax(outputs, dim=-1).detach().cpu().numpy()
            softmax_preds.extend(pred)

            if gmm is not None:
                # passing the logits acquired before the softmax to gmm as inputs
                gmm_inputs = outputs.detach().cpu().numpy()
                gmm_resp = gmm.predict_proba(gmm_inputs)  # N*1, N*K
                if not isinstance(gmm_resp, np.ndarray):
                    gmm_resp = gmm_resp.detach().cpu().numpy()
                gmm_pred = np.argmax(gmm_resp, axis=-1) // args.cluster_per_class
                gmm_preds.extend(gmm_pred)

                if hasattr(gmm, 'predict_cauchy_probability'):
                    cauchy_unnorm_resp, _, cauchy_resp = gmm.predict_cauchy_probability(gmm_inputs)
                    if not isinstance(cauchy_resp, np.ndarray):
                        cauchy_resp = cauchy_resp.detach().cpu().numpy()
                    cauchy_pred = np.argmax(cauchy_resp, axis=-1)
                    cauchy_preds.extend(cauchy_pred)

            if batch_idx % 100 == 0:
                # log loss and current GPU utilization
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

    avg_loss = np.nanmean(loss_list)
    gt_labels = np.asarray(gt_labels)
    softmax_preds = np.asarray(softmax_preds)
    softmax_accuracy = np.mean((softmax_preds == gt_labels).astype(float))

    wall_time = time.time() - start_time
    train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
    train_stats.add(epoch, '{}_loss'.format(split_name), avg_loss)
    train_stats.add(epoch, '{}_softmax_accuracy'.format(split_name), softmax_accuracy)

    if gmm is not None:
        gmm_preds = np.asarray(gmm_preds)
        cauchy_preds = np.asarray(cauchy_preds)
        gt_labels_list = gt_labels.tolist()
        gmm_preds_list = gmm_preds.tolist()
        cauchy_preds_list = cauchy_preds.tolist()
        if len(gmm_preds) > 0:
            gmm_accuracy = (gmm_preds == gt_labels).astype(float)
            gmm_accuracy = np.mean(gmm_accuracy)
            train_stats.add(epoch, '{}_gmm_accuracy'.format(split_name), gmm_accuracy)
            # train_stats.add(epoch, '{}_gmm_gt_labels'.format(split_name), gt_labels_list)
            # train_stats.add(epoch, '{}_gmm_preds'.format(split_name), gmm_preds_list)
            # train_stats.render_and_save_confusion_matrix(gt_labels, gmm_preds, args.output_filepath, '{}_gmm_confusion_matrix'.format(split_name), epoch)

        if len(cauchy_preds) > 0:
            cauchy_accuracy = (cauchy_preds == gt_labels).astype(float)
            cauchy_accuracy = np.mean(cauchy_accuracy)
            train_stats.add(epoch, '{}_cauchy_accuracy'.format(split_name), cauchy_accuracy)
            # train_stats.add(epoch, '{}_cauchy_gt_labels'.format(split_name), gt_labels.tolist())
            # train_stats.add(epoch, '{}_cauchy_preds'.format(split_name), cauchy_preds.tolist())
            #train_stats.render_and_save_confusion_matrix(gt_labels, cauchy_preds, args.output_filepath, '{}_cauchy_confusion_matrix'.format(split_name), epoch)


def psuedolabel_data_fixmatch(model, pytorch_dataset_unlabeled, epoch, train_stats, args):

    lcl_dataset = copy.deepcopy(pytorch_dataset_unlabeled)
    lcl_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_FIXMATCH)

    # create empty dataset to hold the pseudo-labels
    pseudo_labeled_dataset = cifar_datasets.Cifar10(empty=True)
    pseudo_labeled_dataset.set_transforms(None)

    dataloader = torch.utils.data.DataLoader(lcl_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                worker_init_fn=cifar_datasets.worker_init_fn)


    pseudo_label_threshold = np.asarray(args.pseudo_label_threshold)
    model.eval()
    pl_counts_per_class = list()
    gt_counts_per_class = list()
    tp_counter_per_class = list()
    pl_accuracy_per_class = list()
    pl_accuracy = list()
    for i in range(args.num_classes):
        pl_counts_per_class.append(0)
        gt_counts_per_class.append(0)
        tp_counter_per_class.append(0)
        pl_accuracy_per_class.append(0)

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs_weak, inputs_strong_normalized, inputs_strong = tensor_dict[0]

            inputs_weak = inputs_weak.cuda()
            labels = tensor_dict[1].cuda()
            # index = tensor_dict[2].cpu().detach()

            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(inputs_weak)
            else:
                logits = model(inputs_weak)

            logits = logits / args.tau
            logits = torch.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()

            pred = np.argmax(logits, axis=-1)
            score = np.max(logits, axis=-1)
            valid_pl = score > pseudo_label_threshold
            if np.any(valid_pl):
                for idx in range(len(valid_pl)):
                    if valid_pl[idx]:
                        pl_class = pred[idx]
                        true_class = labels[idx]
                        pl_counts_per_class[pl_class] += 1
                        gt_counts_per_class[true_class] += 1
                        acc = int(pl_class == true_class)
                        pl_accuracy.append(acc)
                        pl_accuracy_per_class[true_class] += 1
                        if acc:
                            tp_counter_per_class[true_class] += 1

                        img = inputs_strong[idx,].detach().cpu().numpy()
                        if args.soft_labels:
                            tgt = np.asarray(logits[idx], dtype=float)
                        else:
                            tgt = pl_class

                        pseudo_labeled_dataset.add(img, tgt)

    pl_accuracy = np.mean(pl_accuracy)
    pl_accuracy_per_class = np.asarray(tp_counter_per_class) / np.asarray(gt_counts_per_class)
    pl_accuracy_per_class[np.isnan(pl_accuracy_per_class)] = 0.0
    pl_accuracy_per_class = pl_accuracy_per_class.tolist()

    # get the average accuracy of the pseudo-labels (this data is not available in real SSL applications, since the unlabeled population would truly be unlabeled
    train_stats.add(epoch, 'pseudo_labeling_accuracy', float(np.nanmean(pl_accuracy)))
    train_stats.add(epoch, 'num_pseudo_labels', int(len(pseudo_labeled_dataset)))

    # update the training metadata
    train_stats.add(epoch, 'pseudo_label_counts_per_class', pl_counts_per_class)
    train_stats.add(epoch, 'pseudo_labeling_accuracy_per_class', pl_accuracy_per_class)

    vals = np.asarray(pl_counts_per_class) / np.sum(pl_counts_per_class)
    train_stats.add(epoch, 'pseudo_label_percentage_per_class', vals.tolist())
    vals = np.asarray(tp_counter_per_class) / np.sum(tp_counter_per_class)
    train_stats.add(epoch, 'pseudo_label_gt_percentage_per_class', vals.tolist())


    # TODO add logging and metric capture, now that the code works
    return pseudo_labeled_dataset


def train(args):
    if not os.path.exists(args.output_dirpath):
        # safety check that the output directory exists
        os.makedirs(args.output_dirpath)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(args.output_dirpath, 'log.txt'))

    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(args)

    model, train_dataset_labeled, train_dataset_unlabeled, val_dataset, test_dataset = setup(args)

    # write the args configuration to disk
    with open(os.path.join(args.output_dirpath, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, ensure_ascii=True, indent=2)

    train_start_time = time.time()

    # Move model to device
    model.cuda()

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()



    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    epoch = -1
    best_epoch = 0
    best_model = model

    # *************************************
    # ******** Supervised Training ********
    # *************************************
    # train the model until it has converged on the labeled data
    # setup early stopping on convergence using LR reduction on plateau
    optimizer = get_optimizer(args, model)
    plateau_scheduler_sl = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduction_factor, patience=10, threshold=args.loss_eps, max_num_lr_reductions=0)
    # train epochs until loss converges
    if not args.disable_sl:  # if the user has not disabled supervised learning
        while not plateau_scheduler_sl.is_done() and epoch < trainer.MAX_EPOCHS:
            epoch += 1
            logging.info("Epoch (supervised): {}".format(epoch))

            plot_loss_acc(train_stats, args, best_epoch)

            logging.info("  training against fully labeled dataset.")
            train_epoch(model, train_dataset_labeled, optimizer, criterion, epoch, train_stats, args, nb_reps=10)

            logging.info("  evaluating against validation data.")
            eval_model(model, val_dataset, criterion, train_stats, "val", epoch, None, args)

            val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
            plateau_scheduler_sl.step(val_loss)

            # update global metadata stats
            train_stats.add_global('training_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
            train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
            train_stats.add_global('num_epochs_trained', epoch)

            # write copy of current metadata metrics to disk
            train_stats.export(args.output_dirpath)

            # handle early stopping when loss converges
            # if plateau_scheduler_sl.num_bad_epochs == 0:  # use if you only want literally the best epoch, instead of taking into account the loss eps
            if plateau_scheduler_sl.is_equiv_to_best_epoch:
                logging.info('Updating best model with epoch: {} loss: {}'.format(epoch, val_loss))
                best_model = copy.deepcopy(model)
                best_epoch = epoch

                # update the global metrics with the best epoch
                train_stats.update_global(epoch)


    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_dirpath, 'supervised-model.pt'))

    # ******************************************
    # ******** Semi-Supervised Training ********
    # ******************************************
    gmm = None
    if not args.disable_ssl:
        # get a new copy of the optimizer with a reset learning rate
        optimizer = get_optimizer(args, model)
        # re-setup LR reduction on plateau, so that it uses the learning rate reductions
        plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions)

        # train epochs until loss converges
        # reset model back to the "best" model
        model = copy.deepcopy(best_model)
        model.cuda()  # move the model back to the GPU (saving moved the best model back to the cpu)
        while not plateau_scheduler.is_done() and epoch < trainer.MAX_EPOCHS:
            epoch += 1
            logging.info("Epoch (semi-supervised): {}".format(epoch))

            plot_loss_acc(train_stats, args, best_epoch)

            # TODO rework this so train_epoch_ul is the main function that gets written based on PL methods. So that fixmatch can be dragged into the main train loop instead of being slower and separate?
            logging.info("  pseudo-labeling unannotated examples.")
            # TODO Rushabh, Summet & JD replace this function with the GMM and Cauchy PL methods
            # This moves instances from the unlabeled population into the labeled population
            pseudo_labeled_dataset = psuedolabel_data_fixmatch(model, train_dataset_unlabeled, epoch, train_stats, args)

            logging.info("  training against annotated and pseudo-labeled data.")
            train_epoch_ul(model, train_dataset_labeled, pseudo_labeled_dataset, optimizer, criterion, epoch, train_stats, args)

            logging.info("  building the GMM models on the labeled training data.")
            gmm = build_gmm(model, train_dataset_labeled, epoch, train_stats, args)

            logging.info("  evaluating against validation data, using softmax and gmm")
            eval_model(model, val_dataset, criterion, train_stats, "val", epoch, gmm, args)

            val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
            plateau_scheduler.step(val_loss)

            # update global metadata stats
            train_stats.add_global('training_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
            train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
            train_stats.add_global('num_epochs_trained', epoch)

            # write copy of current metadata metrics to disk
            train_stats.export(args.output_dirpath)

            # handle early stopping when loss converges
            # if plateau_scheduler_sl.num_bad_epochs == 0:  # use if you only want literally the best epoch, instead of taking into account the loss eps
            if plateau_scheduler.is_equiv_to_best_epoch:
                logging.info('Updating best model with epoch: {} loss: {}'.format(epoch, val_loss))
                best_model = copy.deepcopy(model)
                best_epoch = epoch

                # update the global metrics with the best epoch
                train_stats.update_global(epoch)

    best_model.cuda()  # move the model back to the GPU (saving moved the best model back to the cpu)
    logging.info("  building the GMM models on the labeled training data.")
    gmm = build_gmm(best_model, train_dataset_labeled, best_epoch, train_stats, args)

    logging.info('Evaluating model against test dataset using softmax and gmm')
    eval_model(best_model, test_dataset, criterion, train_stats, "test", best_epoch, gmm, args)

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logging.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_dirpath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_dirpath, 'model.pt'))