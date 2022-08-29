import os
import time
import copy
import numpy as np
import torch
import torchvision
import json
import logging

import cifar_datasets
import metadata
from gmm_module import GMM
import lr_scheduler
import flavored_resnet18
import flavored_wideresnet

MAX_EPOCHS = 300

logger = logging.getLogger()


def setup(args):
    # load stock models from https://pytorch.org/vision/stable/models.html
    model = None
    if args.starting_model is not None:
        # warning, this over rides the args.arch selection
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

    logger.info("Total Model params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # setup and load CIFAR10
    if args.num_classes == 10:
        train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TRAIN, train=True, subset=args.debug)
    else:
        raise RuntimeError("unsupported class count: {}".format(args.num_classes))

    train_dataset, val_dataset = train_dataset.train_val_split(val_fraction=args.val_fraction)
    # set the validation augmentation to just normalize (.dataset since val_dataset is a Subset, not a full dataset)
    val_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_TEST)

    train_dataset_labeled, train_dataset_unlabeled = train_dataset.data_split_class_balanced(subset_count=args.num_labeled_datapoints)

    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    return model, train_dataset_labeled, train_dataset_unlabeled, val_dataset, test_dataset


def psuedolabel_data(model, train_dataset_labeled, train_dataset_unlabeled, gmm, epoch, train_stats, args):

    ul_dataloader = torch.utils.data.DataLoader(train_dataset_unlabeled,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                worker_init_fn=cifar_datasets.worker_init_fn)
    # TODO update this to use a per-class threshold, but that is an extension of the baseline technique
    denominator_threshold = 0.0
    # denominator_threshold = 500.0
    resp_threshold = 0.95
    size_threshold = 2  # the number of data points from each class to add to the labeled population
    # size_threshold = 0.02
    filtered_denominators = []
    filtered_labels = []
    filtered_indicies = []
    filtered_data_resp = []
    filtered_data_weighted_prob = []
    model.eval()
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(ul_dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()
            index = tensor_dict[2].cpu().detach()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                gmm_inputs = outputs.detach().cpu()
                gmm_prob_weighted, gmm_prob_sum, gmm_resp = gmm.predict_probability(gmm_inputs)

                # threshold filtering

                # list of boolean where value is False for items having lower prob_sum then threshold, True otherwise
                # denominator_filter = torch.squeeze(gmm_prob_sum > denominator_threshold)
                filtered_denominators.append(gmm_prob_sum)
                filtered_labels.append(labels.cpu().detach())
                filtered_indicies.append(index)
                filtered_data_resp.append(gmm_resp)
                filtered_data_weighted_prob.append(gmm_prob_weighted)

    filtered_denominators = torch.cat(filtered_denominators, dim=0)
    filtered_labels = torch.cat(filtered_labels, dim=-1)
    filtered_indicies = torch.cat(filtered_indicies, dim=-1)
    filtered_data_resp = torch.cat(filtered_data_resp, dim=0)
    filtered_data_weighted_prob = torch.cat(filtered_data_weighted_prob, dim=0)

    filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob = pseudo_label_denominator_filter(filtered_denominators, filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob)

    # DONE: RUSHABH | create a set of subfunctions which sorts/filters the gmm outputs using different methods. So it takes as input a list of all the GMM outputs, and then it returns a sorted list (potentially smaller if there is a filter). This subfunction allows us to swap out different pseudo-labeling strategies.
    # DONE: RUSHABH | i.e. labels, indicies, data_resp = pseudo_label_filter_denominator(labels, indicies, data_resp)
    # DONE: RUSHABH | i.e. labels, indicies, data_resp = pseudo_label_filter_numerator(labels, indicies, data_resp)

    # TODO start figuring out what the filtering and disqualifying filters are for the GMM to remove bad examples
    # DONE: JD | Experiment with pseudo labeling sorting by the largest numerator per class, then do top k

    filtered_labels, filtered_indicies, filtered_preds = pseudo_label_numerator_filter(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, weighted=True)

    class_counter = [0] * int(args.num_classes)
    class_accuracy = [0] * int(args.num_classes)
    added_accuracies = []
    used_indices = []

    for i in range(len(filtered_preds)):
        pred = filtered_preds[i]
        # assumption: 10 classes are 0 to 10
        # if class_counter[pred] < points_per_class:
        if class_counter[pred] < size_threshold:
            class_counter[pred] += 1
            label = filtered_labels[i].item()
            index = filtered_indicies[i].item()
            used_indices.append(index)

            img, target = train_dataset_unlabeled.get_raw_datapoint(index)
            train_dataset_labeled.add_datapoint(img, target)

            acc = float(pred == label)
            class_accuracy[pred] += acc
            added_accuracies.append(acc)
    # delete the indices transferred to the labeled population. Do this after the move, so that we don't modify the index numbers during the move
    train_dataset_unlabeled.remove_datapoints_by_index(used_indices)

    # get the average accuracy of the pseudo-labels (this data is not available in real SSL applications, since the unlabeled population would truly be unlabeled
    pseudo_labeling_accuracy = float(np.mean(added_accuracies))
    # update the training metadata
    train_stats.add(epoch, 'pseudo_labeling_accuracy', pseudo_labeling_accuracy)
    train_stats.add(epoch, 'pseudo_label_counts_per_class', class_counter)
    train_stats.add(epoch, 'pseudo_labeling_accuracy_per_class', class_accuracy)

    # DONE: JD |log the per-class accuracy of the pseudo-labels
    # TODO build confusion matrix for the psuedo labeling
    # TODO build confusion matrix for the validation accuracy using the gmm to predict labels
    # TODO experiment and compute the baseline accuracy when using just 250 or 4000 labels.


def pseudo_label_denominator_filter(denominators, labels, indices, data_resp, data_weighted_probs):
    denominator_threshold = 0.0
    denominator_filter = torch.squeeze(denominators > denominator_threshold)
    labels = labels[denominator_filter]
    data_resp = data_resp[denominator_filter]
    indices = indices[denominator_filter]
    data_weighted_probs = data_weighted_probs[denominator_filter]

    return labels, indices, data_resp, data_weighted_probs


def pseudo_label_numerator_filter(labels, indices, data_resp, data_weighted_probs, weighted=True):
    numerator_resp_threshold = 0.95

    data = data_resp if not weighted else data_weighted_probs
    max_numerator, preds = torch.max(data, dim=-1)
    sorted_numerators, max_sorted_indices = max_numerator.sort(descending=True)

    if not weighted:
        max_sorted_indices = max_sorted_indices[sorted_numerators > numerator_resp_threshold]

    labels = labels[max_sorted_indices]
    indices = indices[max_sorted_indices]
    preds = preds[max_sorted_indices]

    return labels, indices, preds


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


    gmm_list = list()
    bucketed_dataset_logits = list()
    # join together the individual batches of numpy logit data
    dataset_logits = torch.cat(dataset_logits)
    dataset_labels = torch.cat(dataset_labels)
    unique_class_labels = torch.unique(dataset_labels)

    for i in range(len(unique_class_labels)):
        c = unique_class_labels[i]
        bucketed_dataset_logits.append(dataset_logits[dataset_labels == c])


    class_instances = list()
    for i in range(len(unique_class_labels)):
        class_c_logits = bucketed_dataset_logits[i]
        class_instances.append(class_c_logits.shape[0])
        gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
        gmm.fit(class_c_logits)
        while np.any(np.isnan(gmm.get("sigma").detach().cpu().numpy())):
            gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
            gmm.fit(class_c_logits)
        gmm_list.append(gmm)

    class_preval = compute_class_prevalance(dataloader)
    logger.info(class_preval)
    class_preval = torch.tensor(list(class_preval.values()))

    # generate weights, mus and sigmas
    weights = class_preval
    mus = torch.cat([gmm.get("mu") for gmm in gmm_list])
    sigmas = torch.cat([gmm.get("sigma") for gmm in gmm_list])
    gmm = GMM(n_features=args.num_classes, n_clusters=weights.shape[0], weights=weights, mu=mus, sigma=sigmas)

    wall_time = time.time() - start_time
    train_stats.add(epoch, 'gmm_build_wall_time', wall_time)

    return gmm


def train_epoch(model, pytorch_dataset, optimizer, criterion, epoch, train_stats, args, scheduler=None, gmm_models=None):

    avg_loss = 0
    avg_accuracy = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    batch_count = len(dataloader)
    start_time = time.time()
    dataset_logits = list()
    dataset_labels = list()

    for batch_idx, tensor_dict in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = tensor_dict[0].cuda()
        labels = tensor_dict[1].cuda()

        # FP16 training
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        pred = torch.argmax(outputs, dim=-1)
        accuracy = torch.sum(pred == labels) / len(pred)

        if gmm_models is not None:
            dataset_logits.append(outputs.detach().cpu())
            dataset_labels.append(labels.detach().cpu())

        # nan loss values are ignored when using AMP, so ignore them for the average
        if not np.isnan(loss.detach().cpu().numpy()):
            avg_loss += loss.item()
            avg_accuracy += accuracy.item()
            if scheduler is not None:
                scheduler.step()

        if batch_idx % 100 == 0:
            if scheduler is not None:
                logger.info('  batch {}/{}  loss: {:8.8g}, lr: {}, cyclic_lr: {}'.format(batch_idx, batch_count, loss.item(), optimizer.param_groups[0]['lr'], scheduler.get_last_lr()[0]))
            else:
                logger.info('  batch {}/{}  loss: {:8.8g}, lr: {}'.format(batch_idx, batch_count, loss.item(), optimizer.param_groups[0]['lr']))

    avg_loss /= batch_count
    avg_accuracy /= batch_count
    wall_time = time.time() - start_time

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_loss)
    train_stats.add(epoch, 'train_accuracy', avg_accuracy)

    if gmm_models is not None:
        bucketed_dataset_logits = list()
        # join together the individual batches of numpy logit data
        dataset_logits = torch.cat(dataset_logits)
        dataset_labels = torch.cat(dataset_labels)
        unique_class_labels = torch.unique(dataset_labels)

        for i in range(len(unique_class_labels)):
            c = unique_class_labels[i]
            bucketed_dataset_logits.append(dataset_logits[dataset_labels == c])

        class_instances = list()
        for i in range(len(unique_class_labels)):
            class_c_logits = bucketed_dataset_logits[i]
            class_instances.append(class_c_logits.shape[0])
            gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
            gmm.fit(class_c_logits)
            while np.any(np.isnan(gmm.get("sigma").detach().cpu().numpy())):
                gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
                gmm.fit(class_c_logits)
            gmm_models.append(gmm)

    return gmm_models


def eval_model(model, pytorch_dataset, criterion, train_stats, split_name, epoch, gmm, args):
    if pytorch_dataset is None or len(pytorch_dataset) == 0:
        return

    dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    model.eval()
    start_time = time.time()

    loss = list()
    softmax_accuracy = list()

    if gmm is not None:
        gmm_accuracy = list()
        cauchy_accuracy = list()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss.append(criterion(outputs, labels).item())
                pred = torch.argmax(outputs, dim=-1)
                accuracy = torch.sum(pred == labels) / len(pred)
                softmax_accuracy.append(accuracy.reshape(-1))

                if gmm is not None:
                    # passing the logits acquired before the softmax to gmm as inputs
                    gmm_inputs = outputs.detach().cpu()
                    _, _, gmm_resp = gmm.predict_probability(gmm_inputs)  # N*1, N*K

                    _, cauchy_resp,cauchy_unnorm_resp = gmm.predict_cauchy_probability(gmm_inputs)

                    gmm_pred = torch.argmax(gmm_resp, dim=-1)
                    labels_cpu = labels.detach().cpu()
                    accuracy_g = torch.sum(gmm_pred == labels_cpu) / len(gmm_pred)
                    gmm_accuracy.append(accuracy_g.reshape(-1))

                    cauchy_pred = torch.argmax(cauchy_resp,dim=-1)
                    accuracy_c = torch.sum(cauchy_pred == labels_cpu)/ len(cauchy_pred)
                    cauchy_accuracy.append(accuracy_c.reshape(-1))

    avg_loss = np.mean(loss)
    softmax_accuracy = torch.mean(torch.cat(softmax_accuracy, dim=-1))

    wall_time = time.time() - start_time
    train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
    train_stats.add(epoch, '{}_loss'.format(split_name), avg_loss)
    train_stats.add(epoch, '{}_softmax_accuracy'.format(split_name), softmax_accuracy.item())

    if gmm is not None:
        gmm_accuracy = torch.mean(torch.cat(gmm_accuracy, dim=-1))
        train_stats.add(epoch, '{}_gmm_accuracy'.format(split_name), gmm_accuracy.item())

        cauchy_accuracy = torch.mean(torch.cat(cauchy_accuracy,dim=-1))
        train_stats.add(epoch, '{}_cauchy_accuracy'.format(split_name), cauchy_accuracy.item())


def compute_class_prevalance(dataloader):
    label_list = list()
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            #inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            label_list.append(labels.detach().cpu().numpy())

    label_list = np.concatenate(label_list).reshape(-1)
    unique_labels = np.unique(label_list)
    N = len(label_list)
    class_preval = {}
    for i in range(len(unique_labels)):
        c = unique_labels[i]
        count = np.sum(label_list == c)
        class_preval[c] = count/N

    return class_preval



# TODO, setup data augmentation and disqualify samples from being added to the labeled population based on contrastive learning. If the labeled changes under N instances of data augmentation, then the model is not confident about it.

# TODO Test greedy pseudo-labeling, where for each call to this function, we take the top k=2 pseudo-labeled from each class. (how to determine the best samples, look at resp)

def train(args):
    if not os.path.exists(args.output_filepath):
        os.makedirs(args.output_filepath)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(args.output_filepath, 'log.txt'))

    logging.getLogger().addHandler(logging.StreamHandler())

    logger.info(args)

    model, train_dataset_labeled, train_dataset_unlabeled, val_dataset, test_dataset = setup(args)

    # write the args configuration to disk
    dvals = vars(args)
    with open(os.path.join(args.output_filepath, 'config.json'), 'w') as fh:
        json.dump(dvals, fh, ensure_ascii=True, indent=2)

    train_start_time = time.time()

    # Move model to device
    model.cuda()

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()

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

    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    epoch = -1
    best_epoch = 0
    best_model = model

    # *************************************
    # ******** Supervised Training ********
    # *************************************
    # train the model until it has converged on the the labeled data
    # setup early stopping on convergence using LR reduction on plateau
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.0, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=0)
    # train epochs until loss converges
    while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
        epoch += 1
        logger.info("Epoch (supervised): {}".format(epoch))

        logger.info("  training against fully labeled dataset.")
        train_epoch(model, train_dataset_labeled, optimizer, criterion, epoch, train_stats, args)

        logger.info("  evaluating against validation data.")
        eval_model(model, val_dataset, criterion, train_stats, "val", epoch, None, args)

        val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
        plateau_scheduler.step(val_loss)

        # update global metadata stats
        train_stats.add_global('training_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
        train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
        train_stats.add_global('num_epochs_trained', epoch)

        # write copy of current metadata metrics to disk
        train_stats.export(args.output_filepath)

        # handle early stopping when loss converges
        if plateau_scheduler.num_bad_epochs == 0:
            logger.info('Updating best model with epoch: {} loss: {}'.format(epoch, val_loss))
            best_model = copy.deepcopy(model)
            best_epoch = epoch

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

    # ******************************************
    # ******** Semi-Supervised Training ********
    # ******************************************
    # re-setup LR reduction on plateau, so that it uses the learning rate reductions
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions)

    # train epochs until loss converges
    gmm = None
    while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
        epoch += 1
        logger.info("Epoch (semi-supervised): {}".format(epoch))

        logger.info("  training against annotated and pseudo-labeled data.")
        train_epoch(model, train_dataset_labeled, optimizer, criterion, epoch, train_stats, args)

        logger.info("  building the GMM models on the labeled training data.")
        gmm = build_gmm(model, train_dataset_labeled, epoch, train_stats, args)

        logger.info("  evaluating against validation data, using softmax and gmm")
        eval_model(model, val_dataset, criterion, train_stats, "val", epoch, gmm, args)

        # TODO test completely re-psuedo labeling every epoch, instead of it being a one way function
        logger.info("  pseudo-labeling unannotated examples.")
        # This moves instances from the unlabeled population into the labeled population
        psuedolabel_data(model, train_dataset_labeled, train_dataset_unlabeled, gmm, epoch, train_stats, args)

        val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
        plateau_scheduler.step(val_loss)

        # update global metadata stats
        train_stats.add_global('training_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
        train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
        train_stats.add_global('num_epochs_trained', epoch)

        # write copy of current metadata metrics to disk
        train_stats.export(args.output_filepath)

        # handle early stopping when loss converges
        if plateau_scheduler.num_bad_epochs == 0:
            logger.info('Updating best model with epoch: {} loss: {}'.format(epoch, val_loss))
            best_model = copy.deepcopy(model)
            best_epoch = epoch

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)


    logger.info('Evaluating model against test dataset using softmax and gmm')
    eval_model(best_model, test_dataset, criterion, train_stats, "test", best_epoch, gmm, args)

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_filepath, 'model.pt'))
