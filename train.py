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

MAX_EPOCHS = 1000
GMM_ENABLED = True

logger = logging.getLogger()


def setup(args):
    # load stock models from https://pytorch.org/vision/stable/models.html
    model = None
    if args.starting_model is not None:
        # warning, this over rides the args.arch selection
        model = torch.load(args.starting_model)
    else:
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        if args.arch == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False, num_classes=10)
        if args.arch == 'resnext50_32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=10)
        if args.arch == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=10)

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {}.".format(args.arch))

    # setup and load CIFAR10
    train_dataset = cifar_datasets.Cifar10(transforms=cifar_datasets.Cifar10.TRANSFORM_TRAIN, train=True, subset=args.debug)
    train_dataset, val_dataset = train_dataset.train_val_split(val_fraction=args.val_fraction)
    # set the validation augmentation to just normalize (.dataset since val_dataset is a Subset, not a full dataset)
    val_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_TEST)

    test_dataset = cifar_datasets.Cifar10(transforms=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    # # wrap the model into a single node DataParallel
    # if not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    return model, train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, criterion, scheduler, epoch, train_stats, amp=True):
    avg_loss = 0
    avg_accuracy = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    batch_count = len(dataloader)
    start_time = time.time()

    for batch_idx, tensor_dict in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = tensor_dict[0].cuda()
        labels = tensor_dict[1].cuda()

        # FP16 training
        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=-1)
                accuracy = torch.sum(pred == labels)/len(pred)
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
            pred = torch.argmax(outputs, dim=-1)
            accuracy = torch.sum(pred == labels)/len(pred)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # nan loss values are ignored when using AMP, so ignore them for the average
        if not np.isnan(loss.detach().cpu().numpy()):
            avg_loss += loss.item()
            avg_accuracy += accuracy.item()
            if scheduler is not None:
                scheduler.step()

        if batch_idx % 100 == 0:
            logger.info('  batch {}/{}  loss: {:8.8g}'.format(batch_idx, batch_count, loss.item()))

    avg_loss /= batch_count
    avg_accuracy /= batch_count
    wall_time = time.time() - start_time

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_loss)
    train_stats.add(epoch, 'train_accuracy', avg_accuracy)


def eval_model(model, dataloader, criterion, epoch, train_stats, split_name, amp=True):
    if dataloader is None or len(dataloader) == 0:
        train_stats.add(epoch, '{}_wall_time'.format(split_name), 0)
        train_stats.add(epoch, '{}_loss'.format(split_name), 0)
        train_stats.add(epoch, '{}_accuracy'.format(split_name), 0)
        return None, None, None

    batch_count = len(dataloader)
    avg_loss = 0
    avg_accuracy = 0
    start_time = time.time()
    model.eval()

    dataset_logits = list()
    dataset_labels = list()
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if GMM_ENABLED:
                        dataset_logits.append(outputs.detach().cpu())
                        dataset_labels.append(labels.detach().cpu())
                    # TODO get the second to last activations as well
                    pred = torch.argmax(outputs, dim=-1)
                    accuracy = torch.sum(pred == labels) / len(pred)
                    loss = criterion(outputs, labels)
                    avg_loss += loss.item()
                    avg_accuracy += accuracy.item()
            else:
                outputs = model(inputs)
                if GMM_ENABLED:
                    dataset_logits.append(outputs.detach().cpu())
                    dataset_labels.append(labels.detach().cpu())
                # TODO get the second to last activations as well
                pred = torch.argmax(outputs, dim=-1)
                accuracy = torch.sum(pred == labels) / len(pred)
                loss = criterion(outputs, labels)
                avg_loss += loss.item()
                avg_accuracy += accuracy.item()

    wall_time = time.time() - start_time
    avg_loss /= batch_count
    avg_accuracy /= batch_count

    train_stats.add(epoch, '{}_wall_time'.format(split_name), wall_time)
    train_stats.add(epoch, '{}_loss'.format(split_name), avg_loss)
    train_stats.add(epoch, '{}_accuracy'.format(split_name), avg_accuracy)

    bucketed_dataset_logits = list()
    unique_class_labels = list()
    if GMM_ENABLED:
        # join together the individual batches of numpy logit data
        dataset_logits = torch.cat(dataset_logits)
        dataset_labels = torch.cat(dataset_labels)
        unique_class_labels = torch.unique(dataset_labels)

        for i in range(len(unique_class_labels)):
            c = unique_class_labels[i]
            bucketed_dataset_logits.append(dataset_logits[dataset_labels == c])

    return dataset_logits, bucketed_dataset_logits, unique_class_labels


def eval_model_gmm(model, dataloader, gmm_list):

    batch_count = len(dataloader)
    model.eval()

    gmm_preds = list()
    softmax_preds = list()
    gmm_accuracy = list()
    softmax_accuracy = list()
    class_preval = compute_class_prevalance(dataloader)
    logger.info(class_preval)
    class_preval = torch.tensor(list(class_preval.values()))
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=-1)
                softmax_preds.append(pred.reshape(-1))
                accuracy = torch.sum(pred == labels) / len(pred)
                softmax_accuracy.append(accuracy.reshape(-1))

                # passing the logits acquired before the softmax to gmm as inputs
                gmm_inputs = outputs.detach().cpu()

                # generate weights, mus and sigmas
                weights = class_preval
                mus = torch.cat([gmm.get("mu") for gmm in gmm_list])
                sigmas = torch.cat([gmm.get("sigma") for gmm in gmm_list])

                # create new GMM object for combined data
                gmm = GMM(n_features=gmm_inputs.shape[1], n_clusters=weights.shape[0], weights=weights, mu=mus, sigma=sigmas)

                gmm_resp = gmm.predict_probability(gmm_inputs)  # N*K

                gmm_pred = torch.argmax(gmm_resp, dim=-1)
                gmm_preds.append(gmm_pred.reshape(-1))
                labels_cpu = labels.detach().cpu()
                accuracy_g = torch.sum(gmm_pred == labels_cpu) / len(gmm_pred)
                gmm_accuracy.append(accuracy_g.reshape(-1))

        gmm_accuracy = torch.mean(torch.cat(gmm_accuracy, dim=-1))
        softmax_accuracy = torch.mean(torch.cat(softmax_accuracy, dim=-1))
        gmm_preds = torch.mean(torch.cat(gmm_preds, dim=-1).type(torch.float16))
        softmax_preds = torch.mean(torch.cat(softmax_preds, dim=-1).type(torch.float16))

        return softmax_preds, softmax_accuracy, gmm_preds, gmm_accuracy


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


def train(args):
    if not os.path.exists(args.output_filepath):
        os.makedirs(args.output_filepath)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(args.output_filepath, 'log.txt'))

    logging.getLogger().addHandler(logging.StreamHandler())

    logger.info(args)

    model, train_loader, val_loader, test_loader = setup(args)

    # write the arg configuration to disk
    dvals = vars(args)
    with open(os.path.join(args.output_filepath, 'config.json'), 'w') as fh:
        json.dump(dvals, fh, ensure_ascii=True, indent=2)

    train_start_time = time.time()

    # Move model to device
    model.cuda()

    def lr_reduction_callback():
        logger.info("Learning rate reduced due to plateau")

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()

    # Setup optimizer
    if args.weight_decay is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # setup LR reduction on plateau
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions, lr_reduction_callback=lr_reduction_callback)

    if args.cycle_factor is not None:
        cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate/args.cycle_factor, max_lr=args.learning_rate*args.cycle_factor, step_size_up=int(len(train_loader) / 2), cycle_momentum=False)
    else:
        cyclic_scheduler = None

    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    epoch = -1
    best_epoch = 0
    best_model = model

    # train epochs until loss converges
    while not plateau_scheduler.is_done():
        # ensure we don't loop forever
        if epoch >= MAX_EPOCHS:
            break
        epoch += 1
        logger.info("Epoch: {}".format(epoch))
        logger.info("  training")
        # TODO capture and return the full training dataset output layer (pre-softmax) and the labels
        train_epoch(model, train_loader, optimizer, criterion, cyclic_scheduler, epoch, train_stats, args.amp)

        # TODO write function which buckets the output vectors by their true class label (not predicted label)

        logger.info("  evaluating validation data")
        dataset_logits, class_bucketed_dataset_logits, unique_class_labels = eval_model(model, val_loader, criterion, epoch, train_stats, 'val', args.amp)

        val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
        plateau_scheduler.step(val_loss)
        
        # dataset_logits is N x num_classes, where N is the number of examples in the dataset

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_wall_time', sum(train_stats.get('val_wall_time')))

        # TODO transfer this whole process into gmm or new class where these things can be handled by a single class and can be parallelized
        if GMM_ENABLED:  # and epoch > 10:
            logger.info(" gmm work starts")
            gmm_models = list()
            class_instances = list()
            for i in range(len(unique_class_labels)):
                class_c_logits = class_bucketed_dataset_logits[i]
                class_instances.append(class_c_logits.shape[0])
                start_time = time.time()
                gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
                gmm.fit(class_c_logits)
                while np.any(np.isnan(gmm.get("sigma").detach().cpu().numpy())):
                    gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
                    gmm.fit(class_c_logits)
                # else:
                #     gmm.logger.export()
                elapsed_time = time.time() - start_time
                logger.info("Build GMM took: {}s".format(elapsed_time))
                gmm_models.append(gmm)
                train_stats.add(epoch, 'class_{}_gmm_log_likelihood'.format(unique_class_labels[i]), gmm.log_likelihood.detach().cpu().item())

            logger.info(unique_class_labels)
            softmax_preds, softmax_accuracy, gmm_preds, gmm_accuracy = eval_model_gmm(model, val_loader, gmm_models)
            logger.info("Softmax Accuracy: {}".format(softmax_accuracy))
            logger.info("GMM Accuracy: {}".format(gmm_accuracy))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(args.output_filepath)


        # handle early stopping when loss converges
        val_loss = train_stats.get('val_loss')
        # error_from_best = np.abs(val_loss - np.min(val_loss))
        # error_from_best[error_from_best < np.abs(args.loss_eps)] = 0
        
        # if this epoch is with convergence tolerance of the global best, save the weights
        #if error_from_best[epoch] == 0:
        if plateau_scheduler.num_bad_epochs == 0:
            logger.info('Updating best model with epoch: {} loss: {}'.format(epoch, val_loss[epoch]))
            best_model = copy.deepcopy(model)
            best_epoch = epoch

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)


    logger.info('Evaluating model against test dataset')
    eval_model(best_model, test_loader, criterion, best_epoch, train_stats, 'test', args.amp)

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_filepath, 'model.pt'))
    return train_stats