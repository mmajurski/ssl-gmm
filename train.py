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
import train_ssl

MAX_EPOCHS = 300
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
            model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
            #model = flavored_resnet18.ResNet18(num_classes=args.num_classes)
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

    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    return model, train_dataset, val_dataset, test_dataset


# TODO, setup data augmentation and disqualify samples from being added to the labeled population based on contrastive learning. If the labeled changes under N instances of data augmentation, then the model is not confident about it. a la "Unsupervised Data Augmentation for Consistency Training"

# TODO Test greedy pseudo-labeling, where for each call to this function, we take the top k=2 pseudo-labeled from each class. (how to determine the best samples, look at resp)

def train(args):
    if not os.path.exists(args.output_filepath):
        os.makedirs(args.output_filepath)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(args.output_filepath, 'log.txt'))

    logging.getLogger().addHandler(logging.StreamHandler())

    logger.info(args)

    model, train_dataset_labeled, val_dataset, test_dataset = setup(args)

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
    gmm = None

    # *************************************
    # ******** Supervised Training ********
    # *************************************
    # train the model until it has converged on the the labeled data
    # setup early stopping on convergence using LR reduction on plateau
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions)

    # train epochs until loss converges
    while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
        epoch += 1
        logger.info("Epoch (supervised): {}".format(epoch))

        logger.info("  training against fully labeled dataset.")
        train_ssl.train_epoch(model, train_dataset_labeled, optimizer, criterion, epoch, train_stats, args)

        logger.info("  building the GMM models on the labeled training data.")
        gmm = train_ssl.build_gmm(model, train_dataset_labeled, epoch, train_stats, args)

        logger.info("  evaluating against validation data, using softmax and gmm")
        train_ssl.eval_model(model, val_dataset, criterion, train_stats, "val", epoch, gmm, args)

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
    train_ssl.eval_model(best_model, test_dataset, criterion, train_stats, "test", best_epoch, gmm, args)

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_filepath, 'model.pt'))
