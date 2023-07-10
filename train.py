import os
import time
import copy
import numpy as np
import torch
import torchvision
import json
import logging
import shutil

import cifar_datasets
import metadata
import lr_scheduler
import flavored_wideresnet
import utils
import trainer
import trainer_fixmatch



def setup(args):
    # model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)

    if args.arch == 'wide_resnet':
        model = flavored_wideresnet.WideResNetMajurski(num_classes=args.num_classes, last_layer=args.last_layer, embedding_dim=args.embedding_dim, num_pre_fc=args.nprefc, use_tanh=args.use_tanh)
    elif args.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
    elif args.arch == 'flavored_wide_resnet':
        model = flavored_wideresnet.WideResNet(num_classes=args.num_classes)
    else:
        raise RuntimeError("invalid model arch type")

    # # load stock models from https://pytorch.org/vision/stable/models.html
    # model = None
    #
    # if args.starting_model is not None:
    #     # warning, this over rides the args.arch selection
    #     logging.info("Loading requested starting model from '{}'".format(args.starting_model))
    #     model = torch.load(args.starting_model)
    # else:
    #     if args.last_layer == 'fc':
    #         if args.arch == 'resnet18':
    #             model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
    #             # model = flavored_resnets.ResNet18(num_classes=args.num_classes)
    #         if args.arch == 'resnet34':
    #             model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)
    #         if args.arch == 'resnext50_32x4d':
    #             model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=args.num_classes)
    #         if args.arch == 'wide_resnet50_2':
    #             model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=args.num_classes)
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)
    #
    #     elif args.last_layer == 'gmm':
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)
    #     elif args.last_layer == 'cauchy':
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)
    #     elif args.last_layer == 'aa_gmm':
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)
    #     elif args.last_layer == 'aa_gmm_d1':
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)
    #     elif args.last_layer == 'kmeans_cmm':
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)
    #     elif args.last_layer == 'kmeans_distribution':
    #         if args.arch == 'wide_resnet':
    #             model = flavored_wideresnet.WideResNet(num_classes=args.num_classes, last_layer=args.last_layer)

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {} with last layer: {}.".format(args.arch, args.last_layer))

    logging.info("Total Model params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # setup and load CIFAR10
    if args.num_classes == 10:
        if args.strong_augmentation:
            train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_STRONG_TRAIN, train=True, subset=args.debug)
        else:
            train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_WEAK_TRAIN, train=True, subset=args.debug)
    else:
        raise RuntimeError("unsupported CIFAR class count: {}".format(args.num_classes))

    val_dataset = None
    if args.num_labeled_datapoints > 0:
        train_dataset_labeled, train_dataset_unlabeled = train_dataset.data_split_class_balanced(subset_count=args.num_labeled_datapoints)
    else:
        train_dataset_labeled = train_dataset
        train_dataset_unlabeled = None

    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    # adjust the len of the dataset to implement the nb_reps
    train_dataset_labeled.set_epoch_size(args.epoch_size * args.batch_size)
    if train_dataset_unlabeled is not None:
        train_dataset_unlabeled.set_epoch_size(args.mu * args.epoch_size * args.batch_size)

    return model, train_dataset_labeled, train_dataset_unlabeled, test_dataset


def train(args):
    if not os.path.exists(args.output_dirpath):
        # safety check that the output directory exists
        os.makedirs(args.output_dirpath)

    # add the file based handler to the logger
    fh = logging.FileHandler(filename=os.path.join(args.output_dirpath, 'log.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    try:
        # attempt to get the slurm job id and log it
        logging.info("Slurm JobId: {}".format(os.environ['SLURM_JOB_ID']))
    except KeyError:
        pass

    try:
        # attempt to get the hostname and log it
        import socket
        hn = socket.gethostname()
        logging.info("Job running on host: {}".format(hn))
    except RuntimeError:
        pass

    # copy code files to the output directory
    src_ofp = os.path.join(args.output_dirpath, 'src')
    os.makedirs(src_ofp)
    src_fns = [fn for fn in os.listdir('./') if fn.endswith('.py')]
    for fn in src_fns:
        shutil.copyfile(fn, os.path.join(src_ofp, fn))

    model, train_dataset_labeled, train_dataset_unlabeled, test_dataset = setup(args)

    # write the args configuration to disk
    logging.info("writing args to config.json")
    with open(os.path.join(args.output_dirpath, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, ensure_ascii=True, indent=2)

    train_start_time = time.time()

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()
    logging.info("criterion setup")

    # setup the trainer
    # supervised, fixmatch, fixmatch-gmm
    if args.trainer == 'supervised':
        model_trainer = trainer.SupervisedTrainer(args)
    elif args.trainer == 'fixmatch':
        model_trainer = trainer_fixmatch.FixMatchTrainer(args)
    else:
        raise RuntimeError("Invalid trainer request: {}".format(args.trainer))
    logging.info("trainer setup")

    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    epoch = -1
    best_epoch = 0
    best_model = model

    # Move model to device
    model.cuda()

    # setup early stopping on convergence using LR reduction on plateau
    optimizer = model_trainer.configure_optimizer(model)

    def log_lr_reduction():
        logging.info("Learning rate reduced")

    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions, lr_reduction_callback=log_lr_reduction)
    # train epochs until loss converges

    if args.use_ema:
        from model_ema import ModelEMA
        # model needs to already be on device
        ema_model = ModelEMA(model, args.ema_decay)
    else:
        ema_model = None

    if args.num_epochs is not None:
        trainer.MAX_EPOCHS = args.num_epochs

    while not plateau_scheduler.is_done() and epoch <= trainer.MAX_EPOCHS:
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)
        model_trainer.train_epoch(model, train_dataset_labeled, optimizer, criterion, epoch, train_stats, unlabeled_dataset=train_dataset_unlabeled, ema_model=ema_model)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        logging.info("  evaluating against test data")
        model_trainer.eval_model(test_model, test_dataset, criterion, train_stats, "test", epoch, args)

        test_accuracy = train_stats.get_epoch('test_accuracy', epoch=epoch)
        plateau_scheduler.step(test_accuracy)
        train_stats.add_global('test_wall_time', train_stats.get('test_wall_time', aggregator='sum'))

        # update global metadata stats
        train_stats.add_global('training_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
        train_stats.add_global('num_epochs_trained', epoch)

        # write copy of current metadata metrics to disk
        train_stats.export(args.output_dirpath)

        # handle early stopping when loss converges
        if args.num_epochs is not None or plateau_scheduler.is_equiv_to_best_epoch:
            logging.info('Updating best model with epoch: {}'.format(epoch))
            if args.use_ema:
                best_model = copy.deepcopy(ema_model.ema)
            else:
                best_model = copy.deepcopy(model)

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logging.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_dirpath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_dirpath, 'model.pt'))

    return train_stats