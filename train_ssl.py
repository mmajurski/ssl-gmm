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
import flavored_resnets

MAX_EPOCHS = 1000
GMM_ENABLED = True  # always true for ssl

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
            #model = flavored_resnets.ResNet18(num_classes=10)
        if args.arch == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False, num_classes=10)
        if args.arch == 'resnext50_32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=10)
        if args.arch == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=10)

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {}.".format(args.arch))

    # setup and load CIFAR10
    train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TRAIN, train=True, subset=args.debug)

    train_dataset, val_dataset = train_dataset.train_val_split(val_fraction=args.val_fraction)
    # set the validation augmentation to just normalize (.dataset since val_dataset is a Subset, not a full dataset)
    val_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_TEST)

    train_dataset_labeled, train_dataset_unlabeled = train_dataset.data_split_class_balanced(subset_count=4000)

    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    return model, train_dataset_labeled, train_dataset_unlabeled, val_dataset, test_dataset


def psuedolabel_data(model,train_dataset_labeled, train_dataset_unlabeled, combined_gmm, args):

    ul_dataloader = torch.utils.data.DataLoader(train_dataset_unlabeled,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                worker_init_fn=cifar_datasets.worker_init_fn)
    # TODO update this to use a per-class threshold, but that is an extension of the baseline technique
    denominator_threshold = 500.0  # TODO update this to be 80 % confident
    resp_threshold = 0.95
    filtered_inputs = []
    filtered_labels = []
    filtered_data_resp = []
    model.eval()
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(ul_dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                gmm_inputs = outputs.detach().cpu()
                gmm_prob_sum, gmm_resp = combined_gmm.predict_probability(gmm_inputs)

                # threshold filtering

                # list of boolean where value is False for items having lower prob_sum then threshold, True otherwise
                denominator_filter = torch.squeeze(gmm_prob_sum > denominator_threshold)

                filtered_inputs.append(inputs[denominator_filter])
                filtered_labels.append(labels[denominator_filter])
                filtered_data_resp.append(gmm_resp[denominator_filter])

        filtered_inputs = torch.cat(filtered_inputs,dim=0)
        filtered_labels = torch.cat(filtered_labels,dim= -1)
        filtered_data_resp = torch.cat(filtered_data_resp,dim =0)

        max_resps, filtered_preds = torch.max(filtered_data_resp,dim=-1)

        resp_filter = max_resps > resp_threshold #TODO: change this filter

        filtered_inputs = filtered_inputs[resp_filter]
        filtered_labels = filtered_labels[resp_filter]
        filtered_preds = filtered_preds[resp_filter]

    num_additions = len(filtered_preds)
    # print(num_additions)
    for idx in range(num_additions):
        train_dataset_labeled.add_datapoint(filtered_inputs[idx].cpu().numpy(),filtered_preds[idx].cpu().item())
        # if not working then use permute(1,2,0)

    #testing psuedolabel accuracy
    psuedolabel_acc = torch.sum(filtered_preds.cpu() == filtered_labels.cpu()) / len(filtered_preds.cpu())

    # print(psuedolabel_acc) # may add to trainstats

    return train_dataset_labeled, train_dataset_unlabeled

#TODO:[Rushabh/JD] [Priority-LOW] transfer combining GMM at the end of train epoch from eval_model_gmm
def train_epoch(model, pt_dataset, optimizer, criterion, scheduler, epoch, train_stats, args, gmm_models=[]):

    avg_loss = 0
    avg_accuracy = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

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

        if GMM_ENABLED:
            dataset_logits.append(outputs.detach().cpu())
            dataset_labels.append(labels.detach().cpu())

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

    if GMM_ENABLED:
        bucketed_dataset_logits = list()
        # join together the individual batches of numpy logit data
        dataset_logits = torch.cat(dataset_logits)
        dataset_labels = torch.cat(dataset_labels)
        unique_class_labels = torch.unique(dataset_labels)

        for i in range(len(unique_class_labels)):
            c = unique_class_labels[i]
            bucketed_dataset_logits.append(dataset_logits[dataset_labels == c])

        # logger.info(" gmm work starts")

        class_instances = list()
        for i in range(len(unique_class_labels)):
            class_c_logits = bucketed_dataset_logits[i]
            class_instances.append(class_c_logits.shape[0])
            # start_time = time.time()
            gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
            gmm.fit(class_c_logits)
            while np.any(np.isnan(gmm.get("sigma").detach().cpu().numpy())):
                gmm = GMM(n_features=class_c_logits.shape[1], n_clusters=1, tolerance=1e-4, max_iter=50)
                gmm.fit(class_c_logits)
            gmm_models.append(gmm)
            # train_stats.add(epoch, 'class_{}_gmm_log_likelihood'.format(unique_class_labels[i]),
            #                 gmm.log_likelihood.detach().cpu().item())

    return gmm_models



def eval_model(model, pt_dataset, criterion, epoch, train_stats, split_name, args):
    if pt_dataset is None or len(pt_dataset) == 0:
        # train_stats.add(epoch, '{}_wall_time'.format(split_name), 0)
        # train_stats.add(epoch, '{}_loss'.format(split_name), 0)
        # train_stats.add(epoch, '{}_accuracy'.format(split_name), 0)
        return

    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    batch_count = len(dataloader)
    avg_loss = 0
    avg_accuracy = 0
    start_time = time.time()
    model.eval()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            if args.amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

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


def eval_model_gmm(model, pt_dataset, gmm_list, args):

    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=cifar_datasets.worker_init_fn)

    batch_count = len(dataloader)
    model.eval()

    gmm_preds = list()
    softmax_preds = list()
    gmm_accuracy = list()
    softmax_accuracy = list()
    class_preval = compute_class_prevalance(dataloader)
    logger.info(class_preval)
    class_preval = torch.tensor(list(class_preval.values()))

    # generate weights, mus and sigmas
    weights = class_preval
    mus = torch.cat([gmm.get("mu") for gmm in gmm_list])
    sigmas = torch.cat([gmm.get("sigma") for gmm in gmm_list])
    gmm = None

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

                # # generate weights, mus and sigmas
                # weights = class_preval
                # mus = torch.cat([gmm.get("mu") for gmm in gmm_list])
                # sigmas = torch.cat([gmm.get("sigma") for gmm in gmm_list])
                #
                # # create new GMM object for combined data
                # gmm = GMM(n_features=gmm_inputs.shape[1], n_clusters=weights.shape[0], weights=weights, mu=mus, sigma=sigmas)
                gmm = GMM(n_features=gmm_inputs.shape[1], n_clusters=weights.shape[0], weights=weights, mu=mus, sigma=sigmas)

                _, gmm_resp = gmm.predict_probability(gmm_inputs)  # N*1, N*K

                gmm_pred = torch.argmax(gmm_resp, dim=-1)
                gmm_preds.append(gmm_pred.reshape(-1))
                labels_cpu = labels.detach().cpu()
                accuracy_g = torch.sum(gmm_pred == labels_cpu) / len(gmm_pred)
                gmm_accuracy.append(accuracy_g.reshape(-1))

        gmm_accuracy = torch.mean(torch.cat(gmm_accuracy, dim=-1))
        softmax_accuracy = torch.mean(torch.cat(softmax_accuracy, dim=-1))
        gmm_preds = torch.mean(torch.cat(gmm_preds, dim=-1).type(torch.float16))
        softmax_preds = torch.mean(torch.cat(softmax_preds, dim=-1).type(torch.float16))

        return softmax_preds, softmax_accuracy, gmm_preds, gmm_accuracy, gmm


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


    # setup LR reduction on plateau
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions)

    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    epoch = -1
    best_epoch = 0
    best_model = model
    gmm_models = []

    # train epochs until loss converges
    while not plateau_scheduler.is_done():
        # ensure we don't loop forever
        if epoch >= MAX_EPOCHS:
            break

        # TODO test re-psuedo labeling every epoch, instead of it being a one way function

        # TODO add in cycle factor... defining the number of batches somehow
        cyclic_scheduler = None
        # if args.cycle_factor is not None:
        #     num_batches = len(train_dataset) / args.batch_size
        #     cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=plateau_scheduler._last_lr[0] / args.cycle_factor, max_lr=args.learning_rate * plateau_scheduler._last_lr[0], step_size_up=int(num_batches / 2), cycle_momentum=False)
        # else:
        #     cyclic_scheduler = None

        epoch += 1
        logger.info("Epoch: {}".format(epoch))
        logger.info("  training")


        if epoch > 0 and GMM_ENABLED:
            softmax_preds, softmax_accuracy, gmm_preds, gmm_accuracy, combined_gmm = eval_model_gmm(model, val_dataset, gmm_models, args)

            train_stats.add(epoch, "softmax_val_accuracy", softmax_accuracy.detach().cpu().item())
            train_stats.add(epoch, "gmm_val_accuracy", gmm_accuracy.detach().cpu().item())

            # perform pseudo labeling
            train_dataset_labeled, train_dataset_unlabeled = psuedolabel_data(model,train_dataset_labeled, train_dataset_unlabeled, combined_gmm, args)



        gmm_models = list()
        gmm_models = train_epoch(model, train_dataset_labeled, optimizer, criterion, cyclic_scheduler, epoch, train_stats, args, gmm_models)

        logger.info("  evaluating validation data")
        eval_model(model, val_dataset, criterion, epoch, train_stats, 'val', args)

        val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
        plateau_scheduler.step(val_loss)

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_wall_time', sum(train_stats.get('val_wall_time')))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(args.output_filepath)

        # handle early stopping when loss converges
        val_loss = train_stats.get('val_loss')
        if plateau_scheduler.num_bad_epochs == 0:
            logger.info('Updating best model with epoch: {} loss: {}'.format(epoch, val_loss[epoch]))
            best_model = copy.deepcopy(model)
            best_epoch = epoch

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)







    # TODO insert cifar100 pseudo-labeling

    # train_dataset = psuedolabel_data(train_dataset,gmm_models,args)
    # train_dataset_labeled, train_dataset_unlabeled = psuedolabel_data(train_dataset_labeled, train_dataset_unlabeled, gmm_models, args)

    # TODO make a second train function to do the SSL work in

    logger.info('Evaluating model against test dataset')
    eval_model(best_model, test_dataset, criterion, best_epoch, train_stats, 'test', args)

    # GMM on Test dataset on after SSL
    if GMM_ENABLED:  # and epoch > 10:
        softmax_preds, softmax_accuracy, gmm_preds, gmm_accuracy,_ = eval_model_gmm(model, test_dataset, gmm_models, args)

        train_stats.add(epoch, "ssl_softmax_test_accuracy", softmax_accuracy.detach().cpu().item())
        train_stats.add(epoch, "ssl_gmm_test_accuracy", gmm_accuracy.detach().cpu().item())

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_filepath, 'model.pt'))
    return train_stats