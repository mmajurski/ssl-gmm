import os
import time
import copy
import numpy as np
import torch
import torchvision
import json

import dataset
import metadata
import mmm

MAX_EPOCHS = 1000
GMM_ENABLED = True


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

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {}.".format(args.arch))

    # setup and load CIFAR10
    train_loader, val_loader, test_loader = dataset.get_cifar10(args, subset=args.debug)

    # wrap the model into a single node DataParallel
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    return model, train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, criterion, epoch, train_stats):
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

        # nan loss values are ignored when using AMP, so ignore them for the average
        if not np.isnan(loss.detach().cpu().numpy()):
            avg_loss += loss.item()
            avg_accuracy += accuracy.item()

        if batch_idx % 100 == 0:
            print('  batch {}/{}  loss: {:8.8g}'.format(batch_idx, batch_count, loss.item()))

    avg_loss /= batch_count
    avg_accuracy /= batch_count
    wall_time = time.time() - start_time

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_loss)
    train_stats.add(epoch, 'train_accuracy', avg_accuracy)


def eval_model(model, dataloader, criterion, epoch, train_stats, split_name):
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
    print(class_preval)
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

                # create mu_list and cov_list
                mu_list = torch.cat([i.mu for i in gmm_list])
                cov_list = torch.cat([i.cov for i in gmm_list])

                # using single object and list of mu and cov to do the computations faster as the only class variable
                # used is dimension which will be same for all gmms making it safe to do so
                gmm_probs = gmm_list[0].predict_probs(gmm_inputs, mu_list, cov_list)  # N*K
                gmm_probs_sum = torch.sum(gmm_probs, 1)  # N*1


                # TODO (JD) confirm the sequence of k is same for gmm_probs and class_preval
                # (one is done with np.unique one with torch.unique)

                numerator = gmm_probs * class_preval
                gmm_outputs_t = torch.transpose(numerator,0,1) / gmm_probs_sum
                gmm_outputs = torch.transpose(gmm_outputs_t,0,1)

                gmm_pred = torch.argmax(gmm_outputs, dim=-1)
                gmm_preds.append(gmm_pred.reshape(-1))
                labels_cpu = labels.detach().cpu()
                accuracy_g = torch.sum(gmm_pred == labels_cpu) / len(gmm_pred)
                gmm_accuracy.append(accuracy_g.reshape(-1))

        gmm_accuracy = torch.mean(torch.cat(gmm_accuracy, dim=-1))
        softmax_accuracy = torch.mean(torch.cat(softmax_accuracy, dim=-1))
        gmm_preds = torch.mean(torch.cat(gmm_preds, dim=-1))
        softmax_preds = torch.mean(torch.cat(softmax_preds, dim=-1))

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

    model, train_loader, val_loader, test_loader = setup(args)

    # write the arg configuration to disk
    dvals = vars(args)
    with open(os.path.join(args.output_filepath, 'config.json'), 'w') as fh:
        json.dump(dvals, fh, ensure_ascii=True, indent=2)

    train_start_time = time.time()

    # Move model to device
    model.cuda()

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)#, weight_decay=5e-4)

    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    epoch = 0
    best_epoch = 0
    done = False
    best_model = model

    num_epochs = args.num_epochs

    # train epochs until loss converges
    while not done:
        print("Epoch: {}".format(epoch))
        print("  training")
        # TODO capture and return the full training dataset output layer (pre-softmax) and the labels
        train_epoch(model, train_loader, optimizer, criterion, epoch, train_stats)

        # TODO write function which buckets the output vectors by their true class label (not predicted label)

        print("  evaluating validation data")
        dataset_logits, class_bucketed_dataset_logits, unique_class_labels = eval_model(model, val_loader, criterion, epoch, train_stats, 'val')
        # dataset_logits is N x num_classes, where N is the number of examples in the dataset

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_wall_time', sum(train_stats.get('val_wall_time')))

        if GMM_ENABLED:  # and epoch > 10:
            print(" gmm work starts")
            gmm_models = list()
            for i in range(len(unique_class_labels)):
                class_c_logits = class_bucketed_dataset_logits[i]
                start_time = time.time()
                gmm = mmm.GMM(class_c_logits, num_clusters=1, tolerance=1e-4, num_iterations=50)
                gmm.convergence()
                while np.any(np.isnan(gmm.cov.detach().cpu().numpy())):
                    gmm = mmm.GMM(class_c_logits, num_clusters=1, tolerance=1e-4, num_iterations=50)
                    gmm.convergence()
                else:
                    gmm.logger.export()
                elapsed_time = time.time() - start_time
                print("Build GMM took: {}s".format(elapsed_time))
                gmm_models.append(gmm)
                train_stats.add(epoch, 'class_{}_gmm_log_likelihood'.format(unique_class_labels[i]), gmm.ll_prev)

            print(unique_class_labels)
            softmax_preds, softmax_accuracy, gmm_preds, gmm_accuracy = eval_model_gmm(model, val_loader, gmm_models)
            print("Softmax Accuracy: {}".format(softmax_accuracy))
            print("GMM Accuracy: {}".format(gmm_accuracy))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(args.output_filepath)

        if num_epochs is not None:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            # update the global metrics with the best epoch
            train_stats.update_global(epoch)
        else:
            # handle early stopping when loss converges
            val_loss = train_stats.get('val_loss')
            error_from_best = np.abs(val_loss - np.min(val_loss))
            error_from_best[error_from_best < np.abs(args.loss_eps)] = 0
            # if this epoch is with convergence tolerance of the global best, save the weights
            if error_from_best[epoch] == 0:
                print('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'.format(epoch, val_loss[epoch], args.loss_eps))
                best_model = copy.deepcopy(model)
                best_epoch = epoch

                # update the global metrics with the best epoch
                train_stats.update_global(epoch)
            best_val_loss_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
            if epoch >= (best_val_loss_epoch + args.early_stopping_epoch_count):
                print("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
                done = True

        if not done:
            # only advance epoch if we are not done
            epoch += 1
        # in case something goes wrong, we exit after training a long time ...
        if num_epochs is not None and epoch >= num_epochs:
            done = True
        if epoch >= MAX_EPOCHS:
            done = True

    print('Evaluating model against test dataset')
    eval_model(best_model, test_loader, criterion, best_epoch, train_stats, 'test')

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    print("Total WallTime: ", train_stats.get_global('wall_time'), 'seconds')

    train_stats.export(args.output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(args.output_filepath, 'model.pt'))
    return train_stats