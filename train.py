import os
import time
import copy
import numpy as np
import torch
import torchvision
import json

import dataset
import metadata

MAX_EPOCHS = 1000


def setup(args):
    # load stock models from https://pytorch.org/vision/stable/models.html
    model = None
    if args.arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if args.arch == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False, num_classes=10)
    if args.arch == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=10)

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {}.".format(args.arch))

    # setup and load CIFAR10
    train_loader, val_loader, test_loader = dataset.get_cifar10(args)

    # wrap the model into a single node DataParallel
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
            # TODO get the second to last activations as well
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
        return

    batch_count = len(dataloader)
    avg_loss = 0
    avg_accuracy = 0
    start_time = time.time()
    model.eval()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
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


def train(args):
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

    # train epochs until loss converges
    while not done:
        print("Epoch: {}".format(epoch))
        print("  training")
        # TODO capture and return the full training dataset output layer (pre-softmax) and the labels
        train_epoch(model, train_loader, optimizer, criterion, epoch, train_stats)

        # TODO write function which buckets the output vectors by their true class label (not predicted label)

        # TODO GMM goes here

        print("  evaluating test data")
        eval_model(model, val_loader, criterion, epoch, train_stats, 'val')

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_wall_time', sum(train_stats.get('val_wall_time')))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(args.output_filepath)

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