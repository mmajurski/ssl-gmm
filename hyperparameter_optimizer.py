import os
import numpy as np
import argparse
import copy

import train


def find_starting_model(ofp):
    start_n = 0
    fn = "id-{:08}".format(start_n)
    fp = os.path.join(ofp, fn)
    while os.path.exists(fp):
        start_n += 1
        fn = "id-{:08}".format(start_n)
        fp = os.path.join(ofp, fn)
    return start_n


N = 500
best_model_stats = None
ofp = './models'

start_n = find_starting_model(ofp)

for n in range(start_n, N):
    fn = "id-{:08}".format(n)
    fp = os.path.join(ofp, fn)

    epoch = np.random.randint(100, 500)
    loss_eps = None
    early_stopping_epoch_count = None
    # handle early stopping as a 1 in 10 chance
    if np.random.randint(5) <= 1:
        epoch = None
        loss_eps = np.random.uniform(1e-6, 1e-1)
        early_stopping_epoch_count = np.random.randint(2, 31)

    batch_size = int(np.random.choice([8, 16, 32, 64, 128, 256]))
    learning_rate = np.random.uniform(1e-6, 1e-2)


    args = dict()
    args['arch'] = 'resnet18'
    args['num_workers'] = 6
    args['output_filepath'] = fp

    args['batch_size'] = batch_size
    args['learning_rate'] = learning_rate
    args['loss_eps'] = loss_eps
    args['num_epochs'] = epoch
    args['early_stopping_epoch_count'] = early_stopping_epoch_count
    args['starting_model'] = None
    args['debug'] = False

    args = argparse.Namespace(**args)

    stats = train.train(args)

    if best_model_stats is None:
        best_model_stats = stats
    else:
        if stats['test_accuracy'] > best_model_stats['test_accuracy']:
            best_model_stats = copy.deepcopy(stats)
            print("New Best Model Found:")
            print(args)
            print(best_model_stats)

            stats.add_global('id', fn)
            stats.export(os.path.join(ofp, 'best_stats.json'))