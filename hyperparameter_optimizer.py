import os
import numpy as np
import argparse
import copy
import json
import matplotlib.pyplot as plt

import train

FOLDER_PATH = './models'


def search():

    n = 0
    while os.path.exists(os.path.join(FOLDER_PATH, "id-{:08}".format(n))):
        n += 1

    fp = os.path.join(FOLDER_PATH, "id-{:08}".format(n))

    cycle_factor = float(np.random.uniform(2, 5))
    if np.random.rand() > 0.5:
        cycle_factor = None
    lr_reduction_factor = float(np.random.uniform(0.1, 0.2))
    optimizer = np.random.choice(['adamw','sgd'])
    if optimizer == 'sgd':
        weight_decay = np.random.uniform(1e-6, 1e-4)
        learning_rate = float(np.random.uniform(1e-3, 0.2))
        nesterov = bool(np.random.choice([False, True]))
    else:
        weight_decay = float(np.random.uniform(1e-5, 1.0))
        learning_rate = float(np.random.uniform(1e-4, 1e-3))
        nesterov = None
    
    # from "Benchopt: Reproducible, efficient and collaborative optimization benchmarks" page 28
    # from "Lookahead optimizer: k Steps forward, 1 step back" page 17
    # from "Wide Residual Networks" code

    args = dict()
    args['arch'] = 'wide_resnet50_2'
    args['num_workers'] = 2
    args['output_filepath'] = fp
    args['batch_size'] = 128
    args['learning_rate'] = 3e-4 #learning_rate
    args['loss_eps'] = 1e-4
    args['num_lr_reductions'] = 2
    args['lr_reduction_factor'] = 0.2 #lr_reduction_factor
    args['patience'] = 50
    args['weight_decay'] = 0.5  #0.0005 #weight_decay
    args['cycle_factor'] = None #cycle_factor
    args['starting_model'] = None
    args['nesterov'] = None #nesterov
    args['optimizer'] = 'adamw' #optimizer
    args['debug'] = False
    args['amp'] = True #bool(np.random.uniform(0, 1.0) > 0.5)
    args['val_fraction'] = 0.1 #float(np.random.uniform(0.01, 0.1))

    if args['debug']:
        args['loss_eps'] = 0.1
        args['patience'] = 5

    args = argparse.Namespace(**args)

    stats = train.train(args)



def select():
    fns = [fn for fn in os.listdir(FOLDER_PATH) if fn.startswith('id-')]
    fns.sort()

    best_config = None
    best_accuracy = 0
    best_model = ""
    acc_vector = list()
    config_vector = list()
    for fn in fns:
        stats_fp = os.path.join(FOLDER_PATH, fn, 'stats.json')
        config_fp = os.path.join(FOLDER_PATH, fn, 'config.json')
        if os.path.exists(stats_fp) and os.path.exists(config_fp):
            with open(stats_fp, 'r') as fh:
                stats_dict = json.load(fh)
            with open(config_fp, 'r') as fh:
                config_dict = json.load(fh)

            if 'val_accuracy' not in stats_dict.keys():
                print("{} missing val_accuracy".format(fn))
                continue

            config_vector.append(config_dict)
            acc_vector.append(stats_dict['test_accuracy'])

            if best_config is None:
                best_config = config_dict
                best_accuracy = stats_dict['test_accuracy']
                best_model = fn
            else:
                if stats_dict['val_accuracy'] > best_accuracy:
                    best_config = config_dict
                    best_accuracy = stats_dict['test_accuracy']
                    best_model = fn

    print("Best Config (test accuracy = {}):".format(best_accuracy))
    print(best_model)
    print(best_config)



if __name__ == '__main__':
    search()
    # select()



