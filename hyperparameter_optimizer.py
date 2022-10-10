import os
import numpy as np
import argparse
import copy
import json
import matplotlib.pyplot as plt

import train_ssl

FOLDER_PATH = './models'


def search():

    n = 0
    while os.path.exists(os.path.join(FOLDER_PATH, "id-{:08}".format(n))):
        n += 1

    fp = os.path.join(FOLDER_PATH, "id-{:08}".format(n))

    # cycle_factor = float(np.random.uniform(2, 5))
    # if np.random.rand() > 0.5:
    #     cycle_factor = None
    # optimizer = np.random.choice(['adamw','sgd'])
    # if optimizer == 'sgd':
    #     weight_decay = np.random.uniform(1e-6, 1e-4)
    #     learning_rate = float(np.random.uniform(1e-3, 0.2))
    #     nesterov = True
    # else:
    #     weight_decay = float(np.random.uniform(1e-5, 1.0))
    #     learning_rate = float(np.random.uniform(1e-4, 1e-3))
    #     nesterov = None
    
    # from "Benchopt: Reproducible, efficient and collaborative optimization benchmarks" page 28
    # from "Lookahead optimizer: k Steps forward, 1 step back" page 17
    # from "Wide Residual Networks" code

    arch = np.random.choice(['wide_resnet','resnet18','resnet34','resnext50_32x4d'])

    args = dict()
    args['arch'] = arch #'wide_resnet'
    args['num_workers'] = 2
    args['output_filepath'] = fp
    args['batch_size'] = 128
    args['learning_rate'] = 3e-4 #learning_rate
    args['loss_eps'] = 1e-4
    args['num_lr_reductions'] = 2
    args['lr_reduction_factor'] = 0.2
    args['patience'] = 50
    args['weight_decay'] = 1e-5  #weight_decay
    args['cycle_factor'] = 4 #cycle_factor
    args['starting_model'] = None
    args['nesterov'] = None #nesterov
    args['optimizer'] = 'adamw'  #optimizer
    args['debug'] = False
    args['num_classes'] = 10
    args['amp'] = True
    args['val_fraction'] = 0.1
    # args['num_labeled_datapoints'] = 250

    if args['debug']:
        args['loss_eps'] = 0.1
        args['patience'] = 5

    args = argparse.Namespace(**args)

    train_ssl.train(args)



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

    n = 1

    fp = os.path.join(FOLDER_PATH, "id-{:08}".format(n))

    args = dict()
    args['arch'] = 'wide_resnet'
    args['num_workers'] = 4
    args['output_filepath'] = fp
    args['batch_size'] = 128
    args['learning_rate'] = 3e-4
    args['loss_eps'] = 1e-4
    args['num_lr_reductions'] = 2
    args['lr_reduction_factor'] = 0.2
    args['patience'] = 50
    args['weight_decay'] = 1e-5
    args['cycle_factor'] = 4
    args['starting_model'] = None
    args['nesterov'] = None
    args['optimizer'] = 'adamw'
    args['debug'] = False
    args['num_classes'] = 10
    args['disable_amp'] = False
    args['val_fraction'] = 0.1
    args['num_labeled_datapoints'] = 250
    args['re_pseudo_label_each_epoch'] = False
    args['disable_ssl'] = False
    args['pseudo_label_percentile_threshold'] = 0.95
    args['inference_method'] = 'gmm'
    args['cluster_per_class'] = 1


    if args['debug']:
        args['loss_eps'] = 0.1
        args['patience'] = 5

    # train_ssl.train(argparse.Namespace(**args))

    n = 2
    fp = os.path.join(FOLDER_PATH, "id-{:08}".format(n))
    args['output_filepath'] = fp
    args['inference_method'] = 'cauchy'

    train_ssl.train(argparse.Namespace(**args))

    #search()
    #select()



