import os
import numpy as np
import argparse
import copy
import json

import train

FOLDER_PATH = './models'


def search():

    n = 0
    fn = "id-{:08}".format(n)
    fp = os.path.join(FOLDER_PATH, fn)
    while os.path.exists(fp):
        n += 1

    fn = "id-{:08}".format(n)
    fp = os.path.join(FOLDER_PATH, fn)

    loss_eps = np.random.uniform(1e-4, 1e-2)
    patience = np.random.randint(10, 31)

    batch_size = int(np.random.choice([16, 32, 64, 128]))
    learning_rate = np.random.uniform(1e-4, 1e-2)

    args = dict()
    args['arch'] = 'resnet18'
    args['num_workers'] = 6
    args['output_filepath'] = fp
    args['batch_size'] = batch_size
    args['learning_rate'] = learning_rate
    args['loss_eps'] = loss_eps
    args['patience'] = patience
    args['cycle_factor'] = float(np.random.choice([2, 3, 4, 5]))
    args['starting_model'] = None
    args['debug'] = False
    args['val_fraction'] = float(np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]))

    args = argparse.Namespace(**args)

    stats = train.train(args)



def select():
    fns = [fn for fn in os.listdir(FOLDER_PATH) if fn.startswith('id-')]
    fns.sort()

    best_config = None
    best_accuracy = 0
    best_model = ""
    for fn in fns:
        stats_fp = os.path.join(FOLDER_PATH, fn, 'stats.json')
        config_fp = os.path.join(FOLDER_PATH, fn, 'config.json')
        if os.path.exists(stats_fp) and os.path.exists(config_fp):
            with open(stats_fp, 'r') as fh:
                stats_dict = json.load(fh)
            with open(config_fp, 'r') as fh:
                config_dict = json.load(fh)

            if 'test_accuracy' not in stats_dict.keys():
                print(fn)
                continue

            if best_config is None:
                best_config = config_dict
                best_accuracy = stats_dict['test_accuracy']
                best_model = fn
            else:
                if stats_dict['test_accuracy'] > best_accuracy:
                    best_config = config_dict
                    best_model = fn

    print("Best Config (test accuracy = {}):".format(best_accuracy))
    print(best_model)
    print(best_config)



if __name__ == '__main__':
    search()
    #select()



