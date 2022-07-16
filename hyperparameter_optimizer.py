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
    learning_rate = np.random.choice([1e-3, 3e-4])
    weight_decay = np.random.uniform(0.1, 1.0)
    cycle_factor = float(np.random.uniform(2, 5))
    if np.random.rand() > 0.5:
        cycle_factor = None
    
    # from "Benchopt: Reproducible, efficient and collaborative optimization benchmarks" page 28
    # from "Lookahead optimizer: k Steps forward, 1 step back" page 17

    args = dict()
    args['arch'] = 'resnet18'
    args['num_workers'] = 2
    args['output_filepath'] = fp
    args['batch_size'] = 128
    args['learning_rate'] = learning_rate
    args['loss_eps'] = 1e-4
    args['num_lr_reductions'] = 2
    args['lr_reduction_factor'] = 0.25
    args['patience'] = 50
    args['weight_decay'] = weight_decay
    args['cycle_factor'] = cycle_factor
    args['starting_model'] = None
    args['debug'] = False
    args['amp'] = bool(np.random.uniform(0, 1.0) > 0.5)
    args['val_fraction'] = float(0.1)

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
            acc_vector.append(stats_dict['val_accuracy'])

            if best_config is None:
                best_config = config_dict
                best_accuracy = stats_dict['val_accuracy']
                best_model = fn
            else:
                if stats_dict['val_accuracy'] > best_accuracy:
                    best_config = config_dict
                    best_accuracy = stats_dict['val_accuracy']
                    best_model = fn

    print("Best Config (test accuracy = {}):".format(best_accuracy))
    print(best_model)
    print(best_config)


    learning_rate_vals = list()
    loss_eps_vals = list()
    patience_vals = list()
    cycle_factor_vals = list()
    val_fraction_vals = list()

    for c in config_vector:
        learning_rate_vals.append(c['learning_rate'])
        loss_eps_vals.append(c['loss_eps'])
        patience_vals.append(c['patience'])
        cycle_factor_vals.append(c['cycle_factor'])
        val_fraction_vals.append(c['val_fraction'])


    fig = plt.figure(figsize=(16, 9), dpi=100)
    plt.hist(acc_vector, bins=50)
    plt.xlabel('Test Accuracy')
    plt.ylabel('Count')
    plt.title('Test Accuracy Distribution')
    plt.show()

    plt.scatter(learning_rate_vals, acc_vector)
    plt.xlabel('Learning Rage')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy vs Learning Rate')
    plt.show()

    plt.scatter(loss_eps_vals, acc_vector)
    plt.xlabel('Loss Eps')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy vs Loss Eps')
    plt.show()

    plt.scatter(patience_vals, acc_vector)
    plt.xlabel('Patience')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy vs Patience')
    plt.show()

    plt.scatter(cycle_factor_vals, acc_vector)
    plt.xlabel('Cycle Factor')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy vs Cycle Factor')
    plt.show()

    plt.scatter(val_fraction_vals, acc_vector)
    plt.xlabel('Validation Fraction')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy vs Validation Fraction')
    plt.show()




if __name__ == '__main__':
    #search()
    select()



