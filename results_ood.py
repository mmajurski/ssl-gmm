import copy

import numpy as np
import pandas as pd
import json
import os


def gen_key(cdict):
    key = ""
    for col in cdict.keys():
        if col == 'ood_p':
            continue
        key += str(col)[0:4] + ':' + str(cdict[col]) + '-'
    return key

# folder to read files from
post_fix = 'ood'
directory = 'models-{}'.format(post_fix)

# columns to extract from file name
# config_columns = ['trainer', 'last_layer', 'use_ema', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint', 'clip_grad', 'patience', 'nesterov']
config_columns = ['trainer', 'last_layer', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint', 'ood_p']
# columns to extract from result file (stats.json)
result_columns = ['test_accuracy', 'epoch', 'test_accuracy_per_class']
results_df = None

# iterating over folders in the given directory
folder_names = [fn for fn in os.listdir(directory) if fn.startswith('id-')]
folder_names.sort()
dict_of_df_lists = dict()
unique_ood_p_vals = set()


for folder_name in folder_names:
    # print(folder_name)
    if os.path.exists(os.path.join(directory, folder_name, 'failure.txt')):
        print("Model failure: {}".format(folder_name))
    elif not os.path.exists(os.path.join(directory, folder_name, 'success.txt')):
        print("Model non-success: {}".format(folder_name))
    else:
        json_file_path = os.path.join(directory, folder_name, 'config.json')
        with open(json_file_path) as json_file:
            config_dict = json.load(json_file)

        if config_dict['patience'] == 20:
            continue
        if config_dict['clip_grad'] == False:
            continue
        if config_dict['embedding_dim'] != 128:
            continue
        if config_dict['num_labeled_datapoints'] != 250:
            continue

        config_dict = dict((k, config_dict[k]) for k in config_columns)

        # creating dictionary from stats.json file
        json_file_path = os.path.join(directory, folder_name, 'stats.json')
        result_dict = dict()
        if os.path.exists(json_file_path):
            with open(json_file_path) as json_file:
                result_dict = json.load(json_file)

            # only keeping the desired columns
            result_columns_lcl = copy.deepcopy(result_columns)
            for c in result_columns:
                if c not in result_dict.keys():
                    result_columns_lcl.remove(c)
            result_dict = dict((k, result_dict[k]) for k in result_columns_lcl)

            ta = result_dict['test_accuracy_per_class']
            result_dict['min_test_accuracy_per_class'] = np.min(ta)

        # combining both the dictionaries
        combined_dict = config_dict | result_dict
        unique_ood_p_vals.add(combined_dict['ood_p'])

        key = gen_key(config_dict)
        if key not in dict_of_df_lists.keys():
            dict_of_df_lists[key] = list()
        dict_of_df_lists[key].append(combined_dict)

post_fix = 'all'
directory = 'models-{}'.format(post_fix)
folder_names = [fn for fn in os.listdir(directory) if fn.startswith('id-')]
folder_names.sort()

post_fix = 'ood'

for folder_name in folder_names:
    # print(folder_name)
    if os.path.exists(os.path.join(directory, folder_name, 'failure.txt')):
        print("Model failure: {}".format(folder_name))
    elif not os.path.exists(os.path.join(directory, folder_name, 'success.txt')):
        print("Model non-success: {}".format(folder_name))
    else:
        json_file_path = os.path.join(directory, folder_name, 'config.json')
        with open(json_file_path) as json_file:
            config_dict = json.load(json_file)

        if config_dict['patience'] == 20:
            continue
        if config_dict['clip_grad'] == False:
            continue
        if config_dict['num_labeled_datapoints'] != 250:
            continue
        if config_dict['embedding_dim'] != 128:
            continue

        config_dict = dict((k, config_dict[k]) for k in config_columns)

        # creating dictionary from stats.json file
        json_file_path = os.path.join(directory, folder_name, 'stats.json')
        result_dict = dict()
        if os.path.exists(json_file_path):
            with open(json_file_path) as json_file:
                result_dict = json.load(json_file)

            # only keeping the desired columns
            result_columns_lcl = copy.deepcopy(result_columns)
            for c in result_columns:
                if c not in result_dict.keys():
                    result_columns_lcl.remove(c)
            result_dict = dict((k, result_dict[k]) for k in result_columns_lcl)

            ta = result_dict['test_accuracy_per_class']
            result_dict['min_test_accuracy_per_class'] = np.min(ta)

        # combining both the dictionaries
        combined_dict = config_dict | result_dict
        unique_ood_p_vals.add(combined_dict['ood_p'])

        key = gen_key(config_dict)
        if key not in dict_of_df_lists.keys():
            dict_of_df_lists[key] = list()
        dict_of_df_lists[key].append(combined_dict)


df_list = list()
unique_ood_p_vals = list(unique_ood_p_vals)
unique_ood_p_vals.sort()

for config_key in dict_of_df_lists.keys():
    # compute the average test_accuracy for all json dicts in this list
    ta = list()
    ood_p = list()
    for d in dict_of_df_lists[config_key]:
        ta.append(100 * d['test_accuracy'])
        ood_p.append(d['ood_p'])

    a = dict_of_df_lists[config_key][0]
    del a['test_accuracy']
    del a['epoch']
    del a['ood_p']
    del a['test_accuracy_per_class']
    del a['min_test_accuracy_per_class']

    vals = np.unique(ood_p)
    for val in unique_ood_p_vals:
        if val in vals:
            if val == 0.0:
                o = np.mean(ta)
            else:
                idx = np.where(vals == val)[0][0]
                o = ta[idx]
        else:
            o = None

        a['oop={}'.format(val)] = o

    cd = pd.json_normalize(a)
    df_list.append(cd)

results_df = pd.concat(df_list, axis=0)

# split the pandas dataframe results_df into multiple dataframes based on the column num_labeled_datapoints
# and save them to csv files
for num_labeled_datapoints in results_df['num_labeled_datapoints'].unique():
    df = results_df[results_df['num_labeled_datapoints'] == num_labeled_datapoints]
    df.to_csv('results-{}-{}labels.csv'.format(post_fix, num_labeled_datapoints), index=False)


# # exporting to cvs file
# results_df.to_csv('results-{}.csv'.format(post_fix), index=False)
