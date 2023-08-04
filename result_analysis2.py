import copy

import numpy as np
import pandas as pd
import json
import os


def gen_key(cdict):
    key = ""
    for col in cdict.keys():
        key += str(col)[0:4] + ':' + str(cdict[col]) + '-'
    return key

# folder to read files from
# post_fix = 'fixed-seed'
# post_fix = 'noarchmod'
post_fix = 'fixmatch-noarchmod'
directory = 'models-{}'.format(post_fix)

# columns to extract from file name
config_columns = ['trainer', 'last_layer', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint']
# columns to extract from result file (stats.json)
result_columns = ['test_accuracy']
results_df = None

# iterating over folders in the given directory
folder_names = [fn for fn in os.listdir(directory) if fn.startswith('id-')]
folder_names.sort()
dict_of_df_lists = dict()

for folder_name in folder_names:

    json_file_path = os.path.join(directory, folder_name, 'config.json')
    with open(json_file_path) as json_file:
        config_dict = json.load(json_file)

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
    # combining both the dictionaries
    combined_dict = config_dict | result_dict

    key = gen_key(config_dict)
    if key not in dict_of_df_lists.keys():
        dict_of_df_lists[key] = list()
    dict_of_df_lists[key].append(combined_dict)

df_list = list()
for config_key in dict_of_df_lists.keys():
    # compute the average test_accuracy for all json dicts in this list
    ta = list()
    for d in dict_of_df_lists[config_key]:
        ta.append(d['test_accuracy'])

    a = dict_of_df_lists[config_key][0]
    del a['test_accuracy']
    a['mean_test_accuracy'] = float(np.mean(ta))
    a['std_test_accuracy'] = float(np.std(ta))
    a['max_test_accuracy'] = float(np.max(ta))
    a['min_test_accuracy'] = float(np.min(ta))
    a['nb_runs'] = len(ta)
    cd = pd.json_normalize(a)
    df_list.append(cd)

results_df = pd.concat(df_list, axis=0)

# exporting to cvs file
results_df.to_csv('results-{}.csv'.format(post_fix), index=False)
