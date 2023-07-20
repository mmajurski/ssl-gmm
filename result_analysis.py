import copy

import pandas as pd
import json
import os

# folder to read files from
directory = 'models'

# columns to extract from file name
config_columns = ['method', 'last_layer', 'ema', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint',  'model', 'learning_rate', 'nprefc']
# columns to extract from result file (stats.json)
result_columns = ['test_accuracy', 'wall_time', 'epoch', 'num_epochs_trained']
# create dataframe for storing results
final_columns = config_columns + result_columns
results_df = pd.DataFrame(columns=final_columns)


nb_complete = 0
nb_non_complete = 0
# iterating over folders in the given directory
for folder_name in os.listdir(directory):


    # getting configuration dictionary
    # config_dict = process_folder_name(folder_name)

    json_file_path = os.path.join(directory, folder_name, 'config.json')
    with open(json_file_path) as json_file:
        full_config_dict = json.load(json_file)

    config_dict = dict()
    config_dict['method'] = full_config_dict['trainer']
    config_dict['model'] = folder_name
    config_dict['num_labeled_datapoints'] = full_config_dict['num_labeled_datapoints']
    config_dict['embedding_constraint'] = full_config_dict['embedding_constraint']
    config_dict['last_layer'] = full_config_dict['last_layer']
    config_dict['learning_rate'] = full_config_dict['learning_rate']
    config_dict['ema'] = full_config_dict['use_ema']
    if 'embedding_dim' not in full_config_dict.keys():
        full_config_dict['embedding_dim'] = 8
    config_dict['embedding_dim'] = full_config_dict['embedding_dim']
    if 'nprefc' not in full_config_dict.keys():
        full_config_dict['nprefc'] = 0
    config_dict['nprefc'] = full_config_dict['nprefc']


    config_dict = dict((k, config_dict[k]) for k in config_columns)

    # creating dictionary from stats.json file
    json_file_path = os.path.join(directory, folder_name, 'stats.json')
    result_dict = dict()
    if os.path.exists(json_file_path):
        with open(json_file_path) as json_file:
            result_dict = json.load(json_file)

        # do this if adding one of these columns as these are list values which will create multiple rows in excel file
        # result_dict['pseudo_label_counts_per_class'] = str(result_dict['pseudo_label_counts_per_class'])
        # result_dict['pseudo_label_gt_counts_per_class'] = str(result_dict['pseudo_label_gt_counts_per_class'])
        # result_dict['pseudo_label_accuracy_per_class'] = str(result_dict['pseudo_label_accuracy_per_class'])

        # only keeping the desired columns
        result_columns_lcl = copy.deepcopy(result_columns)
        for c in result_columns:
            if c not in result_dict.keys():
                result_columns_lcl.remove(c)
        result_dict = dict((k, result_dict[k]) for k in result_columns_lcl)
    # combining both the dictionaries
    combined_dict = config_dict | result_dict

    if 'wall_time' not in combined_dict.keys():
        nb_non_complete += 1
    else:
        nb_complete += 1
        # adding the row to final results
        row_df = pd.DataFrame([combined_dict])
        results_df = pd.concat([results_df, row_df])

# exporting to cvs file
results_df.to_csv('results-current.csv', index=False)
print("found {} complete results".format(nb_complete))
print("found {} non complete results".format(nb_non_complete))
