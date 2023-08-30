import copy

import pandas as pd
import json
import os

# folder to read files from
post_fix = 'all'
# post_fix = 'ingest'
directory = 'models-{}'.format(post_fix)

# columns to extract from file name
config_columns = ['trainer', 'last_layer', 'use_ema', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint', 'clip_grad', 'patience', 'nesterov']
# columns to extract from result file (stats.json)
result_columns = ['test_accuracy', 'epoch']
results_df = None

nb_complete = 0
# iterating over folders in the given directory
folder_names = [fn for fn in os.listdir(directory) if fn.startswith('id-')]
folder_names.sort()
success_list = list()
for folder_name in folder_names:
    json_file_path = os.path.join(directory, folder_name, 'config.json')
    with open(json_file_path) as json_file:
        config_dict = json.load(json_file)

    config_dict = dict((k, config_dict[k]) for k in config_columns)
    config_dict['model'] = folder_name

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

    # with open(os.path.join(directory, folder_name, 'success.txt'), mode='w', encoding='utf-8') as f:
    #     f.write('success')
    # continue

    if os.path.exists(os.path.join(directory, folder_name, 'failure.txt')):
        print("Model failure: {}".format(folder_name))
    elif not os.path.exists(os.path.join(directory, folder_name, 'success.txt')):
        print("Model non-success: {}".format(folder_name))
        # # adding the row to final results
        # row_df = pd.DataFrame([combined_dict])
        # if results_df is None:
        #     results_df = row_df
        # else:
        #     results_df = pd.concat([results_df, row_df])
    else:
        nb_complete += 1
        success_list.append(folder_name)
        # adding the row to final results
        row_df = pd.DataFrame([combined_dict])
        if results_df is None:
            results_df = row_df
        else:
            results_df = pd.concat([results_df, row_df])

# exporting to cvs file
results_df.to_csv('results-{}.csv'.format(post_fix), index=False)
print("found {} complete results".format(nb_complete))


for fn in success_list:
    print('rm -rf {}'.format(fn))