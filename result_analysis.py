import pandas as pd
import json
import os

# folder to read files from
directory = 'models-20230417'

# columns to extract from file name
config_columns = ['method', 'final_layer', 'run', 'pseudo_labeler', 'pl_target', 'loss']
# columns to extract from result file (stats.json)
result_columns = ['val_gmm_accuracy', 'val_cmm_accuracy', 'num_epochs_trained', 'epoch',
                  'train_gmm_accuracy', 'train_cmm_accuracy']
# create dataframe for storing results
final_columns = config_columns + result_columns
results_df = pd.DataFrame(columns=final_columns)



def process_folder_name(f_name):
    """
       process folder name and return config_dict.

       :param str f_name: Name of the folder
       :return: config_dict containing config_columns and appropriate values
       :rtype: dict
       """
    f_name_list = f_name.split('-')

    return_dict = {'method': f_name_list[0], 'final_layer': f_name_list[1], 'run': f_name_list[2],
                   'pseudo_labeler': f_name_list[3].lstrip('pl'), 'pl_target': f_name_list[4][5:],
                   'loss': f_name_list[5].lstrip('loss'), 'val_acc_tgt': f_name_list[6].replace('valacc', ''),
                   'ema': f_name_list[7].replace('ema', '')}

    return return_dict


# iterating over folders in the given directory
for folder_name in os.listdir(directory):

    # getting configuration dictionary
    config_dict = process_folder_name(folder_name)

    # creating dictionary from stats.json file
    json_file_path = os.path.join(directory, folder_name, 'stats.json')
    with open(json_file_path) as json_file:
        result_dict = json.load(json_file)

    # do this if adding one of these columns as these are list values which will create multiple rows in excel file
    # result_dict['pseudo_label_counts_per_class'] = str(result_dict['pseudo_label_counts_per_class'])
    # result_dict['pseudo_label_gt_counts_per_class'] = str(result_dict['pseudo_label_gt_counts_per_class'])
    # result_dict['pseudo_label_accuracy_per_class'] = str(result_dict['pseudo_label_accuracy_per_class'])

    # only keeping the desired columns
    result_dict = dict((k, result_dict[k]) for k in result_columns)
    # combining both the dictionaries
    combined_dict = config_dict | result_dict

    # adding the row to final results
    row_df = pd.DataFrame([combined_dict])
    results_df = pd.concat([results_df, row_df])

# exporting to excel file
results_df.to_csv('results.csv', index=False)
