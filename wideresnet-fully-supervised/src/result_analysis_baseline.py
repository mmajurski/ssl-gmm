import pandas as pd
import json
import os



# folder to read files from
directory = 'models-fixmatch-baseline'

# columns to extract from file name
config_columns = ['ema', 'tau', 'tau_method', 'num_epochs']
# columns to extract from result file (stats.json)
result_columns = ['train_accuracy', 'test_accuracy', 'num_epochs_trained']
# create dataframe for storing results
final_columns = config_columns + result_columns
results_df = pd.DataFrame(columns=final_columns)



# iterating over folders in the given directory
for folder_name in os.listdir(directory):

    # getting configuration dictionary
    # config_dict = process_folder_name(folder_name)

    json_file_path = os.path.join(directory, folder_name, 'config.json')
    with open(json_file_path) as json_file:
        full_config_dict = json.load(json_file)

    config_dict = dict()
    config_dict['ema'] = full_config_dict['use_ema']
    config_dict['tau'] = full_config_dict['tau']
    config_dict['tau_method'] = full_config_dict['tau_method']
    if 'num_epochs' in full_config_dict.keys():
        config_dict['num_epochs'] = full_config_dict['num_epochs']
    else:
        config_dict['num_epochs'] = 'early stopping'

    config_dict = dict((k, config_dict[k]) for k in config_columns)

    # creating dictionary from stats.json file
    json_file_path = os.path.join(directory, folder_name, 'stats.json')
    with open(json_file_path) as json_file:
        result_dict = json.load(json_file)

    # only keeping the desired columns
    result_dict = dict((k, result_dict[k]) for k in result_columns)
    # combining both the dictionaries
    combined_dict = config_dict | result_dict

    # adding the row to final results
    row_df = pd.DataFrame([combined_dict])
    results_df = pd.concat([results_df, row_df])

# exporting to excel file
results_df.to_csv('results-fixmatch-baseline.csv', index=False)









