import copy

import numpy as np
import pandas as pd
import json
import os


def gen_key(cdict):
    key = ""
    keys = list(cdict.keys())
    keys = [k for k in keys if k != 'model']
    keys.sort()
    for col in keys:
        key += str(col)[0:4] + ':' + str(cdict[col]) + '-'
    return key

# folder to read files from
post_fix = 'cifar10'
post_fix = 'cifar10-unused'
# post_fix = 'cifar100'
directory = 'models-{}'.format(post_fix)

# columns to extract from file name
# config_columns = ['trainer', 'last_layer', 'use_ema', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint', 'clip_grad', 'patience', 'nesterov']
config_columns = ['model', 'trainer', 'last_layer', 'embedding_dim', 'num_labeled_datapoints', 'embedding_constraint', 'clip_grad']
# columns to extract from result file (stats.json)
result_columns = ['test_accuracy', 'epoch', 'test_accuracy_per_class']
results_df = None

# iterating over folders in the given directory
folder_names = [fn for fn in os.listdir(directory) if fn.startswith('id-')]
folder_names.sort()
dict_of_df_lists = dict()

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

        config_dict['model'] = folder_name

        if config_dict['patience'] == 20:
            continue
        # if config_dict['embedding_constraint'] != 'mean_covar' and config_dict['clip_grad'] == True:
        #     continue
        # if config_dict['embedding_dim'] == 8:
        #     continue
        # if config_dict['embedding_constraint'] == 'mean_covar':
        #     continue
        # if config_dict['last_layer'] == 'kmeans' and config_dict['embedding_constraint'] != 'mean_covar' and config_dict['clip_grad'] == True:
        #     continue
        if config_dict['last_layer'] == 'kmeans' and config_dict['embedding_constraint'] == 'none' and config_dict['clip_grad'] == True:
            continue
        if config_dict['last_layer'] == 'kmeans' and config_dict['embedding_constraint'] == 'l2' and config_dict['clip_grad'] == True:
            continue
        if config_dict['last_layer'] == 'kmeans' and config_dict['embedding_constraint'] == 'mean_covar' and config_dict['clip_grad'] == False:
            continue
        if config_dict['embedding_constraint'] == 'gauss_moment3' and config_dict['clip_grad'] == False:
            continue
        if config_dict['embedding_constraint'] == 'gauss_moment4' and config_dict['clip_grad'] == False:
            continue
        if config_dict['last_layer'] == 'aa_gmm' and config_dict['clip_grad'] == False:
            continue
        if config_dict['last_layer'] == 'fc' and config_dict['embedding_constraint'] != 'none':
            continue
        if config_dict['last_layer'] == 'fc' and config_dict['clip_grad'] == True:
            continue
        # if config_dict['embedding_constraint'] == 'mean_covar' and config_dict['clip_grad'] == False:
        #     continue


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

        key = gen_key(config_dict)
        if key not in dict_of_df_lists.keys():
            dict_of_df_lists[key] = list()
        dict_of_df_lists[key].append(combined_dict)


df_list = list()
for config_key in dict_of_df_lists.keys():
    # compute the average test_accuracy for all json dicts in this list
    ta = list()
    ep = list()
    mta = list()
    for d in dict_of_df_lists[config_key]:
        if 'test_accuracy' in d.keys():
            ta.append(100 * d['test_accuracy'])
            ep.append(d['epoch'])
            mta.append(100 * d['min_test_accuracy_per_class'])


    if len(ta) == 0:
        a = dict_of_df_lists[config_key][0]
        cd = pd.json_normalize(a)
        df_list.append(cd)
        continue

    mv = float(np.mean(ta))
    q1 = float(np.quantile(ta, 0.25))
    q3 = float(np.quantile(ta, 0.75))
    iqr = np.inf
    # iqr = 1.5 * (q3 - q1)
    # iqr = 1.0 * (q3 - q1)

    exclude_both = False
    if exclude_both:
        idx = np.logical_or(ta < np.asarray(q1 - iqr), ta > np.asarray(q3 + iqr))
    else:
        idx = ta < np.asarray(q1 - iqr)
    for i in range(len(dict_of_df_lists[config_key])):
        if idx[i]:
            d = dict_of_df_lists[config_key][i]
            # print("outlier: {}".format(d['model']))
            print("mv {} ../models-cifar10-unused/".format(d['model']))

    # if len(ta) > 6:  # and '250' in config_key:
    #     # pick just 6 at random
    #     overage = len(ta) - 6
    #
    #     k = 0
    #     thres = 0.01
    #     while True:
    #         k += 1
    #         idx2 = set(np.random.choice(len(ta), overage, replace=False))
    #         idx1 = set(np.asarray(list(range(len(ta)))))
    #         idx1 = idx1 - idx2
    #         idx1 = list(idx1)
    #         idx2 = list(idx2)
    #         full_mean = np.mean(ta)
    #         full_std = np.std(ta)
    #         subset_mean = np.mean(np.asarray(ta)[idx1])
    #         subset_std = np.std(np.asarray(ta)[idx1])
    #         std_delta = abs(subset_std - full_std)
    #         if subset_std < full_std:
    #             std_delta = 0.0
    #         if abs(subset_mean - full_mean) < thres and std_delta < 4*thres:
    #             break
    #         if k > 1000:
    #             if thres >= 0.2:
    #                 print(config_key)
    #                 raise RuntimeError("Could not find subset with equivalent mean/std")
    #             thres += 0.01
    #             k = 0
    #
    #     for i in idx2:
    #         d = dict_of_df_lists[config_key][i]
    #         # print("unused: {}".format(d['model']))
    #         print("mv {} ../models-cifar10-unused/".format(d['model']))
    #     idx[idx2] = True


    removed_ta = np.asarray(ta)[idx].tolist()
    ta = np.asarray(ta)[np.logical_not(idx)].tolist()
    removed_mta = np.asarray(mta)[idx].tolist()
    mta = np.asarray(mta)[np.logical_not(idx)].tolist()

    # if len(ta) < 6:
    #     print(config_key)
    #     print("missing {}".format(6 - len(ta)))


    a = dict_of_df_lists[config_key][0]
    del a['test_accuracy']
    del a['epoch']
    if 'model' in a.keys():
        del a['model']
    del a['test_accuracy_per_class']
    del a['min_test_accuracy_per_class']
    a['mean_test_accuracy'] = float(np.mean(ta))
    a['std_test_accuracy'] = float(np.std(ta))
    a['max_test_accuracy'] = float(np.max(ta))
    a['nb_runs'] = len(ta)

    ta = [round(10.0 * v) / 10.0 for v in ta]
    removed_ta = [round(10.0 * v) / 10.0 for v in removed_ta]
    mta = [round(10.0 * v) / 10.0 for v in mta]
    removed_mta = [round(10.0 * v) / 10.0 for v in removed_mta]

    a['test_accuracies'] = ta
    a['outliers_test_accuracies'] = removed_ta
    a['epoch_counts'] = ep
    cd = pd.json_normalize(a)
    df_list.append(cd)

results_df = pd.concat(df_list, axis=0)
# results_df.insert(5, 'clip_grad marginal utility', value=None)

# split the pandas dataframe results_df into multiple dataframes based on the column num_labeled_datapoints
# and save them to csv files
for num_labeled_datapoints in results_df['num_labeled_datapoints'].unique():
    df = results_df[results_df['num_labeled_datapoints'] == num_labeled_datapoints]
    df.to_csv('results-{}-{}labels.csv'.format(post_fix, num_labeled_datapoints), index=False)

    # #print("******** {} ********".format(num_labeled_datapoints))
    # for ll in results_df['last_layer'].unique():
    #     for emb in results_df['embedding_constraint'].unique():
    #         for emb_c in results_df['embedding_dim'].unique():
    #             df = results_df[results_df['num_labeled_datapoints'] == num_labeled_datapoints]
    #             df = df[df['last_layer'] == ll]
    #             df = df[df['embedding_constraint'] == emb]
    #             df = df[df['embedding_dim'] == emb_c]
    #             a = df[df['clip_grad'] == True]
    #             b = df[df['clip_grad'] == False]
    #             if ll != 'fc' and len(a) > 0 and len(b) > 0:
    #                 idx = (results_df['num_labeled_datapoints'] == num_labeled_datapoints) & \
    #                                            (results_df['last_layer'] == ll) & \
    #                                            (results_df['embedding_constraint'] == emb) & \
    #                                            (results_df['embedding_dim'] == emb_c) & \
    #                                            (results_df['clip_grad'] == True)
    #
    #
    #                 a = a['mean_test_accuracy'].values[0]
    #                 b = b['mean_test_accuracy'].values[0]
    #                 results_df.loc[idx, 'clip_grad marginal utility'] = a - b
    #                 #df['clip_grad marginal utility'] = a - b
    #
    #                 #print("{},{},{}\tclip_grad = {}, without = {}, delta = {}".format(ll, emb, emb_c, a, b, a-b))
    # df = results_df[results_df['num_labeled_datapoints'] == num_labeled_datapoints]
    # cols = list(df.columns)
    # cols.remove('clip_grad marginal utility')
    # cols = cols.insert(5, 'clip_grad marginal utility')
    # df.to_csv('results-{}-{}labels.csv'.format(post_fix, num_labeled_datapoints), index=False)



    # for ll in results_df['last_layer'].unique():
    #     df = results_df[results_df['num_labeled_datapoints'] == num_labeled_datapoints]
    #     df = df[df['last_layer'] == ll]
    #     df.to_csv('results-{}-{}labels-{}.csv'.format(post_fix, num_labeled_datapoints,ll), index=False)


# # exporting to cvs file
# results_df.to_csv('results-{}.csv'.format(post_fix), index=False)
