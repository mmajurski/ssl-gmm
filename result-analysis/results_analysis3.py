import copy

import numpy as np
import pandas as pd
import json
import os
import tbparse


# TODO capture the model eval_f1 at 15k..5k..40k steps to understand init convergence

# folder to read files from
ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv'

fldrs = [fn for fn in os.listdir(ifp) if os.path.isdir(os.path.join(ifp, fn))]
fldrs = [fn for fn in fldrs if 'debug' not in fn]
fldrs = [fn for fn in fldrs if 'freematch' not in fn]
fldrs = [fn for fn in fldrs if 'flexmatch' not in fn]
fldrs.sort()

df_list = list()

for fldr in fldrs:
    print(fldr)

    toks = fldr.split('_')
    final_layer = toks[0]
    embedding_constraint = None
    try:
        int(final_layer[-1])
        embedding_constraint = int(final_layer[-1:])
        final_layer = final_layer[:-1]
    except:
        pass
    dataset = toks[1]
    num_labeled_datapoints = int(toks[2])
    run_num = int(toks[3])
    grad_clip = 1.0
    embedding_dim = 8
    outlier_detection_method = str(toks[4])
    if len(toks) > 5:
        outlier_filter_method = str(toks[5])
    else:
        outlier_filter_method = 'none'

    if num_labeled_datapoints == 250:
        continue

    tb_fldr = os.path.join(ifp, fldr)
    # https://pypi.org/project/tbparse/
    reader = tbparse.SummaryReader(tb_fldr)
    df = reader.scalars

    eval_loss = df[df['tag'] == 'eval/loss']
    eval_f1 = df[df['tag'] == 'eval/F1']
    train_util_ratio = df[df['tag'] == 'train/util_ratio']
    train_denom_thres = df[df['tag'] == 'train/denom_thres']
    train_denom_thres_rate = df[df['tag'] == 'train/dthres_rate']


    # from matplotlib import pyplot as plt
    # plt.plot(eval_f1['step'], eval_f1['value'])
    # plt.title('eval-f1')
    # plt.show()
    # plt.plot(eval_loss['step'], eval_loss['value'])
    # plt.title('eval-loss')
    # plt.show()
    # plt.plot(train_util_ratio['step'], train_util_ratio['value'])
    # plt.title('train-util_ratio')
    # plt.show()


    a = dict()
    a['final_layer'] = final_layer
    a['dataset'] = dataset
    a['num_labeled_datapoints'] = num_labeled_datapoints
    a['run_num'] = run_num
    a['embedding_constraint'] = embedding_constraint
    a['embedding_dim'] = embedding_dim
    a['grad_clip'] = grad_clip
    a['eval_f1'] = eval_f1['value'].max()
    a['outlier_detection_method'] = outlier_detection_method
    a['outlier_filter_method'] = outlier_filter_method

    cd = pd.json_normalize(a)
    df_list.append(cd)


results_df = pd.concat(df_list, axis=0)

# split the pandas dataframe results_df into multiple dataframes based on the column num_labeled_datapoints
# and save them to csv files
datasets = results_df['dataset'].unique()
num_labeled_datapoints = results_df['num_labeled_datapoints'].unique()
final_layers = results_df['final_layer'].unique()
embedding_constraints = results_df['embedding_constraint'].unique()
grad_clips = results_df['grad_clip'].unique()
embedding_dims = results_df['embedding_dim'].unique()
outlier_detection_methods = results_df['outlier_detection_method'].unique()
outlier_filter_methods = results_df['outlier_filter_method'].unique()

df_list = list()
for dataset in datasets:
    for num_labeled_datapoint in num_labeled_datapoints:
        for final_layer in final_layers:
            for embedding_constraint in embedding_constraints:
                for embedding_dim in embedding_dims:
                    for grad_clip in grad_clips:
                        for outlier_detection in outlier_detection_methods:
                            for outlier_filter in outlier_filter_methods:
                                df = results_df[(results_df['dataset'] == dataset) & (results_df['num_labeled_datapoints'] == num_labeled_datapoint) & (results_df['final_layer'] == final_layer) & (results_df['embedding_constraint'] == embedding_constraint) & (results_df['grad_clip'] == grad_clip) & (results_df['embedding_dim'] == embedding_dim) & (results_df['outlier_detection_method'] == outlier_detection) & (results_df['outlier_filter_method'] == outlier_filter)]
                                if len(df) > 0:
                                    # copy out the first row of the dataframe
                                    df2 = df.iloc[[0]]
                                    df2 = df2.drop(columns=['eval_f1', 'run_num'])
                                    df2['eval_f1_mean'] = 100.0 * df['eval_f1'].mean()
                                    df2['eval_f1_err_mean'] = 100.0 * (1.0 - df['eval_f1'].mean())
                                    df2['eval_f1_err_std'] = 100.0 * df['eval_f1'].std()
                                    df2['n'] = np.max(df['run_num']) + 1

                                    df_list.append(df2)

stats_df = pd.concat(df_list, axis=0)


# exporting to cvs file
results_df.to_csv('results-usb.csv', index=False)
stats_df.to_csv('results-usb-stats.csv', index=False)
