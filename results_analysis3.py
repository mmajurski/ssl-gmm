import copy

import numpy as np
import pandas as pd
import json
import os
import tbparse


# folder to read files from
ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv'

fldrs = [fn for fn in os.listdir(ifp) if os.path.isdir(os.path.join(ifp, fn))]
fldrs.sort()

df_list = list()

for fldr in fldrs:
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
    grad_clip = float(toks[4])

    a = dict()
    a['final_layer'] = final_layer
    a['dataset'] = dataset
    a['num_labeled_datapoints'] = num_labeled_datapoints
    a['run_num'] = run_num
    a['embedding_constraint'] = embedding_constraint
    a['grad_clip'] = grad_clip
    a['eval_f1'] = eval_f1['value'].max()

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

df_list = list()
for dataset in datasets:
    for num_labeled_datapoint in num_labeled_datapoints:
        for final_layer in final_layers:
            for embedding_constraint in embedding_constraints:
                for grad_clip in grad_clips:
                    df = results_df[(results_df['dataset'] == dataset) & (results_df['num_labeled_datapoints'] == num_labeled_datapoint) & (results_df['final_layer'] == final_layer) & (results_df['embedding_constraint'] == embedding_constraint) & (results_df['grad_clip'] == grad_clip)]
                    if len(df) > 0:
                        # copy out the first row of the dataframe
                        df2 = df.iloc[[0]]
                        df2 = df2.drop(columns=['eval_f1', 'run_num'])
                        df2['eval_f1_mean'] = df['eval_f1'].mean()
                        df2['eval_f1_err_mean'] = 1.0 - df['eval_f1'].mean()
                        df2['eval_f1_err_std'] = df['eval_f1'].std()

                        df_list.append(df2)

stats_df = pd.concat(df_list, axis=0)


# exporting to cvs file
results_df.to_csv('results-usb.csv', index=False)
stats_df.to_csv('results-usb-stats.csv', index=False)
