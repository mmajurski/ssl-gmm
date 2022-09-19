import os
import numpy as np
import pandas as pd
import json

ifp = './models'

n_vals = [250, 1000, 4000]
p_vals = [0.99, 0.98, 0.95, 0.9, 0.75]
keys_to_remove = ['pseudo_label_counts_per_class','pseudo_label_true_counts_per_class','pseudo_labeling_accuracy_per_class']


df_list = list()

prefixes = ['only-supervised', 'ssl-resp', 'ssl-neum']

for pre in prefixes:
    for n in n_vals:
        m_fp = os.path.join(ifp, '{}-{}-models'.format(pre, n))
        if not os.path.exists(m_fp):
            print('missing folder: {}'.format(m_fp))
            continue
        fns = [fn for fn in os.listdir(m_fp) if fn.startswith('id-')]
        for fn in fns:
            if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                    stats_dict = json.load(json_file)
                for k in keys_to_remove:
                    if k in stats_dict:
                        del stats_dict[k]
                stats_dict['PL-method'] = pre
                stats_dict['N-SSL'] = n
                cd = pd.json_normalize(stats_dict)
                df_list.append(cd)

# gather percentile results
for n in n_vals:
    for p in p_vals:
        m_fp = os.path.join(ifp, 'ssl-perc{}-{}-models'.format(p, n))
        if not os.path.exists(m_fp):
            print('missing folder: {}'.format(m_fp))
            continue
        fns = [fn for fn in os.listdir(os.path.join(m_fp)) if fn.startswith('id-')]
        for fn in fns:
            if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                    stats_dict = json.load(json_file)
                for k in keys_to_remove:
                    if k in stats_dict:
                        del stats_dict[k]
                stats_dict['PL-method'] = 'percResp-sortNeum'
                stats_dict['N-SSL'] = n
                stats_dict['percentile'] = p
                cd = pd.json_normalize(stats_dict)
                df_list.append(cd)

full_df = pd.concat(df_list, axis=0)
cols = list(full_df.columns.values)

cols_to_remove = ['training_wall_time','val_wall_time','num_epochs_trained','train_wall_time', 'test_wall_time', 'wall_time', 'gmm_build_wall_time', 'pseudo_labeling_accuracy', 'num_added_pseudo_labels', 'train_labeled_dataset_size']
for c in cols_to_remove:
    if c in cols:
        cols.remove(c)

cols.remove('PL-method')
cols.remove('N-SSL')
cols.remove('percentile')

cols.insert(0, 'PL-method')
cols.insert(1, 'N-SSL')
cols.insert(2, 'percentile')
full_df = full_df[cols]


full_df.to_csv('./models/summary.csv', index=False)