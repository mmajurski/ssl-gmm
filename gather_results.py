import os
import numpy as np
import pandas as pd
import json


def gather_models_results():
    ifp = './models'

    n_vals = [250, 1000, 4000]
    p_vals = [0.99, 0.98, 0.95, 0.9, 0.75]
    keys_to_remove = ['pseudo_label_counts_per_class','pseudo_label_true_counts_per_class','pseudo_labeling_accuracy_per_class']

    for n in n_vals:

        df_list = list()
        avg_df_list = list()

        prefixes = ['only-supervised', 'ssl-resp', 'ssl-neum', 'ssl-resp-cauchy', 'ssl-neum-cauchy']

        for pre in prefixes:
            m_fp = os.path.join(ifp, '{}-{}-models'.format(pre, n))
            if not os.path.exists(m_fp):
                print('missing folder: {}'.format(m_fp))
                continue
            fns = [fn for fn in os.listdir(m_fp) if fn.startswith('id-')]
            acc_list = list()
            for fn in fns:
                if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                    with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                        stats_dict = json.load(json_file)
                    for k in keys_to_remove:
                        if k in stats_dict:
                            del stats_dict[k]
                    stats_dict['PL-method'] = pre
                    stats_dict['N-SSL'] = n
                    if 'test_softmax_accuracy' in stats_dict:
                        acc_list.append(stats_dict['test_softmax_accuracy'])
                    cd = pd.json_normalize(stats_dict)
                    df_list.append(cd)

            avg_stats_dict = dict()
            avg_stats_dict['PL-method'] = pre
            avg_stats_dict['N-SSL'] = n
            avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
            avg_stats_dict['std_accuracy'] = np.std(acc_list)
            avg_stats_dict['counts'] = len(acc_list)
            cd = pd.json_normalize(avg_stats_dict)
            avg_df_list.append(cd)


        # gather percentile results
        for p in p_vals:
            m_fp = os.path.join(ifp, 'ssl-perc{}-{}-models'.format(p, n))
            if not os.path.exists(m_fp):
                print('missing folder: {}'.format(m_fp))
                continue
            fns = [fn for fn in os.listdir(os.path.join(m_fp)) if fn.startswith('id-')]
            acc_list = list()
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
                    if 'test_softmax_accuracy' in stats_dict:
                        acc_list.append(stats_dict['test_softmax_accuracy'])
                    cd = pd.json_normalize(stats_dict)
                    df_list.append(cd)

            avg_stats_dict = dict()
            avg_stats_dict['PL-method'] = 'percResp-sortNeum'
            avg_stats_dict['N-SSL'] = n
            avg_stats_dict['percentile'] = p
            avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
            avg_stats_dict['std_accuracy'] = np.std(acc_list)
            avg_stats_dict['counts'] = len(acc_list)
            cd = pd.json_normalize(avg_stats_dict)
            avg_df_list.append(cd)

        for p in p_vals:
            m_fp = os.path.join(ifp, 'ssl-cauchy-perc{}-{}-models'.format(p, n))
            if not os.path.exists(m_fp):
                print('missing folder: {}'.format(m_fp))
                continue
            fns = [fn for fn in os.listdir(os.path.join(m_fp)) if fn.startswith('id-')]
            acc_list = list()
            for fn in fns:
                if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                    with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                        stats_dict = json.load(json_file)
                    for k in keys_to_remove:
                        if k in stats_dict:
                            del stats_dict[k]
                    stats_dict['PL-method'] = 'percResp-sortNeum-cauchy'
                    stats_dict['N-SSL'] = n
                    stats_dict['percentile'] = p
                    if 'test_softmax_accuracy' in stats_dict:
                        acc_list.append(stats_dict['test_softmax_accuracy'])
                    cd = pd.json_normalize(stats_dict)
                    df_list.append(cd)
            avg_stats_dict = dict()
            avg_stats_dict['PL-method'] = 'percResp-sortNeum-cauchy'
            avg_stats_dict['N-SSL'] = n
            avg_stats_dict['percentile'] = p
            avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
            avg_stats_dict['std_accuracy'] = np.std(acc_list)
            avg_stats_dict['counts'] = len(acc_list)
            cd = pd.json_normalize(avg_stats_dict)
            avg_df_list.append(cd)

        full_df = pd.concat(df_list, axis=0)
        avg_full_df = pd.concat(avg_df_list, axis=0)
        cols = list(full_df.columns.values)

        cols_to_remove = ['training_wall_time','val_wall_time','num_epochs_trained','train_wall_time', 'test_wall_time', 'wall_time', 'gmm_build_wall_time', 'num_added_pseudo_labels', 'train_labeled_dataset_size', 'val_gmm_accuracy', 'val_cauchy_accuracy', 'test_gmm_accuracy', 'test_cauchy_accuracy', 'avg_pseudo_labeling_accuracy_per_class']
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


        full_df.to_csv(os.path.join(ifp, 'summary-{}.csv'.format(n)), index=False)
        avg_full_df.to_csv(os.path.join(ifp, 'summary-{}-avg.csv'.format(n)), index=False)


def gather_models_multi_cluster_results():
    ifp = './models-multi-cluster'

    n_vals = [250, 1000, 4000]
    p_vals = [0.99, 0.98, 0.95, 0.9, 0.75]
    c_vals = [1, 2, 4, 8]
    keys_to_remove = ['pseudo_label_counts_per_class', 'pseudo_label_true_counts_per_class', 'pseudo_labeling_accuracy_per_class']

    for n in n_vals:
        df_list = list()
        avg_df_list = list()

        pre = 'only-supervised'
        m_fp = os.path.join(ifp, '{}-n{}-models'.format(pre, n))
        if not os.path.exists(m_fp):
            print('missing folder: {}'.format(m_fp))
            continue
        fns = [fn for fn in os.listdir(m_fp) if fn.startswith('id-')]
        acc_list = list()
        for fn in fns:
            if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                    stats_dict = json.load(json_file)
                for k in keys_to_remove:
                    if k in stats_dict:
                        del stats_dict[k]
                stats_dict['PL-method'] = pre
                stats_dict['N-SSL'] = n
                if 'test_softmax_accuracy' in stats_dict:
                    acc_list.append(stats_dict['test_softmax_accuracy'])
                cd = pd.json_normalize(stats_dict)
                df_list.append(cd)

        avg_stats_dict = dict()
        avg_stats_dict['PL-method'] = pre
        avg_stats_dict['N-SSL'] = n
        avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
        avg_stats_dict['std_accuracy'] = np.std(acc_list)
        avg_stats_dict['counts'] = len(acc_list)
        cd = pd.json_normalize(avg_stats_dict)
        avg_df_list.append(cd)


        for c in c_vals:

            prefixes = ['ssl-resp', 'ssl-neum', 'ssl-resp-cauchy', 'ssl-neum-cauchy']

            for pre in prefixes:
                m_fp = os.path.join(ifp, '{}-n{}-c{}-models'.format(pre, n, c))
                if not os.path.exists(m_fp):
                    print('missing folder: {}'.format(m_fp))
                    continue
                fns = [fn for fn in os.listdir(m_fp) if fn.startswith('id-')]
                acc_list = list()
                for fn in fns:
                    if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                        with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                            stats_dict = json.load(json_file)
                        for k in keys_to_remove:
                            if k in stats_dict:
                                del stats_dict[k]
                        stats_dict['PL-method'] = pre
                        stats_dict['N-SSL'] = n
                        stats_dict['num_clusters_per_class'] = c
                        if 'test_softmax_accuracy' in stats_dict:
                            acc_list.append(stats_dict['test_softmax_accuracy'])
                        cd = pd.json_normalize(stats_dict)
                        df_list.append(cd)

                avg_stats_dict = dict()
                avg_stats_dict['PL-method'] = pre
                avg_stats_dict['N-SSL'] = n
                avg_stats_dict['num_clusters_per_class'] = c
                avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
                avg_stats_dict['std_accuracy'] = np.std(acc_list)
                avg_stats_dict['counts'] = len(acc_list)
                cd = pd.json_normalize(avg_stats_dict)
                avg_df_list.append(cd)

            # gather percentile results
            for p in p_vals:
                m_fp = os.path.join(ifp, 'ssl-perc{}-n{}-c{}-models'.format(p, n, c))
                if not os.path.exists(m_fp):
                    print('missing folder: {}'.format(m_fp))
                    continue
                fns = [fn for fn in os.listdir(os.path.join(m_fp)) if fn.startswith('id-')]
                acc_list = list()
                for fn in fns:
                    if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                        with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                            stats_dict = json.load(json_file)
                        for k in keys_to_remove:
                            if k in stats_dict:
                                del stats_dict[k]
                        stats_dict['PL-method'] = 'percResp-sortNeum'
                        stats_dict['N-SSL'] = n
                        stats_dict['num_clusters_per_class'] = c
                        stats_dict['percentile'] = p
                        if 'test_softmax_accuracy' in stats_dict:
                            acc_list.append(stats_dict['test_softmax_accuracy'])
                        cd = pd.json_normalize(stats_dict)
                        df_list.append(cd)

                avg_stats_dict = dict()
                avg_stats_dict['PL-method'] = 'percResp-sortNeum'
                avg_stats_dict['N-SSL'] = n
                avg_stats_dict['num_clusters_per_class'] = c
                avg_stats_dict['percentile'] = p
                avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
                avg_stats_dict['std_accuracy'] = np.std(acc_list)
                avg_stats_dict['counts'] = len(acc_list)
                cd = pd.json_normalize(avg_stats_dict)
                avg_df_list.append(cd)

            for p in p_vals:
                m_fp = os.path.join(ifp, 'ssl-cauchy-perc{}-n{}-c{}-models'.format(p, n, c))
                if not os.path.exists(m_fp):
                    print('missing folder: {}'.format(m_fp))
                    continue
                fns = [fn for fn in os.listdir(os.path.join(m_fp)) if fn.startswith('id-')]
                acc_list = list()
                for fn in fns:
                    if os.path.exists(os.path.join(m_fp, fn)) and os.path.exists(os.path.join(m_fp, fn, 'stats.json')):
                        with open(os.path.join(m_fp, fn, 'stats.json')) as json_file:
                            stats_dict = json.load(json_file)
                        for k in keys_to_remove:
                            if k in stats_dict:
                                del stats_dict[k]
                        stats_dict['PL-method'] = 'percResp-sortNeum-cauchy'
                        stats_dict['N-SSL'] = n
                        stats_dict['num_clusters_per_class'] = c
                        stats_dict['percentile'] = p
                        if 'test_softmax_accuracy' in stats_dict:
                            acc_list.append(stats_dict['test_softmax_accuracy'])
                        cd = pd.json_normalize(stats_dict)
                        df_list.append(cd)
                avg_stats_dict = dict()
                avg_stats_dict['PL-method'] = 'percResp-sortNeum-cauchy'
                avg_stats_dict['N-SSL'] = n
                avg_stats_dict['num_clusters_per_class'] = c
                avg_stats_dict['percentile'] = p
                avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
                avg_stats_dict['std_accuracy'] = np.std(acc_list)
                avg_stats_dict['counts'] = len(acc_list)
                cd = pd.json_normalize(avg_stats_dict)
                avg_df_list.append(cd)

        full_df = pd.concat(df_list, axis=0)
        avg_full_df = pd.concat(avg_df_list, axis=0)
        cols = list(full_df.columns.values)

        cols_to_remove = ['training_wall_time', 'val_wall_time', 'num_epochs_trained', 'train_wall_time', 'test_wall_time', 'wall_time', 'gmm_build_wall_time', 'num_added_pseudo_labels', 'train_labeled_dataset_size', 'val_gmm_accuracy', 'val_cauchy_accuracy', 'test_gmm_accuracy', 'test_cauchy_accuracy', 'avg_pseudo_labeling_accuracy_per_class']
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

        full_df.to_csv(os.path.join(ifp, 'summary-{}.csv'.format(n)), index=False)
        avg_full_df.to_csv(os.path.join(ifp, 'summary-{}-avg.csv'.format(n)), index=False)


gather_models_multi_cluster_results()