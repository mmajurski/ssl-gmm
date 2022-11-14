import os
import numpy as np
import pandas as pd
import json


def gather_models_results():
    # ifp = '/home/mmajursk/Downloads/strong-aug-results'
    ifp = '/home/mmajursk/Downloads/re-pl-results/'

    n_clusters = 1
    # n_vals = [250, 1000, 4000]
    n_vals = [250]
    # p_vals = [0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    p_vals = [0.8, 0.9, 0.95, 0.98, 0.99]
    methods = ["sort_resp", "sort_neum", "filter_resp_sort_numerator", "filter_resp_sort_resp", "filter_resp_percentile_sort_neum"]
    # methods = ["sort_resp", "sort_neum", "filter_resp_sort_numerator", "filter_resp_sort_resp"]
    inf_types = ["softmax", "gmm", "cauchy"]

    keys_to_remove = ['pseudo_label_counts_per_class', 'pseudo_label_true_counts_per_class', 'pseudo_labeling_accuracy_per_class', 'total_pseudo_label_true_counts_per_class', 'total_pseudo_label_counts_per_class', 'total_pseudo_label_true_percentage_per_class', 'total_pseudo_label_percentage_per_class', 'pseudo_label_percentage_per_class', 'pseudo_label_true_percentage_per_class', 'pseudo_labeling_accuracy', 'used_true_labels', 'used_pseudo_labels']

    # TODO concatenate all epochs together for PL accuracy, and figure out which method has the highest average PL accuracy  (which should result in the best models)

    for n in n_vals:
        df_list = list()
        avg_df_list = list()

        # gather the fully supervised results
        m_fp = os.path.join(ifp, '{}-n{}-models'.format("only-supervised", n))
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
                stats_dict['PL-method'] = "only-supervised"
                stats_dict['INF-method'] = "softmax"
                stats_dict['PL-thres'] = None
                stats_dict['N-SSL'] = n
                if stats_dict['epoch'] >=2000:
                    print(os.path.exists(os.path.join(m_fp, fn)))
                    raise RuntimeError("Model hit epoch limit")
                if 'test_softmax_accuracy' in stats_dict:
                    acc_list.append(stats_dict['test_softmax_accuracy'])
                cd = pd.json_normalize(stats_dict)
                df_list.append(cd)

        avg_stats_dict = dict()
        avg_stats_dict['PL-method'] = "only-supervised"
        avg_stats_dict['INF-method'] = "softmax"
        avg_stats_dict['PL-thres'] = None
        avg_stats_dict['N-SSL'] = n
        if len(acc_list) > 0:
            # avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
            # avg_stats_dict['std_accuracy'] = np.std(acc_list)
            avg_stats_dict['median_test_accuracy'] = np.median(acc_list)

            # acc_list.sort()
            # acc_list = np.asarray(acc_list)
            # q1 = acc_list[int(np.round(0.25*len(acc_list)))]
            # q3 = acc_list[int(np.round(0.75 * len(acc_list)))]
            # q2 = np.median(acc_list)
            # iqr = q3 - q1
            # lth = q2 - iqr
            # uth = q2 + iqr
            # acc_list = acc_list[acc_list >= lth]
            # acc_list = acc_list[acc_list <= uth]
            # avg_stats_dict['1.5iqr_mean_test_accuracy'] = np.mean(acc_list)
            # avg_stats_dict['1.5iqr_std_accuracy'] = np.std(acc_list)
        avg_stats_dict['counts'] = len(acc_list)
        cd = pd.json_normalize(avg_stats_dict)
        avg_df_list.append(cd)


        for m in methods:
            for inf in inf_types:
                if inf == "softmax" and ("neum" in m or "numerator" in m):
                    continue

                lcl_p_vals = [None]
                if "filter" in m:
                    lcl_p_vals = p_vals

                for p in lcl_p_vals:
                    if p is None:
                        m_fp = "ssl-{}-method{}-n{}-c{}-models".format(inf, m, n, n_clusters)
                    else:
                        m_fp = "ssl-{}-method{}-thres{}-n{}-c{}-models".format(inf, m, p, n, n_clusters)

                    m_fp = os.path.join(ifp, m_fp)
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
                            stats_dict['PL-method'] = m
                            stats_dict['INF-method'] = inf
                            stats_dict['PL-thres'] = p
                            stats_dict['N-SSL'] = n
                            if stats_dict['epoch'] >= 2000:
                                print(os.path.exists(os.path.join(m_fp, fn)))
                                raise RuntimeError("Model hit epoch limit")

                            if 'test_softmax_accuracy' in stats_dict:
                                acc_list.append(stats_dict['test_softmax_accuracy'])
                            cd = pd.json_normalize(stats_dict)
                            df_list.append(cd)

                    avg_stats_dict = dict()
                    avg_stats_dict['PL-method'] = m
                    avg_stats_dict['INF-method'] = inf
                    avg_stats_dict['PL-thres'] = p
                    avg_stats_dict['N-SSL'] = n
                    if len(acc_list) > 0:
                        # avg_stats_dict['mean_test_accuracy'] = np.mean(acc_list)
                        # avg_stats_dict['std_accuracy'] = np.std(acc_list)
                        avg_stats_dict['median_test_accuracy'] = np.median(acc_list)

                        # acc_list.sort()
                        # acc_list = np.asarray(acc_list)
                        # q1 = acc_list[int(np.round(0.25 * len(acc_list)))]
                        # q3 = acc_list[int(np.round(0.75 * len(acc_list)))]
                        # q2 = np.median(acc_list)
                        # iqr = q3 - q1
                        # lth = q2 - iqr
                        # uth = q2 + iqr
                        # acc_list = acc_list[acc_list >= lth]
                        # acc_list = acc_list[acc_list <= uth]
                        # avg_stats_dict['1.5iqr_mean_test_accuracy'] = np.mean(acc_list)
                        # avg_stats_dict['1.5iqr_std_accuracy'] = np.std(acc_list)
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
        cols.remove('INF-method')
        cols.remove('N-SSL')
        cols.remove('PL-thres')

        cols.insert(0, 'PL-method')
        cols.insert(1, 'INF-method')
        cols.insert(2, 'N-SSL')
        cols.insert(3, 'PL-thres')
        full_df = full_df[cols]


        full_df.to_csv(os.path.join(ifp, 'summary-{}.csv'.format(n)), index=False)
        avg_full_df.to_csv(os.path.join(ifp, 'summary-{}-avg.csv'.format(n)), index=False)


        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(8, 4), dpi=200)
        x = full_df['test_softmax_accuracy'].to_numpy()
        y = full_df['INF-method'].to_numpy()
        x2 = list()
        for v in np.unique(y):
            idx = y == v
            x3 = list()
            x3.extend(x[idx])
            x2.append(x3)
        plt.hist(x2, bins=50, stacked=True, density=True)
        plt.legend(np.unique(y))
        plt.savefig('test_accuracy_hist_{}.png'.format(n))





gather_models_results()