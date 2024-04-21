import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import random
from sklearn.manifold import TSNE
matplotlib.use('agg')

import cifar_datasets
import trainer_fixmatch
import json
from argparse import Namespace
import metadata
import embedding_constraints



def get_test_embedding(model_fp):
    a = os.path.join(model_fp, 'test_embedding.npy')
    b = os.path.join(model_fp, 'test_labels.npy')
    if os.path.exists(a) and os.path.exists(b):
        embedding_output_test = np.load(a)
        labels_output_test = np.load(b)
        return embedding_output_test, labels_output_test

    else:
        raise RuntimeError("Please build embedding and label .npy files first")



def worker(model_fp):
    model_fn = os.path.basename(model_fp)
    parent_fp = os.path.dirname(model_fp)

    print(model_fn)

    embedding_output_test, labels_output_test = get_test_embedding(model_fp)

    # load model
    model = torch.load(os.path.join(model_fp, 'model_best.pth'))
    model = model['model']
    centers = None
    try:
        centers = model['module.last_layer.centers']
    except:
        pass
    if centers is None:
        return None, None

    centers = centers.detach().cpu().numpy()

    dist_to_center = dict()
    avg_dist = 0
    for c in np.unique(labels_output_test):
        cent = centers[c]
        emb_dat = embedding_output_test[labels_output_test == c]
        delta = emb_dat - cent
        d = np.sqrt(np.sum(delta**2, axis=1))
        # print(f"Class {c} mean dist to center: {np.mean(d)}")
        dist_to_center[c] = np.mean(d)
        avg_dist += np.mean(d)
    avg_dist /= len(np.unique(labels_output_test))

    # print(dist_to_center)
    return dist_to_center, avg_dist




if __name__ == '__main__':
    ifp = './models-tsne'
    fns = [fn for fn in os.listdir(ifp) if '_40_' in fn]
    fns = [fn for fn in fns if not fn.endswith('jpg')]
    fns.sort()

    # create pandas dataframe to store the distance data in
    import pandas as pd
    df = pd.DataFrame(columns=['model', 'emd_constraint', 'avg_dist'])

    tmp_dict = dict()

    for fn in fns:
        dist_to_center, avg_dist = worker(os.path.join(ifp, fn))
        if dist_to_center is None:
            continue
        dist_to_center['model'] = fn
        dist_to_center['avg_dist'] = avg_dist
        toks = fn.split('_')
        method = int(toks[0][-1])
        dist_to_center['emd_constraint'] = method
        df = df.append(dist_to_center, ignore_index=True)

        toks = fn.split('_')
        del toks[3]
        fn2 = '_'.join(toks)
        dist_to_center['model'] = fn2
        dist_to_center['count'] = 1
        if fn2 in tmp_dict.keys():
            ddat = tmp_dict[fn2]
            for k in ddat.keys():
                if k == 'model':
                    continue
                ddat[k] += dist_to_center[k]
        else:
            tmp_dict[fn2] = dist_to_center

    df2 = pd.DataFrame(columns=['model', 'emd_constraint', 'avg_dist'])
    for k in tmp_dict.keys():
        ddat = tmp_dict[k]
        for k in ddat.keys():
            if k == 'model' or k == 'count':
                continue
            ddat[k] /= float(ddat['count'])
        df2 = df2.append(ddat, ignore_index=True)


    df.to_csv('./dist_to_center_all_data.csv')
    df2.to_csv('./dist_to_center.csv')