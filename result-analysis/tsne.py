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



def compute_test_embedding(model_fp):
    a = os.path.join(model_fp, 'test_embedding.npy')
    b = os.path.join(model_fp, 'test_labels.npy')
    if os.path.exists(a) and os.path.exists(b):
        embedding_output_test = np.load(a)
        labels_output_test = np.load(b)
        return embedding_output_test, labels_output_test

    else:
        raise RuntimeError("Please build embedding and label .npy files first")
    #
    # print("Building test embedding")
    #
    # # load model
    # model = torch.load(os.path.join(model_fp, 'model_best.pt'))
    # model.cuda()
    # model.eval()
    #
    # # load config json file into dict
    # with open(os.path.join(model_fp, 'config.json'), 'r') as fh:
    #     args_dict = json.load(fh)
    # args = Namespace(**args_dict)
    #
    # # load data
    # test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)
    # test_dataset.load_data()
    #
    # model_trainer = trainer_fixmatch.FixMatchTrainer(args)
    # # setup the metadata capture object
    # train_stats = metadata.TrainingStats()
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # emb_constraint = None  # no need to use an embedding constraint for this
    # epoch = 0
    # embedding_output_test, labels_output_test = model_trainer.eval_model(model, test_dataset, criterion, train_stats, "test", emb_constraint, epoch, args, return_embedding=True)
    #
    # ofp = os.path.join(model_fp, 'test_embedding.npy')
    # np.save(ofp, embedding_output_test)
    # ofp = os.path.join(model_fp, 'test_labels.npy')
    # np.save(ofp, labels_output_test)
    #
    # return embedding_output_test, labels_output_test



def build_tsne_figure_cifar10(model_fp, epoch=None):
    model_fn = os.path.basename(model_fp)
    parent_fp = os.path.dirname(model_fp)
    if epoch is not None:
        ofp = os.path.join(parent_fp, "{}-tsne-{}.jpg".format(model_fn, epoch))
    else:
        ofp = os.path.join(parent_fp, "{}-tsne.jpg".format(model_fn, epoch))

    # with open(os.path.join(model_fp, 'stats.json'), 'r') as fh:
    #     stats_dict = json.load(fh)
    # if stats_dict['test_accuracy'] < 0.80:
    #     if os.path.exists(ofp):
    #         os.remove(ofp)
    #         return

    if os.path.exists(ofp):
        return
    print(model_fn)

    embedding_output_test, labels_output_test = compute_test_embedding(model_fp)

    # load model
    model = torch.load(os.path.join(model_fp, 'model_best.pth'))
    model = model['model']
    centers = None
    try:
        centers = model['module.last_layer.centers']
    except:
        pass


    matplotlib.rcParams.update({'font.size': 38})
    plt.rc('xtick', labelsize=26)
    plt.rc('ytick', labelsize=26)

    Z = embedding_output_test
    labels = labels_output_test
    tsne_obj = TSNE(n_components=2, random_state=42)
    if centers is None:
        Y = tsne_obj.fit_transform(Z)
    else:
        Z_centers = centers.detach().cpu().numpy()
        labels_centers = np.array(list(range(0, 10)))

        Z2 = np.zeros((Z.shape[0] + Z_centers.shape[0], Z_centers.shape[1]), dtype=np.float32)
        Z2[0:Z.shape[0]] = Z
        Z2[Z.shape[0]:] = Z_centers

        labels2 = np.zeros((labels.shape[0] + labels_centers.shape[0],), dtype=np.int32)
        labels2[0:labels.shape[0]] = labels
        labels2[labels.shape[0]:] = labels_centers

        Y2 = tsne_obj.fit_transform(Z2)
        Y = Y2[0:Z.shape[0]]
        Y_centers = Y2[Z.shape[0]:]

        a = Y_centers[:,0]
        md = 0
        for ia in range(len(a)):
            d = np.max(np.abs(a[ia] - a))
            if d > md:
                md = d
        if md < 2:
            Y_centers[:, 0] += 2.0 * (np.random.rand(Y_centers[:,0].shape[0]) - 0.5)
        a = Y_centers[:, 1]
        md = 0
        for ia in range(len(a)):
            d = np.max(np.abs(a[ia] - a))
            if d > md:
                md = d
        if md < 2:
            Y_centers[:, 1] += 2.0 * (np.random.rand(Y_centers[:,1].shape[0]) - 0.5)


    plt.figure(0, figsize=(12, 12), dpi=300)
    # plt.scatter(Y[:, 0], Y[:, 1], s=100, c=labels, alpha=0.15)
    plt.scatter(Y[:, 0], Y[:, 1], s=50, c=labels, alpha=0.25)

    if centers is not None:
        plt.scatter(Y_centers[:, 0], Y_centers[:, 1], s=400, c=labels_centers, marker="X", linewidths=2.5, edgecolors='black')
    toks = model_fn.split('_')
    ll = "Linear"
    if 'kmeans' in model_fn:
        ll = "KMeans"
    elif 'aagmm' in model_fn:
        ll = "AAGMM"
    emb_c = "None"
    e = int(toks[0][-1])
    if e == 0:
        emb_c = "None"
    elif e == 1:
        emb_c = "1st Order"
    elif e == 2:
        emb_c = "2nd Order"
    elif e == 3:
        emb_c = "3rd Order"
    elif e == 4:
        emb_c = "4th Order"
    plt.title("tSNE: {} + {}".format(ll, emb_c))

    plt.savefig(ofp)
    plt.cla()
    plt.clf()
    plt.close()

    print(' Success!')


if __name__ == '__main__':
    ifp = './models-tsne'
    fns = [fn for fn in os.listdir(ifp) if '_40_' in fn]
    fns = [fn for fn in fns if not fn.endswith('jpg')]
    fns.sort()

    for fn in fns:
        build_tsne_figure_cifar10(os.path.join(ifp, fn))