import os
import time
import copy
import numpy as np
import torch
import torchvision
import json
import logging
import sklearn.mixture

import cifar_datasets
import metadata
from gmm_module import GMM
from skl_cauchy_mm import GMM_SKL
import lr_scheduler
import flavored_resnet18
import flavored_wideresnet
import fixmatch_augmentation


MAX_EPOCHS = 2000
CACHE_FULLY_SUPERVISED_MODEL = False

logger = logging.getLogger()


def setup(args):
    # load stock models from https://pytorch.org/vision/stable/models.html
    model = None
    if args.starting_model is not None:
        # warning, this over rides the args.arch selection
        model = torch.load(args.starting_model)
    else:
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
            #model = flavored_resnets.ResNet18(num_classes=args.num_classes)
        if args.arch == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)
        if args.arch == 'resnext50_32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=args.num_classes)
        if args.arch == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=args.num_classes)
        if args.arch == 'wide_resnet':
            model = flavored_wideresnet.build_wideresnet(num_classes=args.num_classes)

    if model is None:
        raise RuntimeError("Unsupported model architecture selection: {}.".format(args.arch))

    logger.info("Total Model params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # setup and load CIFAR10
    if args.num_classes == 10:
        if args.strong_augmentation:
            train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_STRONG_TRAIN, train=True, subset=args.debug)
        else:
            train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TRAIN, train=True, subset=args.debug)
    else:
        raise RuntimeError("unsupported class count: {}".format(args.num_classes))

    train_dataset, val_dataset = train_dataset.train_val_split(val_fraction=args.val_fraction)
    # set the validation augmentation to just normalize (.dataset since val_dataset is a Subset, not a full dataset)
    val_dataset.set_transforms(cifar_datasets.Cifar10.TRANSFORM_TEST)

    train_dataset_labeled, train_dataset_unlabeled = train_dataset.data_split_class_balanced(subset_count=args.num_labeled_datapoints)

    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    return model, train_dataset_labeled, train_dataset_unlabeled, val_dataset, test_dataset


def psuedolabel_data(model, train_dataset_labeled, train_dataset_unlabeled, gmm, epoch, train_stats, top_k, args):

    ul_dataloader = torch.utils.data.DataLoader(train_dataset_unlabeled,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                worker_init_fn=cifar_datasets.worker_init_fn)
    # # TODO update this to use a per-class threshold, but that is an extension of the baseline technique
    # denominator_threshold = 0.0
    # # denominator_threshold = 500.0
    # resp_threshold = 0.95
    model.eval()
    filtered_denominators = []
    filtered_labels = []
    filtered_indicies = []
    filtered_data_resp = []
    filtered_data_weighted_prob = []

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(ul_dataloader):
            inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()
            index = tensor_dict[2].cpu().detach()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                gmm_inputs = outputs.detach().cpu()

                if args.inference_method == 'gmm':
                    gmm_prob_weighted, gmm_prob_sum, gmm_resp = gmm.predict_probability(gmm_inputs)
                elif args.inference_method == 'cauchy':
                    gmm_prob_weighted, gmm_prob_sum, gmm_resp = gmm.predict_cauchy_probability(gmm_inputs)
                elif args.inference_method == 'softmax':
                    gmm_prob_weighted, gmm_prob_sum, gmm_resp = gmm_inputs, torch.sum(gmm_inputs, dim=1), gmm_inputs
                else:
                    raise RuntimeError("Invalid PL - inference method: {}. Expected gmm or cauchy or softmax".format(args.inference_method))
                # threshold filtering

                # list of boolean where value is False for items having lower prob_sum then threshold, True otherwise
                # denominator_filter = torch.squeeze(gmm_prob_sum > denominator_threshold)
                filtered_denominators.append(gmm_prob_sum)
                filtered_labels.append(labels.cpu().detach())
                filtered_indicies.append(index)
                filtered_data_resp.append(gmm_resp)
                filtered_data_weighted_prob.append(gmm_prob_weighted)

    filtered_denominators = torch.cat(filtered_denominators, dim=0)
    filtered_labels = torch.cat(filtered_labels, dim=-1)
    filtered_indicies = torch.cat(filtered_indicies, dim=-1)
    filtered_data_resp = torch.cat(filtered_data_resp, dim=0)
    filtered_data_weighted_prob = torch.cat(filtered_data_weighted_prob, dim=0)


    # filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob = pseudo_label_denominator_filter(filtered_denominators, filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob)
    # filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob = pseudo_label_denominator_filter_1perc(filtered_denominators, filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob)

    # DONE: RUSHABH | create a set of subfunctions which sorts/filters the gmm outputs using different methods. So it takes as input a list of all the GMM outputs, and then it returns a sorted list (potentially smaller if there is a filter). This subfunction allows us to swap out different pseudo-labeling strategies.
    # DONE: RUSHABH | i.e. labels, indicies, data_resp = pseudo_label_filter_denominator(labels, indicies, data_resp)
    # DONE: RUSHABH | i.e. labels, indicies, data_resp = pseudo_label_filter_numerator(labels, indicies, data_resp)

    # TODO start figuring out what the filtering and disqualifying filters are for the GMM to remove bad examples
    # DONE: JD | Experiment with pseudo labeling sorting by the largest numerator per class, then do top k

    # filtered_labels, filtered_indicies, filtered_preds = pseudo_label_numerator_filter_1perc(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, weighted=True)

    if args.inference_method == 'softmax':
        args.pseudo_label_method = 'sort_resp'

    if args.pseudo_label_method == "sort_resp":
        filtered_labels, filtered_indicies, filtered_preds = pseudo_label_sort_resp(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, cluster_per_class=args.cluster_per_class)

    elif args.pseudo_label_method == "sort_neum":
        filtered_labels, filtered_indicies, filtered_preds = pseudo_label_sort_neum(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, cluster_per_class=args.cluster_per_class)

    elif args.pseudo_label_method == "filter_resp_sort_numerator":
        # keep only resp > threshold, and then sort based on neumerator
        filtered_labels, filtered_indicies, filtered_preds = pseudo_label_filter_resp_sort_numerator(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, thres=float(args.pseudo_label_threshold), cluster_per_class=args.cluster_per_class)

    elif args.pseudo_label_method == "filter_resp_sort_resp":
        # keep only resp > threshold, and then sort based on neumerator
        filtered_labels, filtered_indicies, filtered_preds = pseudo_label_filter_resp_sort_resp(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, thres=float(args.pseudo_label_threshold), cluster_per_class=args.cluster_per_class)

    elif args.pseudo_label_method == "filter_resp_percentile_sort_neum":
        # take to 1% of the resp, and then sort based on neumerator
        filtered_labels, filtered_indicies, filtered_preds = pseudo_label_filter_resp_percentile_sort_neum(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, perc=float(args.pseudo_label_threshold), cluster_per_class=args.cluster_per_class)

    else:
        raise RuntimeError("Unexpected pseudo-label selection algorithm: {}".format(args.pseudo_label_method))

    if epoch % 50 == 0:
        from matplotlib import pyplot as plt
        if args.pseudo_label_method == "sort_resp" or args.pseudo_label_method == "sort_neum":
            acc = (filtered_labels == filtered_preds)
            max_resp, _ = torch.max(filtered_data_resp, dim=-1)
            max_resp = max_resp.detach().cpu().numpy()

            plt.figure(figsize=(8, 6))
            plt.scatter(max_resp, acc, alpha=0.01)
            plt.title("Max Resp vs Pseudo-Label Accuracy")
            plt.ylabel('Pseudo-Label Accuracy')
            plt.xlabel('Max Resp')
            plt.savefig(os.path.join(args.output_dirpath, 'max-resp-vs-pl-acc-epoch{:03d}.png'.format(epoch)))
            plt.close()

        if args.pseudo_label_method == "filter_resp_percentile_sort_neum":
            perc = float(args.pseudo_label_threshold)
            max_resp, _ = torch.max(filtered_data_resp, dim=-1)
            sorted_resp, _ = max_resp.sort(descending=False)
            idx = int(len(sorted_resp) * perc)
            threshold = float(sorted_resp[idx])

            vals = max_resp.detach().cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.hist(vals, bins=100, label='max_resp')
            plt.yscale("log")
            plt.title("Max Resp Hist ({}th percentile = {:.16g})".format(perc, threshold))
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(args.output_dirpath, 'max-resp-hist-epoch{:03d}.png'.format(epoch)))
            plt.close()

            vals2, _ = filtered_data_resp.reshape(-1).sort(descending=False)
            vals2 = vals2.detach().cpu().numpy()
            idx = int(len(vals2) * perc)
            threshold = float(vals2[idx])
            plt.figure(figsize=(8, 6))
            plt.hist(vals2, bins=100, label='resp')
            plt.yscale("log")
            plt.title("Resp Hist ({}th percentile = {:.16g})".format(perc, threshold))
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(args.output_dirpath, 'resp-hist-epoch{:03d}.png'.format(epoch)))
            plt.close()

            max_neum, _ = torch.max(filtered_data_weighted_prob, dim=-1)
            sorted_neum, _ = max_neum.sort(descending=False)
            idx = int(len(sorted_neum) * perc)
            threshold = float(sorted_neum[idx])
            vals = max_neum.detach().cpu().numpy()

            plt.figure(figsize=(8, 6))
            plt.hist(vals, bins=100, label='max_neum')
            plt.yscale("log")
            plt.title("Max Numerator Hist ({}th percentile = {:.16g})".format(perc, threshold))
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(args.output_dirpath, 'max-neum-hist-epoch{:03d}.png'.format(epoch)))
            plt.close()

            vals2, _ = filtered_data_weighted_prob.reshape(-1).sort(descending=False)
            vals2 = vals2.detach().cpu().numpy()
            idx = int(len(vals2) * perc)
            threshold = float(vals2[idx])
            plt.figure(figsize=(8, 6))
            plt.hist(vals2, bins=100, label='neum')
            plt.yscale("log")
            plt.title("Numerator Hist ({}th percentile = {:.16g})".format(perc, threshold))
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(args.output_dirpath, 'neum-hist-epoch{:03d}.png'.format(epoch)))
            plt.close()

    # ratio based selection
    # filtered_labels, filtered_indicies, filtered_preds = pseudo_label_numerator_filter(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, weighted=True, thres=0.0)
    # # numerator based selection
    # filtered_labels, filtered_indicies, filtered_preds = pseudo_label_numerator_filter(filtered_labels, filtered_indicies, filtered_data_resp, filtered_data_weighted_prob, weighted=False, thres=0.0)

    class_counter = [0] * int(args.num_classes)
    true_class_counter = [0] * int(args.num_classes)
    class_accuracy = [0] * int(args.num_classes)
    used_indices = []
    used_filtered_indices = []

    for i in range(len(filtered_preds)):
        pred = filtered_preds[i]
        # assumption: 10 classes are 0 to 10
        # if class_counter[pred] < points_per_class:
        if class_counter[pred] < top_k:

            class_counter[pred] += 1
            label = filtered_labels[i]
            index = filtered_indicies[i]
            used_indices.append(index)
            used_filtered_indices.append(i)

            img, target = train_dataset_unlabeled.get_raw_datapoint(index)
            # add the predicted pseudo-label to the training dataset
            train_dataset_labeled.add_datapoint(img, pred)

            acc = float(pred == label)
            class_accuracy[target] += acc
            true_class_counter[target] += 1

    if len(used_indices) == 0:
        logger.warning("Pseudo-labeler found no examples to add to the training dataset")
        train_stats.add(epoch, 'num_added_pseudo_labels', int(np.sum(class_counter)))
    else:
        # delete the indices transferred to the labeled population. Do this after the move, so that we don't modify the index numbers during the move
        train_dataset_unlabeled.remove_datapoints_by_index(used_indices)

        used_true_labels = filtered_labels[used_filtered_indices]
        used_pseudo_labels = filtered_preds[used_filtered_indices]

        for i in range(len(class_accuracy)):
            if true_class_counter[i] == 0:
                class_accuracy[i] = np.nan
            else:
                class_accuracy[i] /= true_class_counter[i]

        cur_added_true_class_counts = train_stats.get_epoch('total_pseudo_label_true_counts_per_class', epoch - 1)
        cur_added_pred_class_counts = train_stats.get_epoch('total_pseudo_label_counts_per_class', epoch - 1)

        if cur_added_true_class_counts is None:
            train_stats.add(epoch, 'total_pseudo_label_true_counts_per_class', true_class_counter)
        else:
            vals = np.add(cur_added_true_class_counts, true_class_counter)  # elementwise sum
            train_stats.add(epoch, 'total_pseudo_label_true_counts_per_class', vals.tolist())

        if cur_added_pred_class_counts is None:
            train_stats.add(epoch, 'total_pseudo_label_counts_per_class', class_counter)
        else:
            vals = np.add(cur_added_pred_class_counts, class_counter)  # elementwise sum
            train_stats.add(epoch, 'total_pseudo_label_counts_per_class', vals.tolist())

        vals = train_stats.get_epoch('total_pseudo_label_true_counts_per_class', epoch)
        vals = vals / np.sum(vals)
        train_stats.add(epoch, 'total_pseudo_label_true_percentage_per_class', vals.tolist())

        vals = train_stats.get_epoch('total_pseudo_label_counts_per_class', epoch)
        vals = vals / np.sum(vals)
        train_stats.add(epoch, 'total_pseudo_label_percentage_per_class', vals.tolist())


        # update the training metadata
        train_stats.add(epoch, 'pseudo_label_counts_per_class', class_counter)
        train_stats.add(epoch, 'pseudo_label_true_counts_per_class', true_class_counter)
        train_stats.add(epoch, 'pseudo_labeling_accuracy_per_class', class_accuracy)

        vals = np.asarray(class_counter) / np.sum(class_counter)
        train_stats.add(epoch, 'pseudo_label_percentage_per_class', vals.tolist())
        vals = np.asarray(true_class_counter) / np.sum(true_class_counter)
        train_stats.add(epoch, 'pseudo_label_true_percentage_per_class', vals.tolist())
        # get the average accuracy of the pseudo-labels (this data is not available in real SSL applications, since the unlabeled population would truly be unlabeled
        train_stats.add(epoch, 'pseudo_labeling_accuracy', float(np.nanmean(class_accuracy)))
        train_stats.add(epoch, 'num_added_pseudo_labels', int(np.sum(class_counter)))
        train_stats.add(epoch, 'used_true_labels', int(np.sum(used_true_labels)))
        train_stats.add(epoch, 'used_pseudo_labels', int(np.sum(used_pseudo_labels)))

        #train_stats.render_and_save_confusion_matrix(used_true_labels, used_pseudo_labels, args.output_filepath, 'pseudo_labeling_confusion_matrix', epoch)


# def pseudo_label_denominator_filter(denominators, labels, indices, data_resp, data_weighted_probs):
#     denominator_threshold = 0.0
#     denominator_filter = torch.squeeze(denominators > denominator_threshold)
#     labels = labels[denominator_filter]
#     data_resp = data_resp[denominator_filter]
#     indices = indices[denominator_filter]
#     data_weighted_probs = data_weighted_probs[denominator_filter]
#
#     return labels, indices, data_resp, data_weighted_probs


def pseudo_label_filter_resp_percentile_sort_neum(labels, indices, resp, neumerator, perc, cluster_per_class=1):

    max_resp, _ = torch.max(resp, dim=-1)

    sorted_resp, _ = max_resp.sort(descending=False)

    for p in range(50, 100):
        p = (float(p) / 100.0)
        idx = int(len(sorted_resp) * p)
        t = float(sorted_resp[idx])
        logger.info("{}th Percentile Threshold = {:.16g}".format(p, t))

    idx = int(len(sorted_resp) * perc)
    threshold = float(sorted_resp[idx])
    filter = torch.squeeze(max_resp >= threshold)
    logger.info("Percentile Threshold ({}th) of resp values for PL = {:.16g}".format(perc, threshold))

    neumerator = neumerator[filter]
    labels = labels[filter]
    indices = indices[filter]

    max_neum, preds = torch.max(neumerator, dim=-1)
    preds = torch.div(preds, cluster_per_class, rounding_mode='floor')
    sorted_neum, sorted_neum_idx = max_neum.sort(descending=True)

    labels = labels[sorted_neum_idx].detach().cpu().numpy()
    indices = indices[sorted_neum_idx].detach().cpu().numpy()
    preds = preds[sorted_neum_idx].detach().cpu().numpy()

    return labels, indices, preds
#
#
# def pseudo_label_numerator_filter(labels, indices, data_resp, data_weighted_probs, weighted=True, thres=None,cluster_per_class=1):
#     if thres is None:
#         numerator_resp_threshold = 0.95
#     else:
#         numerator_resp_threshold = thres
#
#     data = data_resp if not weighted else data_weighted_probs
#     max_numerator, preds = torch.max(data, dim=-1)
#     preds = torch.div(preds, cluster_per_class, rounding_mode='floor')
#     sorted_numerators, max_sorted_indices = max_numerator.sort(descending=True)
#
#     if not weighted:
#         max_sorted_indices = max_sorted_indices[sorted_numerators > numerator_resp_threshold]
#
#     labels = labels[max_sorted_indices].detach().cpu().numpy()
#     indices = indices[max_sorted_indices].detach().cpu().numpy()
#     preds = preds[max_sorted_indices].detach().cpu().numpy()
#
#     return labels, indices, preds


def pseudo_label_sort_resp(labels, indices, resp, neumerator, cluster_per_class=1):

    max_resp, _ = torch.max(resp, dim=-1)

    _, preds = torch.max(neumerator, dim=-1)
    preds = torch.div(preds, cluster_per_class, rounding_mode='floor')
    _, sorted_idx = max_resp.sort(descending=True)

    labels = labels[sorted_idx].detach().cpu().numpy()
    indices = indices[sorted_idx].detach().cpu().numpy()
    preds = preds[sorted_idx].detach().cpu().numpy()

    return labels, indices, preds


def pseudo_label_sort_neum(labels, indices, resp, neumerator, cluster_per_class=1):

    max_resp, _ = torch.max(resp, dim=-1)

    max_neum, preds = torch.max(neumerator, dim=-1)
    preds = torch.div(preds, cluster_per_class, rounding_mode='floor')
    _, sorted_idx = max_neum.sort(descending=True)

    labels = labels[sorted_idx].detach().cpu().numpy()
    indices = indices[sorted_idx].detach().cpu().numpy()
    preds = preds[sorted_idx].detach().cpu().numpy()

    return labels, indices, preds


def pseudo_label_filter_resp_sort_numerator(labels, indices, resp, neumerator, thres, cluster_per_class=1):
    max_resp, _ = torch.max(resp, dim=-1)
    filter = torch.squeeze(max_resp >= thres)

    neumerator = neumerator[filter]
    labels = labels[filter]
    indices = indices[filter]

    max_neum, preds = torch.max(neumerator, dim=-1)
    preds = torch.div(preds, cluster_per_class, rounding_mode='floor')
    _, sorted_idx = max_neum.sort(descending=True)

    labels = labels[sorted_idx].detach().cpu().numpy()
    indices = indices[sorted_idx].detach().cpu().numpy()
    preds = preds[sorted_idx].detach().cpu().numpy()

    return labels, indices, preds


def pseudo_label_filter_resp_sort_resp(labels, indices, resp, neumerator, thres, cluster_per_class=1):
    max_resp, _ = torch.max(resp, dim=-1)
    filter = torch.squeeze(max_resp >= thres)

    neumerator = neumerator[filter]
    labels = labels[filter]
    indices = indices[filter]
    max_resp = max_resp[filter]

    _, preds = torch.max(neumerator, dim=-1)
    preds = torch.div(preds, cluster_per_class, rounding_mode='floor')
    _, sorted_idx = max_resp.sort(descending=True)

    labels = labels[sorted_idx].detach().cpu().numpy()
    indices = indices[sorted_idx].detach().cpu().numpy()
    preds = preds[sorted_idx].detach().cpu().numpy()

    return labels, indices, preds


