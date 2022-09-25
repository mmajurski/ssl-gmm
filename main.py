import argparse
import os

import train
import train_ssl


def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
    parser.add_argument('--arch', default='wide_resnet', type=str, help='Name of the image classification architecture to train')
    parser.add_argument('--num-workers', type=int, default=3, help='number of workers')
    parser.add_argument('--output-filepath', default='./model', type=str, help='filepath to where the outputs will be saved.')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--loss_eps', default=1e-4, type=float, help='loss value eps for determining early stopping loss equivalence.')
    parser.add_argument('--val_fraction', default=0.1, type=float, help='fraction of the training data to use for validation.')
    parser.add_argument('--patience', default=50, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--cycle_factor', default=4.0, type=float, help='Cycle factor for cyclic learning rate scheduler.')
    parser.add_argument('--num_lr_reductions', default=2, type=int)
    parser.add_argument('--lr_reduction_factor', default=0.2, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--inference-method', default='gmm', type=str, help='whether to use gmm or cauchy for inference.')
    parser.add_argument('--re-pseudo-label-each-epoch', action='store_true')
    parser.add_argument('--disable-ssl', action='store_true')
    parser.add_argument('--pseudo-label-percentile-threshold', default=0.99)
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in the dataset.')
    parser.add_argument('--num_labeled_datapoints', default=250, type=int, help='number of labeled annotations in the dataset.')
    parser.add_argument('--starting_model', type=str, default=None, help='Pytorch model checkpoint to load instead of starting from random')
    parser.add_argument('--optimizer', type=str, default='adamw',help='optimizer if nothing is passed AdamW would be used (currently supported sgd,adamw)')
    parser.add_argument('--cluster_per_class', default=2, type=int, help='number of clusters to create per class')
    args = parser.parse_args()

    if os.path.exists(args.output_filepath):
        return

    if not os.path.exists(args.output_filepath):
        os.makedirs(args.output_filepath)
    if args.debug:
        args.num_workers = 0

    # import sys
    # gettrace = getattr(sys, 'gettrace', None)
    # if gettrace():
    #     print("Detected debug mode, setting number of workers to 0")
    #     args.num_workers = 0

    # train.train(args)
    train_ssl.train(args)


if __name__ == '__main__':
    main()
