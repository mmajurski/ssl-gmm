import argparse
import os

import train


def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
    parser.add_argument('--arch', default='resnet18', type=str, help='Name of the image classification architecture to train')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--output-filepath', default='./model', type=str, help='filepath to where the outputs will be saved.')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--loss_eps', default=1e-3, type=float, help='loss value eps for determining early stopping loss equivalence.')
    parser.add_argument('--val_fraction', default=0.05, type=float, help='fraction of the training data to use for validation.')
    parser.add_argument('--patience', default=10, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
    parser.add_argument('--cycle_factor', default=4.0, type=float, help='Cycle factor for cyclic learning rate scheduler.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--starting_model', type=str, default=None, help='Pytorch model checkpoint to load instead of starting from random')

    args = parser.parse_args()
    if not os.path.exists(args.output_filepath):
        os.makedirs(args.output_filepath)
    if args.debug:
        args.num_workers = 0

    # import sys
    # gettrace = getattr(sys, 'gettrace', None)
    # if gettrace():
    #     print("Detected debug mode, setting number of workers to 0")
    #     args.num_workers = 0

    train.train(args)


if __name__ == '__main__':
    main()
