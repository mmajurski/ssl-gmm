import argparse
import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])
import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import utils
import train





def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--arch', default='wide_resnet', type=str, help='Name of the image classification architecture to train')
    parser.add_argument('--num-workers', type=int, default=10, help='number of workers')
    parser.add_argument('--output-dirpath', default='./model', type=str, help='filepath to where the outputs will be saved.')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--tau', default=1.0, type=float, help='temperature value to sharpen the logits. Set to 1 to disable. If tau is 1, hard pseudo-labeling is used instead of soft pseudo-labeling.')
    parser.add_argument('--tau-method', type=str, default='fixmatch', help='what type of tau temp scaling to use, (fixmatch or mixmatch)')
    parser.add_argument('--mu', default=7, type=int, help='the number of unlabeled batches per labeled batch factor.')
    parser.add_argument('--eps', default=1e-3, type=float, help='eps value for determining early stopping metric equivalence.')
    parser.add_argument('--patience', default=50, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
    parser.add_argument('--embedding_dim', default=16, type=int, help='dimensionality of the embedding space.')
    parser.add_argument('--weight-decay', default=0.0005, type=float)  # 5e-4
    parser.add_argument('--cycle-factor', default=2.0, type=float, help='Cycle factor for cyclic learning rate scheduler.')
    parser.add_argument('--num-lr-reductions', default=2, type=int)
    parser.add_argument('--epoch-size', default=1024, help='the number of batches in an epoch', type=int)
    parser.add_argument('--lr-reduction-factor', default=0.2, type=float)
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--clip-grad', action='store_true', default=False)
    parser.add_argument('--pseudo-label-threshold', default=0.95, type=float, help='Threshold logits when filtering pseudo-labeling.')
    parser.add_argument('--pseudo-label-negative-threshold', default=0.1, type=float, help='Threshold logits when filtering negative pseudo-labeling. I.e. PL with logits < this will be included as negative examples.')
    parser.add_argument('--cosine-sim-pl-threshold', default=0.0, type=float, help='Threshold when filtering pseudo-labeling using cosine similarity. Only PL with a min CosineSim to the labeled embedding > x are used. Set to 0 to disable.')
    parser.add_argument('--cosine-sim-topk', default=6, type=float, help='Nearest N neighbors to consider when using cosine similarity to select psudo-labels.')

    parser.add_argument('--num-classes', default=10, type=int, help='number of classes in the dataset.')
    parser.add_argument('--num-labeled-datapoints', default=250, type=int, help='number of labeled annotations in the dataset.')
    parser.add_argument('--optimizer', type=str, default='sgd',help='optimizer if nothing is passed AdamW would be used (currently supported sgd,adamw)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--debug', help='enables debugging mode', action='store_true')
    parser.add_argument('--last-layer', type=str, default='fc', help='last layer to use in the NN (currently supported fc, kmeans, aa_gmm, aa_cmm, aa_gmmcmm, aa_gmm_d1, aa_cmm_d1, aa_gmmcmm_d1')
    parser.add_argument('--num-epochs', default=None, type=int, help='number of epochs to train. If this is non-None it will suppress the use of a test split, and blindly run the training for N epochs.')
    parser.add_argument('--trainer', type=str, default='fixmatch', help='trainer to use (currently supported supervised, fixmatch)')
    parser.add_argument('--embedding-constraint', type=str, default=None, help='embedding constraint to enforce (currently supported None, mean_covar, gauss_moment)')

    parser.add_argument('--seed', type=int, default=None, help='seed for the random number generator')
    parser.add_argument('--ood_p', type=float, default='0.0', help='percentage of unlabeled data drawn from cifar100 which is ood for cifar10')

    args = parser.parse_args()

    if args.seed is None or args.seed <= 0:
        args.seed = np.random.randint(0, pow(2,32) - 1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # check if IDE is in debug mode, and set the args debug flag and set num parallel worker to 0
    utils.check_for_ide_debug_mode(args)

    # handle setup or exit based on output directory existance and debug mode
    utils.validate_output_directory(args)

    try:
        train.train(args)
        with open(os.path.join(args.output_dirpath, 'success.txt'), mode='w', encoding='utf-8') as f:
            f.write('success')
        logging.shutdown()
        return 0
    except:
        with open(os.path.join(args.output_dirpath, 'failure.txt'), mode='w', encoding='utf-8') as f:
            f.write('failure')
        import traceback
        tb = traceback.format_exc()
        logging.info(tb)
        logging.shutdown()
        exit(1)  # pass failure back to caller


if __name__ == '__main__':
    main()

