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

import utils
import train





def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--arch', default='wide_resnet', type=str, help='Name of the image classification architecture to train')
    parser.add_argument('--num-workers', type=int, default=12, help='number of workers')
    parser.add_argument('--output-dirpath', default='./model', type=str, help='filepath to where the outputs will be saved.')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='initial learning rate')  # 3e-4
    parser.add_argument('--tau', default=1.0, type=float, help='temperature value to sharpen the logits. Set to 1 to disable. If tau is 1, hard pseudo-labeling is used instead of soft pseudo-labeling.')
    parser.add_argument('--tau-method', type=str, default='fixmatch', help='what type of tau temp scaling to use, (fixmatch or mixmatch)')
    parser.add_argument('--mu', default=7, type=int, help='the number of unlabeled batches per labeled batch factor.')
    parser.add_argument('--loss-eps', default=1e-4, type=float, help='loss value eps for determining early stopping loss equivalence.')
    parser.add_argument('--patience', default=20, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
    parser.add_argument('--embedding_dim', default=16, type=int, help='dimensionality of the embedding space.')
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--cycle-factor', default=2.0, type=float, help='Cycle factor for cyclic learning rate scheduler.')
    parser.add_argument('--num-lr-reductions', default=2, type=int)
    parser.add_argument('--epoch-size', default=1024, help='the number of batches in an epoch', type=int)
    parser.add_argument('--lr-reduction-factor', default=0.2, type=float)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--pseudo-label-threshold', default=0.95, type=float, help='Threshold when filtering pseudo-labeling.')
    parser.add_argument('--num-classes', default=10, type=int, help='number of classes in the dataset.')
    parser.add_argument('--num-labeled-datapoints', default=250, type=int, help='number of labeled annotations in the dataset.')
    parser.add_argument('--optimizer', type=str, default='sgd',help='optimizer if nothing is passed AdamW would be used (currently supported sgd,adamw)')
    parser.add_argument('--strong-augmentation', help='enables strong augmentation', action='store_true')
    parser.add_argument('--debug', help='enables debugging mode', action='store_true')
    parser.add_argument('--last-layer', type=str, default='fc', help='last layer to use in the NN (currently supported fc, kmeans, aa_gmm, aa_cmm, aa_gmmcmm, aa_gmm_d1, aa_cmm_d1, aa_gmmcmm_d1')
    parser.add_argument('--nprefc', type=int, default=0, help='number of pre-fc linear layers')
    parser.add_argument('--num-epochs', default=None, type=int, help='number of epochs to train. If this is non-None it will suppress the use of a test split, and blindly run the training for N epochs.')
    parser.add_argument('--trainer', type=str, default='fixmatch', help='trainer to use (currently supported supervised, fixmatch)')
    parser.add_argument('--embedding-constraint', type=str, default=None, help='embedding constraint to enforce (currently supported None, mean_covar, gauss_moment)')




    args = parser.parse_args()

    # check if IDE is in debug mode, and set the args debug flag and set num parallel worker to 0
    utils.check_for_ide_debug_mode(args)

    # handle setup or exit based on output directory existance and debug mode
    utils.validate_output_directory(args)

    try:
        train.train(args)
        logging.shutdown()
        return 0
    except:
        import traceback
        tb = traceback.format_exc()
        logging.info(tb)
        logging.shutdown()
        return 1


if __name__ == '__main__':
    main()

