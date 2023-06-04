import argparse
import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

import utils
import train





def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--arch', default='wide_resnet', type=str, help='Name of the image classification architecture to train')
    parser.add_argument('--num-workers', type=int, default=12, help='number of workers')
    parser.add_argument('--output-dirpath', default='./model', type=str, help='filepath to where the outputs will be saved.')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--learning-rate', default=0.03, type=float, help='initial learning rate')  # 3e-4
    parser.add_argument('--tau', default=1.0, type=float, help='temperature value to sharpen the logits. Set to 1 to disable. If tau is 1, hard pseudo-labeling is used instead of soft pseudo-labeling.')
    parser.add_argument('--tau-method', type=str, default='fixmatch', help='what type of tau temp scaling to use, (fixmatch or mixmatch)')
    parser.add_argument('--mu', default=7, type=int, help='the number of unlabeled batches per labeled batch factor.')
    parser.add_argument('--loss-eps', default=1e-4, type=float, help='loss value eps for determining early stopping loss equivalence.')
    # parser.add_argument('--val-fraction', default=0.1, type=float, help='fraction of the training data to use for validation.')
    parser.add_argument('--patience', default=50, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--cycle-factor', default=2.0, type=float, help='Cycle factor for cyclic learning rate scheduler.')
    parser.add_argument('--num-lr-reductions', default=2, type=int)
    parser.add_argument('--nb-reps', default=128, help='the number of reps through the labeled portion of the dataset to form an epoch', type=int)
    parser.add_argument('--lr-reduction-factor', default=0.2, type=float)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--pseudo-label-threshold', default=0.95, type=float, help='Threshold when filtering pseudo-labeling.')
    parser.add_argument('--num-classes', default=10, type=int, help='number of classes in the dataset.')
    parser.add_argument('--num-labeled-datapoints', default=250, type=int, help='number of labeled annotations in the dataset.')
    parser.add_argument('--starting-model', type=str, default=None, help='Pytorch model checkpoint to load instead of starting from random')
    parser.add_argument('--optimizer', type=str, default='sgd',help='optimizer if nothing is passed AdamW would be used (currently supported sgd,adamw)')
    parser.add_argument('--strong-augmentation', help='enables strong augmentation', action='store_true')
    parser.add_argument('--debug', help='enables debugging mode', action='store_true')
    # parser.add_argument('--skl',help='uses sklearn implementation of Gaussian Mixture',action='store_true')
    parser.add_argument('--last-layer', type=str, default='fc', help='last layer to use in the NN')
    parser.add_argument('--num-epochs', default=None, type=int, help='number of epochs to train. If this is non-None it will suppress the use of a validation split, and blindly run the training for N epochs.')
    # parser.add_argument('--loss-terms', type=str, default='gmm', help='what loss terms to be included (options are gmm, cmm, cluster_dist). For example "cmm+gmm+cluster_dist"')
    # parser.add_argument('--pseudo-label-determination', default="gmm", type=str, help='which set of logits to use when picking valid pseudo-labels. (gmm, cmm)')
    # parser.add_argument('--pseudo-label-target-logits', default="gmm", type=str, help='which set of logits to use as the optimization target for valid pseudo-labels. (gmm, cmm)')

    parser.add_argument('--trainer', type=str, default='fixmatch', help='trainer to use (currently supported supervised, fixmatch, fixmatch-gmm)')




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
