
import torch
import numpy as np

from .utils import AAGMMThresholdingHook
from .models import AagmmModelWrapper

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from .embedding_constraints import GaussianMoments


@ALGORITHMS.register('aagmm')
class AAGMM(AlgorithmBase):
    """
        AAGMM algorithm

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
        self.embedding_criterion = torch.nn.MSELoss()

        if args.embedding_constraint is None:
            self.emb_constraint = None
        elif args.embedding_constraint == 1:
            self.emb_constraint = GaussianMoments(embedding_dim=args.embedding_dim, num_classes=args.num_classes, moment_order=1)
        elif args.embedding_constraint == 2:
            self.emb_constraint = GaussianMoments(embedding_dim=args.embedding_dim, num_classes=args.num_classes, moment_order=2)
        elif args.embedding_constraint == 3:
            self.emb_constraint = GaussianMoments(embedding_dim=args.embedding_dim, num_classes=args.num_classes, moment_order=3)
        elif args.embedding_constraint == 4:
            self.emb_constraint = GaussianMoments(embedding_dim=args.embedding_dim, num_classes=args.num_classes, moment_order=4)
        else:
            raise RuntimeError("Invalid embedding constraint type: {}".format(args.embedding_constraint))

    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(AAGMMThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(
            num_classes=self.num_classes,
            pretrained=self.args.use_pretrain,
            pretrained_path=self.args.pretrain_path,
        )

        # wrap the model with aagmm_layer
        model = AagmmModelWrapper(model, num_classes=self.num_classes, last_layer=self.args.last_layer, embedding_dim=self.args.embedding_dim)

        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.set_model()
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                emb_lb = outputs['feat'][:num_lb]
                emb_ulb_w, emb_ulb_s = outputs['feat'][num_lb:].chunk(2)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                raise RuntimeError("Data must be catted together")
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            denom_lb = outputs['denom'][:num_lb]
            denom_ulb_w, denom_ulb_s = outputs['denom'][num_lb:].chunk(2)



            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb, denom_lb=denom_lb, denom_ulb=denom_ulb_w)

            # remove outliers
            denom_thres = torch.max(denom_lb)
            mask[denom_ulb_w > denom_thres] = 0
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

            if self.emb_constraint is not None:
                # labeled data embedding constraint
                if hasattr(self.model, 'module'):
                   emb_constraint_l = self.emb_constraint(emb_lb, self.model.module.last_layer.centers, logits_x_lb)
                else:
                   emb_constraint_l = self.emb_constraint(emb_lb, self.model.last_layer.centers, logits_x_lb)
                emb_constraint_loss_l = self.embedding_criterion(emb_constraint_l, torch.zeros_like(emb_constraint_l))

                # unlabeled data embedding constraint
                if hasattr(self.model, 'module'):
                    emb_constraint_ul_strong = self.emb_constraint(emb_ulb_s, self.model.module.last_layer.centers, logits_x_ulb_w)
                    emb_constraint_ul_weak = self.emb_constraint(emb_ulb_w, self.model.module.last_layer.centers, logits_x_ulb_w)
                else:
                    emb_constraint_ul_strong = self.emb_constraint(emb_ulb_s, self.model.last_layer.centers, logits_x_ulb_w)
                    emb_constraint_ul_weak = self.emb_constraint(emb_ulb_w, self.model.last_layer.centers, logits_x_ulb_w)

                emb_constraint_loss_ul_strong = self.embedding_criterion(emb_constraint_ul_strong, torch.zeros_like(emb_constraint_ul_strong))
                emb_constraint_loss_ul_weak = self.embedding_criterion(emb_constraint_ul_weak, torch.zeros_like(emb_constraint_ul_weak))

                total_loss = total_loss + emb_constraint_loss_l + emb_constraint_loss_ul_strong + emb_constraint_loss_ul_weak
            else:
                emb_constraint_loss_l = None
                emb_constraint_loss_ul_strong = None
                emb_constraint_loss_ul_weak = None



        pl_acc = (pseudo_label == y_ulb).detach().float()
        pl_acc_per_class = dict()
        for i in range(self.num_classes):
            pl_acc_per_class[i] = torch.mean(pl_acc[y_ulb == i]).item()
        pl_acc = pl_acc[mask > 0]
        pl_acc = torch.mean(pl_acc)

        lb_acc = torch.mean((torch.argmax(logits_x_lb, dim=-1) == y_lb).detach().float())

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         denom_thres=denom_thres.item(),
                                         pl_acc=pl_acc.item(),
                                         lb_acc=lb_acc.item(),
                                         emb_x_lb_loss=emb_constraint_loss_l.item(),
                                         emb_x_ulb_s_loss=emb_constraint_loss_ul_strong.item(),
                                         emb_x_ulb_w_loss=emb_constraint_loss_ul_weak.item())
        for i in range(self.num_classes):
            log_dict['train/pl_acc_per_class_{}'.format(i)] = pl_acc_per_class[i]
        return out_dict, log_dict
        

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
