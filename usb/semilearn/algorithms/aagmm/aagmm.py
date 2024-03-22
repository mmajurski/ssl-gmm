import os
import torch
import numpy as np
import sklearn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
        self.args = args
        self.lcl_stats = dict()

        if args.embedding_constraint is None:
            self.emb_constraint = None
        elif args.embedding_constraint == 0:
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

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        GENERATE_CSV = False

        # inference and calculate sup/unsup losses
        with (self.amp_cm()):
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
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

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

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb) #, denom_lb=denom_lb, denom_ulb=denom_ulb_w)

            # remove outliers
            # denom_thres = torch.min(denom_lb)
            # denom_thres = torch.quantile(denom_lb, 0.25)  # 5th percentile
            # dthres_rate = torch.sum((denom_ulb_w <= denom_thres).float()) / denom_ulb_w.shape[0]

            if isinstance(self.args.outlier_thres, (int, float)):
                # denom_thres = torch.tensor(self.args.outlier_thres, device=denom_ulb_w.device, dtype=denom_ulb_w.dtype, requires_grad=False)
                denom_thres = torch.quantile(denom_lb, self.args.outlier_thres)
                dthres_rate = torch.sum((denom_ulb_w >= denom_thres).float()) / denom_ulb_w.shape[0]
                mask[denom_ulb_w > denom_thres] = 0
            else:
                denom_thres = torch.tensor(torch.nan)
                dthres_rate = torch.tensor(torch.nan)

            if GENERATE_CSV:
                bins = np.linspace(0, 10, 50)
                denom_lb_hist_c, denom_lb_hist_x = np.histogram(denom_lb.detach().cpu().numpy(), bins=bins)
                denom_ulb_w_hist_c, denom_ulb_w_hist_x = np.histogram(denom_ulb_w.detach().cpu().numpy(), bins=bins)
                denom_ulb_s_hist_c, denom_ulb_s_hist_x = np.histogram(denom_ulb_s.detach().cpu().numpy(), bins=bins)

                if 'denom_lb_hist_c' not in self.lcl_stats.keys():
                    self.lcl_stats['denom_lb_hist_c'] = []
                self.lcl_stats['denom_lb_hist_c'].append(denom_lb_hist_c)
                if 'denom_lb_hist_x' not in self.lcl_stats.keys():
                    self.lcl_stats['denom_lb_hist_x'] = []
                self.lcl_stats['denom_lb_hist_x'].append(denom_lb_hist_x)

                if 'denom_ulb_w_hist_c' not in self.lcl_stats.keys():
                    self.lcl_stats['denom_ulb_w_hist_c'] = []
                self.lcl_stats['denom_ulb_w_hist_c'].append(denom_ulb_w_hist_c)
                if 'denom_ulb_w_hist_x' not in self.lcl_stats.keys():
                    self.lcl_stats['denom_ulb_w_hist_x'] = []
                self.lcl_stats['denom_ulb_w_hist_x'].append(denom_ulb_w_hist_x)

                if 'denom_ulb_s_hist_c' not in self.lcl_stats.keys():
                    self.lcl_stats['denom_ulb_s_hist_c'] = []
                self.lcl_stats['denom_ulb_s_hist_c'].append(denom_ulb_s_hist_c)
                if 'denom_ulb_s_hist_x' not in self.lcl_stats.keys():
                    self.lcl_stats['denom_ulb_s_hist_x'] = []
                self.lcl_stats['denom_ulb_s_hist_x'].append(denom_ulb_s_hist_x)

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

        lb_acc_elementwise = (torch.argmax(logits_x_lb, dim=-1) == y_lb).float()

        lb_acc = torch.mean(lb_acc_elementwise).detach()

        if GENERATE_CSV:
            lb_D = denom_lb.detach().cpu().numpy()
            ulb_w_D = denom_ulb_w.detach().cpu().numpy()
            ulb_s_D = denom_ulb_s.detach().cpu().numpy()

            if 'denom_lb_vals' not in self.lcl_stats.keys():
                self.lcl_stats['denom_lb_vals'] = []
            self.lcl_stats['denom_lb_vals'].append(lb_D)
            if 'pl_acc' not in self.lcl_stats.keys():
                self.lcl_stats['pl_acc'] = []
            self.lcl_stats['pl_acc'].append(lb_acc_elementwise.detach().cpu().numpy())

            for c in range(self.num_classes):
                idx = y_lb.detach().cpu().numpy() == c
                cur_lb_D = lb_D[idx]
                cur_lb_acc_elementwise = lb_acc_elementwise[idx]
                k = 'denom_lb_c{}_vals'.format(c)
                if k not in self.lcl_stats.keys():
                    self.lcl_stats[k] = []
                self.lcl_stats[k].append(cur_lb_D)
                k = 'pl_acc_c{}_vals'.format(c)
                if k not in self.lcl_stats.keys():
                    self.lcl_stats[k] = []
                self.lcl_stats[k].append(cur_lb_acc_elementwise)

            k = 'denom_ulb_w_vals'
            if k not in self.lcl_stats.keys():
                self.lcl_stats[k] = []
            self.lcl_stats[k].append(ulb_w_D)
            k = 'denom_ulb_s_vals'
            if k not in self.lcl_stats.keys():
                self.lcl_stats[k] = []
            self.lcl_stats[k].append(ulb_s_D)

            for c in range(self.num_classes):
                idx = pseudo_label.detach().cpu().numpy() == c
                cur_ulb_w_D = ulb_w_D[idx]
                cur_ulb_s_D = ulb_s_D[idx]
                k = 'denom_ulb_w_c{}_vals'.format(c)
                if k not in self.lcl_stats.keys():
                    self.lcl_stats[k] = []
                self.lcl_stats[k].append(cur_ulb_w_D)
                k = 'denom_ulb_s_c{}_vals'.format(c)
                if k not in self.lcl_stats.keys():
                    self.lcl_stats[k] = []
                self.lcl_stats[k].append(cur_ulb_s_D)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        if self.emb_constraint is not None:
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item(),
                                             util_ratio=mask.float().mean().item(),
                                             denom_thres=denom_thres.item(),
                                             dthres_rate=dthres_rate.item(),
                                             lb_acc=lb_acc.item(),
                                             emb_x_lb_loss=emb_constraint_loss_l.item(),
                                             emb_x_ulb_s_loss=emb_constraint_loss_ul_strong.item(),
                                             emb_x_ulb_w_loss=emb_constraint_loss_ul_weak.item())
        else:
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item(),
                                             util_ratio=mask.float().mean().item(),
                                             denom_thres=denom_thres.item(),
                                             dthres_rate=dthres_rate.item(),
                                             lb_acc=lb_acc.item())
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

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()
        print("running evaluate")

        for key in self.lcl_stats.keys():
            save_path = os.path.join(self.args.save_dir, self.args.save_name, '{}.csv'.format(key))
            csv_row = ''
            for k in range(len(self.lcl_stats[key])):
                for v_idx in range(len(self.lcl_stats[key][k])):
                    v = self.lcl_stats[key][k][v_idx]
                    if v_idx > 0:
                        csv_row += ','
                    csv_row += str(v)
                csv_row += '\n'
            with open(save_path, 'w') as fd:
                fd.write(csv_row)

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        embedding_output_test = list()
        labels_output_test = list()

        with torch.no_grad():
            for data in eval_loader:
                x = data["x_lb"]
                y = data["y_lb"]

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = self.model(x)
                logits = outputs[out_key]
                embedding = outputs['feat']

                embedding_output_test.append(embedding.detach().cpu().numpy())
                labels_output_test.append(y.detach().cpu().numpy())


                loss = torch.nn.functional.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        F1 = f1_score(y_true, y_pred, average="macro")

        per_class_accuracy = dict()
        acc = (y_true == y_pred).astype(float)
        for c in range(self.num_classes):
            per_class_accuracy[c] = np.mean(acc[y_true == c])

        cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
        save_path = os.path.join(self.args.save_dir, self.args.save_name, 'confusion_matrix.png')
        import matplotlib.pyplot as plt
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cf_mat)
        disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
        plt.savefig(save_path)


        # embedding_output_test = utils.multiconcat_numpy(embedding_output_test)
        embedding_output_test = np.concatenate(embedding_output_test, axis=0)
        save_path = os.path.join(self.args.save_dir, self.args.save_name, 'test_embedding.npy')
        np.save(save_path, embedding_output_test)

        # labels_output_test = utils.multiconcat_numpy(labels_output_test)
        labels_output_test = np.concatenate(labels_output_test, axis=0)
        save_path = os.path.join(self.args.save_dir, self.args.save_name, 'test_labels.npy')
        np.save(save_path, labels_output_test)

        torch.save(self.model.cpu(), os.path.join(self.args.save_dir, self.args.save_name, "model.pt"))

        self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/top-1-acc": top1,
            eval_dest + "/balanced_acc": balanced_top1,
            eval_dest + "/precision": precision,
            eval_dest + "/recall": recall,
            eval_dest + "/F1": F1,
        }
        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits
        for c in range(self.num_classes):
            eval_dict[eval_dest + "/class_{}_acc".format(c)] = per_class_accuracy[c]
        return eval_dict