import warnings
import numpy as np
import torch.optim
import logging

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau

# LR scheduler which does early stopping and exponential decay

class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Wrapper around torch.optim.lr_scheduler.ReduceLROnPlateau which exposes a subset of that classes parameters while additionally keeping track of the total number of learning rate reductions which have been applied.
    This class provides two additional callbacks (which must be functions), lr_reduction_callback which gets called when the learning rate is reduced due to the metric having plateaued, and termination_callback which is called when the metric has once again plateaued and the learning rate would have been reduced, but max_num_lr_reductions have already happened. So the scheduler should not reduce the learning rate any more. This is effectively a termination callback, letting the user know when the metric has stopped improving after n lr reductions.

    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        max_num_lr_reductions (int): The maximum number of learning rate reductions which will happend before the termination callback is called.
        lr_reduction_callback (function handle): default = None, function handle which gets called when the learning rate is reduced.
        termination_callback (function handle): default = None, function handle which gets called when the learning rate would have been reduced, but max_num_lr_reductions has been reached.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-3, max_num_lr_reductions=2, lr_reduction_callback=None, termination_callback=None):
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)

        self.num_lr_reductions = 0
        self.max_num_lr_reductions = max_num_lr_reductions
        self.lr_reduction_callback = lr_reduction_callback
        self.termination_callback = termination_callback
        self.best_metric_epoch = 0
        self.metric_values = list()

    def is_done(self):
        return self.num_lr_reductions > self.max_num_lr_reductions

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        self.metric_values.append(current)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.mode == 'min':
            error_from_best = np.abs(np.asarray(self.metric_values) - np.nanmin(self.metric_values))
            error_from_best[error_from_best < np.abs(self.threshold)] = 0
        else:
            error_from_best = np.abs(np.asarray(self.metric_values) + np.nanmax(self.metric_values))
            error_from_best[error_from_best > np.abs(self.threshold)] = 0
        # unpack numpy array, select first time since that value has happened
        self.best_metric_epoch = np.where(error_from_best == 0)[0][0]

        # update the number of "bad" epochs. The (-1) handles 0 based indexing vs natural counting of epochs
        self.num_bad_epochs = (epoch-1) - self.best_metric_epoch

        if self.num_bad_epochs > self.patience:
            self.num_lr_reductions += 1
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

            if self.num_lr_reductions > self.max_num_lr_reductions:
                # we have completed the requested number of learning rate reductions, call the provided function handle to let the user respond to this
                if self.termination_callback is not None:
                    self.termination_callback()
            else:
                self._reduce_lr(epoch)
                # invalidate the metric values from before the learning rate change, as those values should no longer be used in evaluating convergence
                self.metric_values = [np.nan for v in self.metric_values]
                if self.lr_reduction_callback is not None:
                    self.lr_reduction_callback()

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]