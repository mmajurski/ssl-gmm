import numpy as np
import math
import torch
import random
from logger import GMMLogger

# TODO: decide what do we want to log based on space required and log those things


class GMM(torch.nn.Module):
    def __init__(self, dataset, num_clusters, eps=1e-6):
        """
                :param dataset: type: torch.Tensor
                :param num_clusters: type: int
        """

        super(GMM, self).__init__()

        self.X = dataset
        self.k = num_clusters
        self._eps = eps

        self._init_params()

    def _init_params(self):

        self.N = self.X.shape[0]  # number of data points
        self.d = self.X.shape[1]  # dimension
        self._log_likelihood = -np.inf
        self._cur_epoch = 0
        self._best_epoch = -1

        # TODO: make means, cov and priors torch.nn.Parameters if required

        # selecting k random indices out of N and initializing means with those data-points
        indices = torch.tensor(random.sample(range(self.N), self.k))
        self._means = self.X[torch.tensor(indices)].double()  # dim: k * d (num_cluster * dimension)
        # CHANGE: used torch implementation instead of numpy and then converting to torch
        self._best_means = self._means  # to keep track of best parameters

        # sets mixture covariance matrix to initialize cov_mat of each cluster
        x_norm = self.X - torch.mean(self.X, 0).double()

        # TODO: change this implementation, we only need the diagonal N*v*v where v = min(k,d)
        x_cov = ((x_norm.t() @ x_norm) / self.N).double()
        # CHANGE: in denominator it was N-1 changed it to N

        self._cov = self.k * [x_cov]
        self._best_cov = self._cov

        # CHANGE: removed numpy_cov as it was not being used anywhere, just assigned
        # CHANGE: removed cov_inv and cov_det and it's calculations as we won't require them anymore

        # uniform initialization of class probabilities
        self._priors = torch.empty(self.k).fill_(1. / self.k).double()

        # # cluster probabilities for each data-point
        # self._log_resp = torch.zeros((self.N, self.k)).double()  # dim: N * k (num_data_points * num_clusters)

        # calculating d * log(2*pi) and saving for future use
        self._dlog2pi = self.d * math.log(2 * math.pi)

        # setting up the logger
        self._logger = GMMLogger()

    def fit(self, delta=1e-4, n_epochs=50, early_stopping_epoch_count=10):

        epoch = 0
        best_epoch = 0
        done = False
        # ll_diff = np.inf  # difference between previous and current log_likelihood

        while epoch < n_epochs:
            self.__em()
            # TODO: implement the early stopping here
            log_likelihood = self.logger.get('log_likelihood')
            error_from_best = np.abs(np.max(log_likelihood) - log_likelihood)
            error_from_best[error_from_best < np.abs(self.tolerance)] = 0

            # if this epoch is with convergence tolerance of the global best, save the weights
            if error_from_best[self.cur_epoch] == 0:
                print('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'
                      .format(self.cur_epoch, log_likelihood[self.cur_epoch], self.tolerance))

                self._best_means = self._means
                self._best_cov = self._cov
                self._best_epoch = self._cur_epoch

            best_val_loss_epoch = np.where(error_from_best == 0)[0][0]

            if self.cur_epoch >= (best_val_loss_epoch + self.early_stopping_count):
                print("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(
                    self.cur_epoch))
                return True

            epoch += 1
            self._cur_epoch += 1

        return False

    def __em(self):

        self._logger.new_epoch(self._cur_epoch)
        log_resp = self._e_step()
        self._m_step(log_resp)

    def _e_step(self):

        weighted_log_probabilities = self._get_log_prob() + torch.log(self._priors)

        normalized_log_probabilities = torch.logsumexp(weighted_log_probabilities, dim=1, keepdim=True)

        log_resp = weighted_log_probabilities - normalized_log_probabilities

        return log_resp

    def _get_log_prob(self):

        precision = torch.rsqrt(self._cov)

        # TODO: try this out, most probably will have to make some modifications, either in eq or in dimensions of input
        log_prob = torch.sum((self._means * self._means + self.X * self.X - 2 * self.X * self._means) * (precision ** 2), dim=2, keepdim=True)
        log_det = torch.sum(torch.log(precision), dim=2, keepdim=True)

        return -0.5 * (self._dlog2pi + log_prob) + log_det

    def _m_step(self, log_resp):

        resp = torch.exp(log_resp)

        self._priors = torch.sum(resp, dim=0, keepdim=True) + self._eps

        self._means = torch.sum(resp * self.X, dim=0, keepdim=True)

        x2 = (resp * self.X * self.X).sum(0, keepdim=True) / self._priors
        mu2 = self._means * self._means
        xmu = (resp * self._means * self.X).sum(0, keepdim=True) / self._priors
        # TODO: check cov and it's dimensions
        self._cov = x2 - 2 * xmu + mu2 + self._eps

        self._priors = self._priors / self.N
