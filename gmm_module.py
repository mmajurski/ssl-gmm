import torch
import numpy as np
from math import gamma
"""
----------------:Change log:----------------
- replaced squeeze() with squeeze(dim=0) to avoid unintentional squeezing in case of n_clusters being 1
- change sigma initialization from torch.randn to be identity matrices as randn was gnerating negative numbers
    which aren't allowed in finding cholskey
- set eps = (10 * torch.finfo(resp.dtype).eps)
    as resp.type() was returning str
- set resp_x = torch.matmul(resp.transpose(2, 1), x)
    as resp.transpose(0,1) was resulting in shape (N,1,k)
"""

# TODO transfer everything to gpu
# TODO add logger and log required params


class GMM(torch.nn.Module):

    def __init__(self, n_features, n_clusters, tolerance=1e-5, max_iter=100, weights=None, mu=None, sigma=None):
        """
            _pi = clusters _pi(probabilities)
            _mu = clusters means
            _sigma = clusters full covariance matrices
        """
        super(GMM, self).__init__()
        self.n_features = n_features  # Dimensions
        self.n_clusters = n_clusters  # k
        self.tol = tolerance
        self._max_iter = max_iter
        self._pi = weights
        self._mu = mu
        self._sigma = sigma
        # additional initializations
        self.log_likelihood = -np.inf
        self._pi_shape = (self.n_clusters,)
        self._mu_shape = (self.n_clusters, self.n_features)
        self._sigma_shape = (self.n_clusters, self.n_features, self.n_features)
        self._converged = False
        self._eps = 1e-6
        self._init_params()

    def _init_params(self):
        # validate or set initial cluster _pi
        if self._pi is not None:
            if self._pi.size() != self._pi_shape:
                raise ValueError("Invalid pi provided")
            elif not torch.allclose(self._pi.sum(), torch.tensor(1.0, dtype=torch.float64)):
                raise ValueError(
                    f"The parameter 'weights' should be normalized, but got sum(weights) = {self._pi.sum():.5f}")
            elif any(torch.less(self._pi, 0.0)) or any(torch.greater(self._pi, 1.0)):
                raise ValueError(f"The parameter 'weights' should be in the range [0, 1], but got max value {self._pi.min().item():.5f}, min value {self._pi.min().item():.5f}")
            self._pi = torch.nn.Parameter(self._pi, requires_grad=False)
        else:
            self._pi = torch.nn.Parameter(torch.Tensor(*self._pi_shape), requires_grad=False).fill_(1. / self.n_clusters)
        # validate or set initial cluster means
        if self._mu is not None:
            if self._mu.size() != self._mu_shape:
                raise ValueError("Invalid means provided")
            self._mu = torch.nn.Parameter(self._mu, requires_grad=False)
        else:
            self._mu = torch.nn.Parameter(torch.randn(*self._mu_shape), requires_grad=False)
        # validate or set initial cluster covariance matrices
        if self._sigma is not None:
            if self._sigma.size() != self._sigma_shape:
                raise ValueError("Invalid covariance matrices provided")
            self._sigma = torch.nn.Parameter(self._sigma, requires_grad=False)
        else:
            self._sigma = torch.nn.Parameter(torch.eye(self.n_features).reshape(1, self.n_features, self.n_features).repeat(self.n_clusters, 1, 1), requires_grad=False)

        self.precision_cholesky = self._compute_precision_cholesky()

    def get(self, var: str):
        """
        var can be "mu", "sigma" or "pi"
        """
        return getattr(self, '_'+var)


    def fit(self, x):
        """
        fit the GMM module to the data by training it for max_iter epochs or till it is converged
        returns: True if converged before reaching max epochs, False if ran max epochs and didn't converge
        """
        # training for _max_iter epochs
        for epoch in range(self._max_iter):

            # performing one EM step
            log_likelihood = self._em_iteration(x)

            # difference between current and previous iteration's log_likelihood
            ll_diff = log_likelihood - self.log_likelihood
            self.log_likelihood = log_likelihood
            # setting converged to True and terminating if the difference is insignificant
            if abs(ll_diff) < self.tol:
                self._converged = True
                break
        return self._converged

    def _estimate_log_prob(self, x):
        n_samples = x.size(dim=0)

        # calculating log_det
        log_det = torch.sum(
            torch.log(self.precision_cholesky.reshape(self.n_clusters, -1)[:, ::self.n_features + 1]), 1)
        log_prob = torch.empty((n_samples, self.n_clusters))

        # assuming covariance_type to be full and calculating the log_prob
        for k, (mu, prec_chol) in enumerate(zip(self._mu, self.precision_cholesky)):
            y = torch.matmul(torch.sub(x, mu), prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)

        return -0.5 * (self.n_features * np.log(2 * np.pi) + log_prob) + log_det  # + more data

    def _estimate_weighted_log_prob(self, x):
        # equivalent of log(likelihood * pi)
        return self._estimate_log_prob(x) + torch.log(self._pi)

    def _estimate_log_prob_resp(self, x):
        # adding log of cluster probability to log probability
        weighted_log_prob = self._estimate_weighted_log_prob(x)
        # log(sum(exp(weighted_log_prob elements across dim=1)))
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        # equivalent of dividing by sum in the normal space
        # verify if we need to normalize in case of 1 cluster? as it is returning all 0s in log space (1 in normal)
        log_resp = weighted_log_prob - log_prob_norm
        return weighted_log_prob, log_prob_norm, log_resp

    def _e_step(self, x):
        _, log_prob_norm, log_resp = self._estimate_log_prob_resp(x)
        return torch.mean(log_prob_norm), log_resp

    def _estimate_parameters(self, x, resp):

        # typecasting x to be the same type as resp for multiplication support
        x = x.type(resp.dtype)

        # calculating pi and adding (10 * smallest possible value for resp) to avoid 0
        pi = torch.sum(resp, dim=0) + self._eps
        # pi = torch.add(torch.sum(resp, dim=0), 10 * torch.finfo(resp.dtype).eps)


        # means = sum(resp * x, dim=0) / pi
        resp_x = torch.matmul(resp.transpose(0, 1), x)
        means = torch.div(input=resp_x, other=pi.unsqueeze(-1))

        # finding covariances for full covariance_type
        # ref: https://github.com/ldeecke/gmm-torch/blob/8e72b1ddad80e08cb55be928a2d7d66ee9d16f3d/gmm.py#L344

        # regularization value to be added on the diagonal
        eps = (torch.eye(self.n_features) * self._eps).to(x.device)
        # mean reduced x
        x_mr = x - means
        # (n,1,d,d) = ( (n,d,1) * (n,1,d) ).unsqueeze(1)
        x_mr_n1dd = x_mr.unsqueeze(-1).matmul(x_mr.unsqueeze(-2)).unsqueeze(1)
        resp_nk11 = resp.unsqueeze(-1).unsqueeze(-1)
        numerator = torch.sum(x_mr_n1dd * resp_nk11, dim=0, keepdim=True)  # (1,k,d,d)
        denominator = torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # (1,k,1,1)
        var = numerator / denominator + eps  # (1,k,d,d)
        var = var.squeeze(0)
        return pi, means, var

    def _m_step(self, x, log_resp):
        pi, mus, sigmas = self._estimate_parameters(x, torch.exp(log_resp))
        self._pi = torch.nn.Parameter(pi, requires_grad=False)
        self._mu = torch.nn.Parameter(mus, requires_grad=False)
        self._sigma = torch.nn.Parameter(sigmas, requires_grad=False)
        # normalizing pi
        torch.div(input=self._pi, other=torch.sum(self._pi), out=self._pi)
        # computing precision_cholesky with new covariance (sigma)
        self.precision_cholesky = self._compute_precision_cholesky()

    def _em_iteration(self, x):
        log_likelihood, log_resp = self._e_step(x)
        self._m_step(x, log_resp)
        return log_likelihood

    def predict_probability(self, x):
        log_prob_weighted, log_prob_norm, log_resp = self._estimate_log_prob_resp(x)
        return torch.exp(log_prob_weighted), torch.exp(log_prob_norm), torch.exp(log_resp)

    def _cauchy_estimate_log_prob(self, x):
        n_samples = x.size(dim=0)

        power_value = (1 + self.n_features) / 2
        numerator = np.log(gamma(power_value))
        denom_1 = np.log(gamma(1 / 2))
        denom_2 = (self.n_features / 2) * np.log(np.pi)

        log_det = torch.sum(
            torch.log(self.precision_cholesky.reshape(self.n_clusters, -1)[:, ::self.n_features + 1]), 1)

        # mahalanobis distance
        m_dist = torch.empty((n_samples, self.n_clusters))

        for k, (mu, prec_chol) in enumerate(zip(self._mu, self.precision_cholesky)):
            y = torch.matmul(torch.sub(x, mu), prec_chol)
            m_dist[:, k] = torch.sum(torch.square(y), dim=1)

        denom_3 = torch.mul(log_det, 0.5)

        denom_4 = torch.mul(torch.log(m_dist), power_value)

        tensor_part = torch.mul(denom_4.add(denom_3), -1)

        scalar_part = numerator - denom_1 - denom_2

        log_pdf = torch.add(tensor_part, scalar_part)

        return log_pdf

    def _cauchy_estimate_weighted_log_prob(self, x):
        return self._cauchy_estimate_log_prob(x) + torch.log(self._pi)

    def _cauchy_estimate_log_prob_resp(self, x):
        weighted_log_prob = self._cauchy_estimate_weighted_log_prob(x)
        # log(sum(exp(weighted_log_prob elements across dim=1)))
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        # equivalent of dividing by sum in the normal space
        # verify if we need to normalize in case of 1 cluster? as it is returning all 0s in log space (1 in normal)
        log_resp = weighted_log_prob - log_prob_norm
        return log_prob_norm, log_resp , weighted_log_prob

    def predict_cauchy_probability(self, x):
        log_prob_norm, log_resp, unnorm_log_resp = self._cauchy_estimate_log_prob_resp(x)
        return torch.exp(log_prob_norm), torch.exp(log_resp) , unnorm_log_resp
    
    def _compute_precision_cholesky(self):
        """
        Calculates precision_cholesky from _sigma (covariances)
        returns:
            precision_cholesky:     torch.Tensor (k,d,d)
        """
        precision_cholesky = torch.empty(self._sigma_shape)
        for k, covar_mat in enumerate(self._sigma):
            try:
                cov_chol = torch.linalg.cholesky(covar_mat)
            except:
                raise ValueError("increase _eps because covariances are bad")
            precision_cholesky[k] = torch.linalg.solve_triangular(cov_chol, torch.eye(self.n_features),
                                                                  upper=False).transpose(0, 1)
        return precision_cholesky

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        cov_params = self.n_clusters * self.n_features * (self.n_features + 1) / 2.0
        mean_params = self.n_features * self.n_clusters
        return int(cov_params + mean_params + self.n_clusters - 1)
