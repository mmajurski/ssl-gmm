import torch
import numpy as np

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


class GMM(torch.nn.Module):

    def __init__(self, n_features, n_clusters, tolerance=1e-5, max_iter=100, weights=None, mus=None, sigmas=None):
        """
            _pi = clusters _pi(probabilities)
            _mus = clusters means
            _sigmas = clusters full covariance matrices
        """
        super(GMM, self).__init__()
        self.n_features = n_features  # Dimensions
        self.n_clusters = n_clusters  # k
        self.tol = tolerance
        self._max_iter = max_iter
        self._pi = weights
        self._mus = mus
        self._sigmas = sigmas
        # additional initializations
        self.log_likelihood = -np.inf
        self._pi_shape = (1, self.n_clusters, 1)
        self._mus_shape = (1, self.n_clusters, self.n_features)
        self._sigmas_shape = (1, self.n_clusters, self.n_features, self.n_features)
        self._converged = False
        self.reg_covar = 1e-6
        self._init_params()

    def _init_params(self):
        # validate or set initial cluster _pi
        if self._pi is not None:
            if self._pi.size() != self._pi_shape or self._pi.sum(dim=1).item() != 1:
                raise ValueError("Invalid pi provided")
            self._pi = torch.nn.Parameter(self._pi, requires_grad=False)
        else:
            self._pi = torch.nn.Parameter(torch.Tensor(*self._pi_shape), requires_grad=False).fill_(1. / self.n_clusters)
        # validate or set initial cluster means
        if self._mus is not None:
            if self._mus.size() != self._mus_shape:
                raise ValueError("Invalid means provided")
            self._mus = torch.nn.Parameter(self._mus, requires_grad=False)
        else:
            self._mus = torch.nn.Parameter(torch.randn(*self._mus_shape), requires_grad=False)
        # validate or set initial cluster covariance matrices
        if self._sigmas is not None:
            if self._sigmas.size() != self._sigmas_shape:
                raise ValueError("Invalid covariance matrices provided")
            self._sigmas = torch.nn.Parameter(self._sigmas, requires_grad=False)
        else:
            self._sigmas = torch.nn.Parameter(torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_clusters, 1, 1), requires_grad=False)

        self.precision_cholesky = self._compute_precision_cholesky()

    def get(self, var: str):
        """
        var can be "mu", "sigma" or "pi"
        """
        if var == "mu":
            return torch.squeeze(self._mus, dim=0)
        if var == "sigma":
            return torch.squeeze(self._sigmas, dim=0)
        if var == "pi":
            return torch.squeeze(self._pi, dim=0)
        else:
            return

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
            torch.log(self.precision_cholesky.squeeze(0).reshape(self.n_clusters, -1)[:, ::self.n_features + 1]), 1)
        log_prob = torch.empty((n_samples, self.n_clusters))

        # assuming covariance_type to be full and calculating the log_prob
        for k, (mu, prec_chol) in enumerate(zip(self._mus.squeeze(0), self.precision_cholesky.squeeze(0))):
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
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        # equivalent of dividing by sum in the normal space
        log_resp = weighted_log_prob - log_prob_norm
        return log_prob_norm, log_resp

    def _e_step(self, x):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(x)
        return torch.mean(log_prob_norm), log_resp

    def _estimate_parameters(self, x, resp):
        # taking epsilon as 10 * the smallest possible value for resp
        eps = (10 * torch.finfo(resp.dtype).eps)
        # adding epsilon
        nk = torch.add(torch.sum(resp), eps)
        # TODO: test the resp.transpose change
        resp_x = torch.matmul(resp.transpose(2, 1), x)
        means = torch.div(input=resp_x, other=nk.unsqueeze(1))
        # finding covariances for full covariance_type
        covariances = torch.empty(self._sigmas_shape).squeeze(0)
        for k in range(self.n_components):
            x_mr = torch.sub(input=x, other=means[k])
            resp_xmrt = torch.matmul(resp[:, k], x_mr.transpose(0, 1))
            resp_xmrt_xmr = torch.matmul(resp_xmrt, x_mr)

            covariances[k] = torch.div(resp_xmrt_xmr, nk[k])
            covariances[k, ::self.n_features + 1] = torch.add(input=covariances[k, ::self.n_features + 1],
                                                              other=self.reg_covar)
        return nk, means, covariances

    def _m_step(self, x, log_resp):
        self._pi, self._mus, self._sigmas = self._estimate_parameters(x, torch.exp(log_resp))
        # normalizing pi
        torch.div(input=self._pi, other=torch.sum(self._pi), out=self._pi)
        # computing precision_cholesky with new covariance (sigma)
        self.precision_cholesky = self._compute_precision_cholesky()

    def _em_iteration(self, x):
        log_likelihood, log_resp = self._e_step(x)
        self._m_step(x, log_resp)
        return log_likelihood

    def predict_probability(self, x):
        _, log_resp = self._estimate_log_prob_resp(x)
        return torch.exp(log_resp)

    def _compute_precision_cholesky(self):
        precision_cholesky = torch.empty(self._sigmas_shape).squeeze(dim=0)
        for k, covar_mat in enumerate(self._sigmas.squeeze(dim=0)):
            # try:
            cov_chol = torch.linalg.cholesky(covar_mat)
            # except():
            #     raise ValueError("increase reg_covar because covariances are bad")
            precision_cholesky[k] = torch.linalg.solve_triangular(cov_chol, torch.eye(self.n_features),
                                                                  upper=False).transpose(0, 1)
        return precision_cholesky.unsqueeze(0)
