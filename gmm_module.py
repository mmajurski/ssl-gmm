import torch
import numpy as np


class GMM(torch.nn.Module):

    def __init__(self, n_features, n_clusters, tolerance=1e-5, max_iter=100, weights=None, mus=None, sigmas=None):
        """
            pi = clusters pi(probabilities)
            mus = clusters means
            sigmas = clusters full covariance matrices
        """
        super(GMM, self).__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.tol = tolerance
        self.max_iter = max_iter
        self.pi = weights
        self.mus = mus
        self.sigmas = sigmas
        # additional initializations
        self.log_likelihood = -np.inf
        self.pi_shape = (1, self.n_clusters, 1)
        self.mus_shape = (1, self.n_clusters, self.n_features)
        self.sigmas_shape = (1, self.n_clusters, self.n_features, self.n_features)
        self._init_params()
        self.converged = False

        self.reg_covar = 1e-6

    def _init_params(self):
        # validate or set initial cluster pi
        if self.pi is not None:
            if self.pi.size() != self.pi_shape or self.pi.sum(dim=1).item() != 1:
                raise ValueError("Invalid pi provided")
            self.pi = torch.nn.Parameter(self.pi, requires_grad=False)
        else:
            self.pi = torch.nn.Parameter(torch.Tensor(*self.pi_shape), requires_grad=False).fill_(1. / self.n_clusters)
        # validate or set initial cluster means
        if self.mus is not None:
            if self.mus.size() != self.mus_shape:
                raise ValueError("Invalid means provided")
            self.mus = torch.nn.Parameter(self.mus, requires_grad=False)
        else:
            self.mus = torch.nn.Parameter(torch.randn(*self.mus_shape), requires_grad=False)
        # validate or set initial cluster covariance matrices
        if self.sigmas is not None:
            if self.sigmas.size() != self.sigmas_shape:
                raise ValueError("Invalid covariance matrices provided")
            self.sigmas = torch.nn.Parameter(self.sigmas, requires_grad=False)
        else:
            self.sigmas = torch.nn.Parameter(torch.randn(*self.sigmas_shape), requires_grad=False)

        self.precision_cholesky = self._compute_precision_cholesky()

    def fit(self, x):
        for epoch in range(self.max_iter):
            self._em_iteration(x)

    def _estimate_log_prob(self,x):
        n_samples = x.size(dim=0)
        log_det = torch.sum(torch.log(self.precision_cholesky.squeeze().reshape(self.n_clusters,-1)[:,::self.n_features+1]),1)
        log_prob = torch.empty((n_samples, self.n_clusters))
        for k,(mu,prec_chol) in enumerate(zip(self.mus.squeeze(), self.precision_cholesky.squeeze())):
            y = torch.matmul(torch.sub(x,mu),prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y),dim=1)

        return -0.5 * (self.n_features * np.log(2 * np.pi) + log_prob) + log_det # + more data

    def _estimate_weighted_log_prob(self,x):
        return self._estimate_log_prob(x) + torch.log(self.pi)

    def _estimate_log_prob_resp(self,x):
        weighted_log_prob = self._estimate_weighted_log_prob(x)
        log_prob_norm = torch.logsumexp(weighted_log_prob,dim=1)
        log_resp = weighted_log_prob - log_prob_norm
        return log_prob_norm, log_resp

    def _e_step(self, x):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(x)
        return torch.mean(log_prob_norm), log_resp

    def _estimate_parameters(self, x, resp):
        eps = (10 * torch.finfo(resp.type()).eps)
        nk = torch.add(torch.sum(resp), eps)
        resp_x = torch.matmul(resp.transpose(0, 1), x)
        means = torch.div(input=resp_x, other=nk.unsqueeze(1))
        covariances = torch.empty(self.sigmas_shape).squeeze()
        for k in range(self.n_components):
            x_mr = torch.sub(input=x, other=means[k])
            resp_xmrt = torch.matmul(resp[:, k], x_mr.transpose(0, 1))
            resp_xmrt_xmr = torch.matmul(resp_xmrt, x_mr)

            covariances[k] = torch.div(resp_xmrt_xmr, nk[k])
            covariances[k, ::self.n_features + 1] = torch.add(input=covariances[k, ::self.n_features + 1], other=self.reg_covar)
        return nk, means, covariances

    def _m_step(self, x, log_resp):
        self.pi,self.mus, self.sigmas = self._estimate_parameters(x, torch.exp(log_resp))
        torch.div(input=self.pi,other=torch.sum(self.pi),out=self.pi)
        self.precision_cholesky = self._compute_precision_cholesky()

    def _em_iteration(self, x):
        _, log_resp = self._e_step(x)
        self._m_step(x, log_resp)

    def predict_probability(self, x):
        pass

    def _compute_precision_cholesky(self):
        precision_cholesky = torch.empty(self.sigmas_shape).squeeze()
        for k, covar_mat in enumerate(self.sigmas.squeeze()):
            try:
                cov_chol = torch.linalg.cholesky(covar_mat)
            except:
                raise ValueError("increase reg_covar because covariances are bad")
            precision_cholesky[k] = torch.linalg.solve_triangular(cov_chol, torch.eye(self.n_features), upper=False).transpose(0, 1)
        return precision_cholesky.unsqueeze(0)

