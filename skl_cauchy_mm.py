import numpy as np
import sklearn.mixture
from math import gamma


class CMM(sklearn.mixture.GaussianMixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cauchy_estimate_log_prob(self, x):
        n_samples = x.shape[0]
        n_features = self.means_.shape[0]
        n_clusters = self.n_components

        power_value = (1 + n_features) / 2
        numerator = np.log(gamma(power_value))
        denom_1 = np.log(gamma(1 / 2))
        denom_2 = (n_features / 2) * np.log(np.pi)

        log_det = np.sum(np.log(self.precisions_cholesky_.reshape(n_clusters, -1)[:, ::n_features + 1]), 1)

        # mahalanobis distance
        m_dist = np.empty((n_samples, n_clusters))

        for k, (mu, prec_chol) in enumerate(zip(self.means_, self.precisions_cholesky_)):
            y = np.matmul(np.subtract(x, mu), prec_chol)
            m_dist[:, k] = np.sum(np.square(y), axis=1)

        denom_3 = log_det * 0.5

        denom_4 = np.log(m_dist) * power_value

        tensor_part = -1 * (denom_4 + denom_3)

        scalar_part = numerator - denom_1 - denom_2

        log_pdf = tensor_part + scalar_part

        return log_pdf

    def _cauchy_estimate_weighted_log_prob(self, x):
        return self._cauchy_estimate_log_prob(x) + np.log(self.weights_)

    def _cauchy_estimate_log_prob_resp(self, x):
        weighted_log_prob = self._cauchy_estimate_weighted_log_prob(x)
        # log(sum(exp(weighted_log_prob elements across dim=1)))
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1, keepdims=True))

        # equivalent of dividing by sum in the normal space
        # verify if we need to normalize in case of 1 cluster? as it is returning all 0s in log space (1 in normal)
        log_resp = weighted_log_prob - log_prob_norm
        return log_prob_norm, log_resp, weighted_log_prob

    def predict_cauchy_probability(self, x):
        log_prob_norm, log_resp, unnorm_log_resp = self._cauchy_estimate_log_prob_resp(x)
        return np.exp(log_prob_norm), np.exp(log_resp), unnorm_log_resp