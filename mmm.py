import numpy as np
import math
import torch
from logger import GMMLogger

class GMM:
    def __init__(self, dataset, num_clusters, tolerance=0, num_iterations=100):
        '''
        :param dataset: type: torch.Tensor
        :param num_clusters: type: int
        :param tolerance: if non-zero check for convergence with the given tolerance, 
            otherwise do not check for convergence, type: float
        :param num_iterations: number of times to iterate EM, type: int
        mu: mean vector for each cluster. Select k points from the dataset
            to initialize mean vectors, 
            type: list of tensors k*torch.Tensor(features)
        cov: list of covariance matrices for each cluster, 
            initialized by setting each one of K covariance matrix equal to the
            covariance of the entire dataset
            type: list of tensors k*(torch.Tensor(features, features))
        priors: subjective priors assigned to clusters type: torch.Tensor(k)
        likelihoods: multivariate gaussian distributions
            type: torch.Tensor(num_samples, features)
        N: number of samples in the dataset, type: int
        '''

        self.X = dataset
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.k = num_clusters
        self.tolerance = tolerance
        self.converge = (tolerance != 0)
        self.max_epochs = num_iterations
        self.cur_epoch = 0
        self.ll_prev = 0

        # chooses k random indices out of N
        self.mu = self.X[torch.from_numpy(np.random.choice(self.N, self.k, replace=False)).long()].double()
        
        # sets mixture covariance matrix to initialize cov_mat of each cluster
        x_norm = self.X - torch.mean(self.X, 0).double()

        x_cov = ((x_norm.t() @ x_norm) / (self.N - 1)).double()

        self.cov = self.k * [x_cov]
        self.numpy_cov = None
        # self.cov = torch.ones((self.d, self.d))

        # for inverse of covariance_matrices
        self.cov_inv = self.k * [x_cov]

        # for determinant of covariance matrices
        x_cov_det = torch.det(x_cov)
        
        self.cov_det = self.k * [x_cov_det] 

        #cluster probabilities
        self.priors = torch.empty(self.k).fill_(1. / self.k).double()  # uniform priors

        # datapoint cluster responsibilities
        self.likelihoods = torch.zeros((self.N, self.k)).double()

        #const_terms

        self.dlog2pi = self.d * math.log(2 * math.pi)

        #logger object

        self.logger = GMMLogger()

    def log_pdf(self, x_mr, k_id):
        """
        Def:
        Computes log probability of Multivariate Normal PDF
        \mathbf{M} =\mathbf{(x-\mu)}^\top\mathbf{\Sigma}^{-1}\mathbf{(x-\mu)}
        log(\mathcal{N}(x|\mu, \Sigma)) = - 0.5 * (d\ log(2\pi)\ +\ log(|\Sigma|)\ +\ \mathbf{M})

        :param x_mr: type: torch.Tensor(1, features)
        :param k_id: type: int
        """
        #calculate Mahalanobis distance
        M = torch.matmul(torch.matmul(x_mr.t(), self.cov_inv[k_id]), x_mr) 
        #\mathbf{M} =\mathbf{(x-\mu)}^\top\mathbf{\Sigma}^{-1}\mathbf{(x-\mu)} \\
        #log(\mathcal{N}(x|\mu, \Sigma)) = - 0.5 * (d\ log(2\pi)\ +\ log(|\Sigma|)\ +\ \mathbf{M})     
        return  -0.5 * ( self.dlog2pi + torch.log(self.cov_det[k_id]) + M) 


    def predict_probs(self, x, mus, covs, log_prob=True):
        '''
        Def:
        Computes the (log if log_prob==True) probability of data point in x belonging to cluster defined by mus and covs pairs
        Purpose: for Prediction and Testing uses
        :param x: type: torch.Tensor(num_samples, features)
        :param mus: type: torch.Tensor(num_clusters, features)
        :param covs: type: torch.Tensor(num_clusters, features, features)
        :param log_prob: type: Boolean default:True

        :returns results: type: torch.Tensor(num_samples, num_clusters)
        '''

        k = mus.shape[0] #num_clusters

        n = x.shape[0]  #num_samples

        result = torch.zeros((n, k)).double()

        for i in range(k):
            #get mean reduced data point
            x_mr = x - mus[i]

            #get covariance inverse and det
            cov_inv = torch.inverse(covs[i])

            #get determinant of  cov matrix
            cov_det = torch.det(covs[i])

            for j in range(n):
                    
                #calculate Mahalanobis distance
                M = torch.matmul(torch.matmul(x_mr[j].t(), cov_inv), x_mr[j])

                result[j, i] = -0.5 * ( self.dlog2pi + torch.log(cov_det) + M) 

        # standard probability if log_prob is False
        if not log_prob:
            result = result.exp_()
        
        return result.double()


    def e_step(self):
        '''
        Def:
        Computes the likelihoods and posteriors
        :return posteriors: type: torch.Tensor(num_samples, clusters) or None if converged
        '''

        for j in range(self.k):
            # Mean reduced data points
            X_mr = self.X - self.mu[j]

            # Inverse cov. Matrix of cluster j
            self.cov_inv[j] = torch.inverse(self.cov[j])

            # Determinant of cov. Matrix of cluster j
            self.cov_det[j] = torch.det(self.cov[j])
            
            #calculate likelihoods
            for i, data in enumerate(X_mr):
                self.likelihoods[i, j] = self.log_pdf(data, j).exp_().double()
        
        # posteriors based on likelihoods and priors
        posteriors = torch.mul(self.likelihoods, self.priors)
        
        #normalization vector
        norm_vec = torch.sum(posteriors, 1)

        #normalized posteriors
        posteriors = torch.div(posteriors.t(), norm_vec).t()

        #logging
        self.logger.add_epoch_data("means", self.mu.tolist())
        self.logger.add_epoch_data("covariance_mats", list([mat.tolist() for mat in self.cov]))
        self.logger.add_epoch_data("priors", self.priors.tolist())

        #checking for convergence
        if self.converge:
            ll_new = torch.sum(torch.log(norm_vec)).item()
            if abs(ll_new - self.ll_prev) <= self.tolerance:
                return None
            self.ll_prev = ll_new

        # exit()
        return posteriors


    def m_step(self,posteriors, eps=1e-6, min_var=1e-3):
        '''
        Def:
        Sets new mean, covariance and prior to the Gaussian clusters
        :param posteriors: type: torch.Tensor(num_samples, clusters)
        :param eps: to avoid getting NaN, type: double
        :param min_var: type: double
        :return mu_updated: updated means, 
            type: list of tensors k*torch.Tensor(features)
        :return cov_updated: updated covariance matrices, 
            list of tensors k*(torch.Tensor(features, features))
        '''
        
        #initialization
        mu_updated = torch.zeros((self.k, self.d)).double()
        cov_updated = torch.zeros((self.k, self.d, self.d)).double()

        norm = torch.sum(posteriors, dim=0) + eps  # normalizer (k)
        for i in range(self.k):
            for j, data in enumerate(self.X):
                mu_updated[i] += data.double() * posteriors[j, i]
            #mu_updated[i] /= norm[i]
        
        #mu_updated = torch.div(mu_updated.t(), norm).t() # (k, d)
        mu_updated = torch.transpose(mu_updated, 0, 1)
        mu_updated = torch.div(mu_updated, norm)
        mu_updated = torch.transpose(mu_updated, 0, 1)  # (k, d)

        for i in range(self.k):
            for j, data in enumerate(self.X):
                a = (data.double() - mu_updated[i]).view(self.d, 1)
                b = (data.double() - mu_updated[i]).view(1, self.d)
                c = torch.matmul(a, b)
                d = c * posteriors[j, i]
                cov_updated[i] += d

                # https://stackoverflow.com/questions/53370003/pytorch-mapping-operators-to-functions
                # cov_updated[i] += ((data.double() - mu_updated[i]).view(self.d, 1) @
                #                    (data.double() - mu_updated[i]).view(1, self.d)) \
                #                   * posteriors[j, i]

            cov_updated[i] = cov_updated[i] / norm[i]
            cov_updated[i] = torch.clamp(cov_updated[i], min=min_var)  # (d,d)
            # cov_updated[i] = torch.clamp((cov_updated[i] / (norm[i])), min=min_var) # (d, d)

            self.priors[i] = norm[i] / self.N #(k)

        
        return mu_updated, cov_updated

    def convergence(self):
        '''
        runs until convergence or till max number of epochs
        :return mu: type: torch.Tensor (clusters,features)
        :return cov: type: torch.Tensor (clusters,features,features)
        '''
        while self.cur_epoch < self.max_epochs:
            ret_val = self.run_epoch()
            #if converged then stop
            if ret_val is None:
                break
        #self.logger.export()
        return self.mu, self.cov

    def run(self, epochs):
        '''
        runs given number of epochs
        :return mu: type: torch.Tensor (cluster,features)
        :return cov: type: torch.Tensor (cluster,features,features)
        '''
        for _ in range(epochs):
            ret_val = self.run_epoch()
            #if converged then stop
            if ret_val is None:
                break
        return self.mu, self.cov

    def run_epoch(self):
        '''
        runs one epoch
        :return None: if converged during the epoch
        :return True: for successful completion of full epoch.
        '''
        self.logger.new_epoch(self.cur_epoch)
        posteriors = self.e_step()
        if posteriors is None: #if converged
            return None
        mu_updated, cov_updated = self.m_step(posteriors)
        self.mu, self.cov = mu_updated, cov_updated
        self.numpy_cov = self.cov.detach().cpu().numpy()
        self.cur_epoch += 1
        return True #upon successful completion of epoch
