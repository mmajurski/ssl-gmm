import copy
import os
import numpy as np
import torch.nn
import torchvision.models
from matplotlib import pyplot as plt


class kmeans_distribution_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int, return_mean_mse: bool = True, return_covar_mse: bool = True):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

        self.return_mean_mse = return_mean_mse
        self.return_covar_mse = return_covar_mse

    def forward(self, x):
        batch = x.size()[0]  # batch size

        # ---
        # Calculate distance to cluster centers
        # ---
        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, num_classes, dim]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, num_classes, dim]
        diff = x_rep - centers_rep

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        dist_sq = diff * diff
        dist_sq = torch.sum(dist_sq, 2)

        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the exponents
        expo_kmeans = -0.5 * dist_sq

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_kmeans, _ = torch.max(expo_kmeans, dim=-1, keepdim=True)
        expo_safe_kmeans = expo_kmeans - expo_safe_off_kmeans  # use broadcast instead of the repeat

        # Calculate the responsibilities kmeans
        numer_safe_kmeans = torch.exp(expo_safe_kmeans)
        denom_safe_kmeans = torch.sum(numer_safe_kmeans, 1, keepdim=True)
        resp_kmeans = numer_safe_kmeans / denom_safe_kmeans  # use broadcast

        # Obtain cluster assignment from dist_sq directly
        cluster_assignment = torch.argmin(dist_sq, dim=-1)
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, dist_sq.shape[1])

        resp_kmeans = torch.clip(resp_kmeans, min=1e-8)
        resp_kmeans = torch.log(resp_kmeans)

        # ----------------------------------------
        # Calculate the empirical cluster mean / covariance
        #   OUTPUT:  empirical_mean  [classes dim]
        #                  cluster centers for the current minibatch
        #   OUTPUT:  empirical_covar [classes dim dim]
        #                  gaussian covariance matrices for the current minibatch
        #   OUTPUT:  cluster_weight  [classes]
        #                  number of samples for each class
        # ----------------------------------------
        cluster_weight = torch.sum(cluster_assignment_onehot, axis=0)
        cluster_assignment_onehot_rep = cluster_assignment_onehot.unsqueeze(2).repeat(1, 1, self.dim)
        x_onehot_rep = x_rep * cluster_assignment_onehot_rep

        #
        # Calculate the empirical mean
        #
        empirical_total = torch.sum(x_onehot_rep, axis=0)
        empirical_count = cluster_weight.unsqueeze(1).repeat(1, self.dim)
        empirical_mean = empirical_total / (empirical_count + 0.0000001)

        #
        # Calculate the empirical covariance
        #
        empirical_mean_rep = empirical_mean.unsqueeze(0).repeat(batch, 1, 1)
        empirical_mean_rep = empirical_mean_rep * cluster_assignment_onehot_rep
        x_mu_rep = x_onehot_rep - empirical_mean_rep

        # perform batch matrix multiplication
        x_mu_rep_B = torch.transpose(x_mu_rep, 0, 1)
        x_mu_rep_A = torch.transpose(x_mu_rep_B, 1, 2)

        empirical_covar_total = torch.bmm(x_mu_rep_A, x_mu_rep_B)
        empirical_covar_count = empirical_count.unsqueeze(2).repeat(1, 1, self.dim)

        empirical_covar = empirical_covar_total / (empirical_covar_count + 0.0000001)

        # ----------------------------------------
        # Calculate a loss distance of the empirical measures from ideal
        # ----------------------------------------

        # ------
        # calculate empirical_mean_loss
        #  weighted L2 loss of each empirical mean from the cluster centers
        # ------

        # calculate empirical weighted dist squares (for means)
        empirical_diff = empirical_mean - self.centers
        empirical_diff_sq = empirical_diff * empirical_diff
        empirical_dist_sq = torch.sum(empirical_diff_sq, axis=1)
        empirical_wei_dist_sq = cluster_weight * empirical_dist_sq

        # create identity covariance of size [class dim dim]
        # identity_covar = torch.eye(self.dim).unsqueeze(0).repeat(self.num_classes,1,1)
        zeros = empirical_covar - empirical_covar
        zeros = torch.sum(zeros, axis=0)
        identity_covar = zeros.fill_diagonal_(1.0)
        identity_covar = identity_covar.unsqueeze(0).repeat(self.num_classes, 1, 1)

        # separate diagonal and off diagonal elements for covar loss
        empirical_covar_diag = empirical_covar * identity_covar
        empirical_covar_off_diag = empirical_covar * (1.0 - identity_covar)

        # calculate diagonal distance squared
        empirical_covar_diag_dist_sq = empirical_covar_diag - identity_covar
        empirical_covar_diag_dist_sq = empirical_covar_diag_dist_sq * empirical_covar_diag_dist_sq
        empirical_covar_diag_dist_sq = torch.sum(empirical_covar_diag_dist_sq, axis=2)
        empirical_covar_diag_dist_sq = torch.sum(empirical_covar_diag_dist_sq, axis=1)

        # calculate diagonal weighted distance squared
        empirical_covar_diag_wei_dist_sq = cluster_weight * empirical_covar_diag_dist_sq / (batch * self.dim)

        # calculate off diagonal distance squared
        empirical_covar_off_diag_dist_sq = empirical_covar_off_diag * empirical_covar_off_diag
        empirical_covar_off_diag_dist_sq = torch.sum(empirical_covar_off_diag_dist_sq, axis=2)
        empirical_covar_off_diag_dist_sq = torch.sum(empirical_covar_off_diag_dist_sq, axis=1)

        # Calculate off-diagonal weighted distance squared
        empirical_covar_off_diag_wei_dist_sq = cluster_weight * empirical_covar_off_diag_dist_sq / (batch * self.dim * (self.dim - 1.0))

        # Add together to get covariance loss
        empirical_covar_dist_sq = empirical_covar_diag_wei_dist_sq + empirical_covar_off_diag_wei_dist_sq

        # ------------------
        # return mean and covariance weighted mse loss
        # ------------------
        empirical_mean_mse = torch.sum(empirical_wei_dist_sq) / (batch * self.dim)
        empirical_covar_mse = torch.sum(empirical_covar_dist_sq)

        # TODO make the above work per image, instead of for the whole batch

        outputs = list()
        outputs.append(resp_kmeans)
        if self.return_mean_mse:
            # outputs.append(empirical_mean_mse)  # outputs a single value
            outputs.append(empirical_wei_dist_sq)
        if self.return_covar_mse:
            # outputs.append(empirical_covar_mse)  # outputs a single value
            outputs.append(empirical_covar_dist_sq)

        return outputs


class kmeans_cmm_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True, return_cluster_dist: bool = True):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

        self.return_gmm = return_gmm
        self.return_cmm = return_cmm
        self.return_cluster_dist = return_cluster_dist

    def forward(self, x):
        batch = x.size()[0]  # batch size

        # ---
        # Calculate distance to cluster centers
        # ---
        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, num_classes, dim]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, num_classes, dim]
        diff = x_rep - centers_rep

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        dist_sq = diff * diff
        dist_sq = torch.sum(dist_sq, 2)

        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the exponents
        expo_kmeans = -0.5 * dist_sq
        a = torch.add(dist_sq, 1)
        b = -((1 + self.dim) / 2)
        expo_cmm = torch.pow(a, b)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_kmeans, _ = torch.max(expo_kmeans, dim=-1, keepdim=True)
        expo_safe_kmeans = expo_kmeans - expo_safe_off_kmeans  # use broadcast instead of the repeat
        expo_safe_off_cmm, _ = torch.max(expo_cmm, dim=-1, keepdim=True)
        expo_safe_cmm = expo_cmm - expo_safe_off_cmm  # use broadcast instead of the repeat

        # Calculate the responsibilities kmeans
        numer_safe_kmeans = torch.exp(expo_safe_kmeans)
        denom_safe_kmeans = torch.sum(numer_safe_kmeans, 1, keepdim=True)
        resp_kmeans = numer_safe_kmeans / denom_safe_kmeans  # use broadcast

        # Calculate the responsibilities cmm
        numer_safe_cmm = torch.exp(expo_safe_cmm)
        denom_safe_cmm = torch.sum(numer_safe_cmm, 1, keepdim=True)
        resp_cmm = numer_safe_cmm / denom_safe_cmm  # use broadcast

        # # Build the kmeans resp
        # dist_sq_kmeans = torch.sum(diff * diff, dim=2)
        # # Obtain the exponents
        # expo = -0.5 * dist_sq_kmeans
        #
        # # Obtain the "safe" (numerically stable) versions of the
        # #  exponents.  These "safe" exponents produce fake numer and denom
        # #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        # #  where fake_numer and fake_denom are numerically stable
        # expo_safe_off = torch.mean(expo, dim=-1, keepdim=True)
        # expo_safe = expo - expo_safe_off
        #
        # # Calculate the responsibilities
        # numer_safe = torch.exp(expo_safe)
        # denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        # resp_kmeans = numer_safe / denom_safe

        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss

        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss
        cluster_assignment = torch.argmax(resp_kmeans, dim=-1)

        # version of cluster_dist which is based on the centroid distance from self.centers
        cluster_dist = torch.zeros_like(resp_kmeans[0, :])
        for c in range(self.num_classes):
            if torch.any(c == cluster_assignment):
                x_centroid = torch.mean(x[c == cluster_assignment, :], dim=0)
                delta = self.centers[c, :] - x_centroid
                delta = torch.sqrt(torch.sum(torch.pow(delta, 2), dim=-1))
                cluster_dist[c] = delta

        resp_kmeans = torch.clip(resp_kmeans, min=1e-8)
        resp_cmm = torch.clip(resp_cmm, min=1e-8)
        resp_kmeans = torch.log(resp_kmeans)
        resp_cmm = torch.log(resp_cmm)

        #		print("resp_cmm")
        #		print(resp_cmm)
        #		input("")

        outputs = list()
        if self.return_gmm:
            outputs.append(resp_kmeans)
        if self.return_cmm:
            outputs.append(resp_cmm)
        if self.return_cluster_dist:
            outputs.append(cluster_dist)

        return outputs


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class axis_aligned_gmm_cmm_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True, return_cluster_dist: bool = True):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        # this is roughly equivalent to init to identity (i.e. kmeans)
        a = torch.rand(size=(self.num_classes, self.dim), requires_grad=True)
        with torch.no_grad():
            a = 0.2 * a + 0.9  # rand of [0.9, 1.1]
        self.D = torch.nn.Parameter(a)

        self.return_gmm = return_gmm
        self.return_cmm = return_cmm
        self.return_cluster_dist = return_cluster_dist

    def forward(self, x):
        batch = x.size()[0]  # batch size

        #
        #  Sigma  = L D Lt
        #
        #  Sigma_inv  = Lt-1 D-1 L-1
        #

        log_det = torch.zeros((self.num_classes), device=x.device, requires_grad=False)
        Sigma_inv = [None] * self.num_classes
        for k in range(self.num_classes):
            D = self.D[k,]  # get the num_classes x 1 vector of covariances
            # ensure positive
            D = torch.abs(D) + 1e-8

            # create inverse of D
            D_inv = 1.0 / D
            # Upsample from Nx1 to NxN diagonal matrix
            D_inv_embed = torch.diag_embed(D_inv)

            Sigma_inv[k] = D_inv_embed

            # Safe version of Determinant
            log_det[k] = torch.sum(torch.log(D), dim=0)

        # Safe det version
        det_scale_factor = -0.5 * log_det
        det_scale_factor_safe = det_scale_factor - torch.max(det_scale_factor)
        det_scale_safe = torch.exp(det_scale_factor_safe)

        # ---
        # Calculate distance to cluster centers
        # ---
        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, num_classes, dim]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, num_classes, dim]
        diff = x_rep - centers_rep

        # Calculate each dist_sq entry separately
        dist_sq = torch.zeros_like(torch.sum(diff, 2))
        for k in range(self.num_classes):
            curr_diff = diff[:, k]
            curr_diff_t = torch.transpose(curr_diff, 0, 1)
            Sig_inv_curr_diff_t = torch.mm(Sigma_inv[k], curr_diff_t)
            Sig_inv_curr_diff_t_t = torch.transpose(Sig_inv_curr_diff_t, 0, 1)
            curr_dist_sq = curr_diff * Sig_inv_curr_diff_t_t
            curr_dist_sq = torch.sum(curr_dist_sq, 1)
            dist_sq[:, k] = curr_dist_sq

        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the exponents
        expo_gmm = -0.5 * dist_sq
        a = torch.add(dist_sq, 1)
        b = -((1 + self.dim) / 2)
        expo_cmm = torch.pow(a, b)

        # Safe version
        det_scale_rep_safe = det_scale_safe.unsqueeze(0).repeat(batch, 1)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_gmm, _ = torch.max(expo_gmm, dim=-1, keepdim=True)
        expo_safe_gmm = expo_gmm - expo_safe_off_gmm  # use broadcast instead of the repeat
        expo_safe_off_cmm, _ = torch.max(expo_cmm, dim=-1, keepdim=True)
        expo_safe_cmm = expo_cmm - expo_safe_off_cmm  # use broadcast instead of the repeat

        # Calculate the responsibilities
        numer_safe_gmm = det_scale_rep_safe * torch.exp(expo_safe_gmm)
        denom_safe_gmm = torch.sum(numer_safe_gmm, 1, keepdim=True)
        resp_gmm = numer_safe_gmm / denom_safe_gmm  # use broadcast


        numer_safe_cmm = det_scale_rep_safe * torch.exp(expo_safe_cmm)
        denom_safe_cmm = torch.sum(numer_safe_cmm, 1, keepdim=True)
        resp_cmm = numer_safe_cmm / denom_safe_cmm  # use broadcast

        # # Build the kmeans resp
        # dist_sq_kmeans = torch.sum(diff * diff, dim=2)
        # # Obtain the exponents
        # expo = -0.5 * dist_sq_kmeans
        #
        # # Obtain the "safe" (numerically stable) versions of the
        # #  exponents.  These "safe" exponents produce fake numer and denom
        # #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        # #  where fake_numer and fake_denom are numerically stable
        # expo_safe_off = torch.mean(expo, dim=-1, keepdim=True)
        # expo_safe = expo - expo_safe_off
        #
        # # Calculate the responsibilities
        # numer_safe = torch.exp(expo_safe)
        # denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        # resp_kmeans = numer_safe / denom_safe

        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss
        cluster_assignment = torch.argmax(resp_gmm, dim=-1)

        # version of cluster_dist which is based on the centroid distance from self.centers
        cluster_dist = torch.zeros_like(resp_gmm[0, :])
        for c in range(self.num_classes):
            if torch.any(c == cluster_assignment):
                x_centroid = torch.mean(x[c==cluster_assignment, :], dim=0)
                delta = self.centers[c, :] - x_centroid
                delta = torch.sqrt(torch.sum(torch.pow(delta, 2), dim=-1))
                cluster_dist[c] = delta

        resp_gmm = torch.clip(resp_gmm, min=1e-8)
        resp_cmm = torch.clip(resp_cmm, min=1e-8)
        resp_gmm = torch.log(resp_gmm)
        resp_cmm = torch.log(resp_cmm)

        outputs = list()
        if self.return_gmm:
            outputs.append(resp_gmm)
        if self.return_cmm:
            outputs.append(resp_cmm)
        if self.return_cluster_dist:
            outputs.append(cluster_dist)

        return outputs


class axis_aligned_gmm_cmm_D1_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True, return_cluster_dist: bool = True):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        self.return_gmm = return_gmm
        self.return_cmm = return_cmm
        self.return_cluster_dist = return_cluster_dist

    def forward(self, x):
        batch = x.size()[0]  # batch size

        #
        #  Sigma  = L D Lt
        #
        #  Sigma_inv  = Lt-1 D-1 L-1
        #

        with torch.no_grad():
            # with D fixed at 1 diagonal, the determininat is the product of all ones, or just one
            det = torch.ones((self.num_classes), device=x.device, requires_grad=False)
            Sigma_inv = torch.eye(self.num_classes, device=x.device, requires_grad=False)

        det_scale = det  #torch.rsqrt(det)  1/sqrt(1) is 1... skip the math

        # ---
        # Calculate distance to cluster centers
        # ---
        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, num_classes, dim]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, num_classes, dim]
        diff = x_rep - centers_rep

        # Calculate each dist_sq entry separately
        dist_sq = torch.zeros_like(torch.sum(diff, 2))
        for k in range(self.num_classes):
            curr_diff = diff[:, k]
            curr_diff_t = torch.transpose(curr_diff, 0, 1)
            Sig_inv_curr_diff_t = torch.mm(Sigma_inv, curr_diff_t)
            Sig_inv_curr_diff_t_t = torch.transpose(Sig_inv_curr_diff_t, 0, 1)
            curr_dist_sq = curr_diff * Sig_inv_curr_diff_t_t
            curr_dist_sq = torch.sum(curr_dist_sq, 1)
            dist_sq[:, k] = curr_dist_sq

        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the exponents
        expo_gmm = -0.5 * dist_sq
        a = torch.add(dist_sq, 1)
        b = -((1 + self.dim) / 2)
        expo_cmm = torch.pow(a, b)

        # Safe version
        det_scale_rep = det_scale.unsqueeze(0).repeat(batch, 1)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_gmm, _ = torch.max(expo_gmm, dim=-1, keepdim=True)
        expo_safe_gmm = expo_gmm - expo_safe_off_gmm  # use broadcast instead of the repeat
        expo_safe_off_cmm, _ = torch.max(expo_cmm, dim=-1, keepdim=True)
        expo_safe_cmm = expo_cmm - expo_safe_off_cmm  # use broadcast instead of the repeat

        # Calculate the responsibilities
        numer_safe_gmm = det_scale_rep * torch.exp(expo_safe_gmm)
        denom_safe_gmm = torch.sum(numer_safe_gmm, 1, keepdim=True)
        resp_gmm = numer_safe_gmm / denom_safe_gmm  # use broadcast

        numer_safe_cmm = det_scale_rep * torch.exp(expo_safe_cmm)
        denom_safe_cmm = torch.sum(numer_safe_cmm, 1, keepdim=True)
        resp_cmm = numer_safe_cmm / denom_safe_cmm  # use broadcast

        # # Build the kmeans resp
        # dist_sq_kmeans = torch.sum(diff * diff, dim=2)
        # # Obtain the exponents
        # expo = -0.5 * dist_sq_kmeans
        #
        # # Obtain the "safe" (numerically stable) versions of the
        # #  exponents.  These "safe" exponents produce fake numer and denom
        # #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        # #  where fake_numer and fake_denom are numerically stable
        # expo_safe_off = torch.mean(expo, dim=-1, keepdim=True)
        # expo_safe = expo - expo_safe_off
        #
        # # Calculate the responsibilities
        # numer_safe = torch.exp(expo_safe)
        # denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        # resp_kmeans = numer_safe / denom_safe


        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss
        cluster_assignment = torch.argmax(resp_gmm, dim=-1)

        # version of cluster_dist which is based on the centroid distance from self.centers
        cluster_dist = torch.zeros_like(resp_gmm[0, :])
        for c in range(self.num_classes):
            if torch.any(c == cluster_assignment):
                x_centroid = torch.mean(x[c==cluster_assignment, :], dim=0)
                delta = self.centers[c, :] - x_centroid
                delta = torch.sqrt(torch.sum(torch.pow(delta, 2), dim=-1))
                cluster_dist[c] = delta

        # #
        # # Vectorized version of cluster_dist
        # #
        # # Obtain cluster assignment from dist_sq directly
        # cluster_assignment = torch.argmin(dist_sq, dim=-1)
        #
        # # Use one-hot encoding trick to extract the dist_sq
        # cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, dist_sq.shape[1])
        # cluster_dist_sq_onehot = cluster_assignment_onehot * dist_sq
        # cluster_dist_sq = torch.sum(cluster_dist_sq_onehot, dim=-1)
        # # Take square root of dist_sq to get L2 norm
        # cluster_dist = torch.sqrt(cluster_dist_sq)


        resp_gmm = torch.clip(resp_gmm, min=1e-8)
        resp_cmm = torch.clip(resp_cmm, min=1e-8)
        resp_gmm = torch.log(resp_gmm)
        resp_cmm = torch.log(resp_cmm)

        outputs = list()
        if self.return_gmm:
            outputs.append(resp_gmm)
        if self.return_cmm:
            outputs.append(resp_cmm)
        if self.return_cluster_dist:
            outputs.append(cluster_dist)

        return outputs


# class gmm_layer(torch.nn.Module):
#     def __init__(self, dim: int, num_classes: int, isCauchy: bool = False):
#         super().__init__()
#
#         self.dim = dim
#         self.num_classes = num_classes
#         self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
#         # does each of these need to start as identity?
#         self.L = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.num_classes, self.dim), requires_grad=True))
#
#         # self.L = torch.nn.Parameter(torch.ones(size=(self.num_classes, self.num_classes, self.dim), requires_grad=True))
#
#         # self.L = torch.nn.Parameter(torch.zeros(size=(self.num_classes, self.num_classes, self.dim), requires_grad=True))
#         # with torch.no_grad():
#         #     for k1 in range(num_classes):
#         #         self.L[k1, k1] = 1.0
#         self.isCauchy = isCauchy
#
#     def forward(self, x):
#         batch = x.size()[0]  # batch size
#
#         #
#         #  Sigma  = L D Lt
#         #
#         #  Sigma_inv  = Lt-1 D-1 L-1
#         #
#
#         det = torch.zeros((self.num_classes), device=x.device, requires_grad=False)
#         Sigma = [None] * self.num_classes
#         Sigma_inv = [None] * self.num_classes
#         for k in range(self.num_classes):
#             raw_mat = self.L[k, ]
#
#             # Construct the L matrix for LDL
#             L = torch.tril(raw_mat, diagonal=-1)
#             L = L.fill_diagonal_(1.0)  # L           (lower triangular)
#             Lt = torch.transpose(L, 0, 1)  # L-transpose (upper triangular)
#
#             # Construct diagonal D which must be positive
#             root_D = torch.diag(raw_mat)
#             D = root_D * root_D + 0.0001  # Diagonal D
#             D_embed = torch.diag_embed(D)  # Upsample to NxN diagonal matrix
#             if torch.any(torch.isnan(D_embed)):
#                 raise RuntimeError("Nan at \"D_embed = torch.diag_embed(D)\"")
#
#             # ---
#             # Construct the Covariance Matrix Sigma
#             # ---
#             LD = torch.mm(L, D_embed)
#             if torch.any(torch.isnan(LD)):
#                 raise RuntimeError("Nan at \"LD = torch.mm(L, D_embed)\"")
#
#             Sigma[k] = torch.mm(LD, Lt)  # k'th Sigma matrix
#             if torch.any(torch.isnan(Sigma[k])):
#                 raise RuntimeError("Nan at \"Sigma[k] = torch.mm(LD, Lt)\"")
#
#             # ---
#             # Construct the inverse Covariance Matrix Sigma_inv
#             # ---
#             Identity = raw_mat - raw_mat
#             Identity.fill_diagonal_(1.0)
#
#             # L inverse
#             L_inv = torch.linalg.solve_triangular(L, Identity, upper=False)
#             if torch.any(torch.isnan(L_inv)):
#                 raise RuntimeError("Nan at \"L_inv = torch.linalg.solve_triangular(L, Identity, upper=False)\"")
#
#             L_inv_t = torch.transpose(L_inv, 0, 1)
#
#             # D inverse
#             D_inv = 1.0 / D
#             D_inv_embed = torch.diag_embed(D_inv)
#             if torch.any(torch.isnan(D_inv_embed)):
#                 raise RuntimeError("Nan at \"D_inv_embed = torch.diag_embed(D_inv)\"")
#
#             # Sigma inverse
#             D_inv_L_inv = torch.mm(D_inv_embed, L_inv)
#             if torch.any(torch.isnan(D_inv_L_inv)):
#                 raise RuntimeError("Nan at \"D_inv_L_inv = torch.mm(D_inv_embed, L_inv)\"")
#
#             Sigma_inv[k] = torch.mm(L_inv_t, D_inv_L_inv)
#             if torch.any(torch.isnan(Sigma_inv[k])):
#                 raise RuntimeError("Nan at \"Sigma_inv[k] = torch.mm(L_inv_t, D_inv_L_inv)\"")
#
#             # Determinant
#             det[k] = torch.prod(D, 0)
#             if torch.any(torch.isnan(det[k])):
#                 raise RuntimeError("Nan at \"det[k] = torch.prod(D, 0)\"")
#
#         det_scale = torch.rsqrt(det)
#         if torch.any(torch.isnan(det_scale)):
#             raise RuntimeError("Nan at \"det_scale = torch.rsqrt(det)\"")
#
#         # ---
#         # Calculate distance to cluster centers
#         # ---
#
#         # Upsample the x-data to [batch, num_classes, dim]
#         x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)
#         if torch.any(torch.isnan(x_rep)):
#             raise RuntimeError("Nan at \"x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)\"")
#
#         # Upsample the clusters to [batch, num_classes, dim]
#         centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)
#         if torch.any(torch.isnan(centers_rep)):
#             raise RuntimeError("Nan at \"centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)\"")
#
#         # Subtract to get diff of [batch, num_classes, dim]
#         diff = x_rep - centers_rep
#         if torch.any(torch.isnan(diff)):
#             raise RuntimeError("Nan at \"diff = x_rep - centers_rep\"")
#
#         # # Obtain the square distance to each cluster
#         # #  of size [batch, num_classes]
#         # dist_sq = diff * diff
#         # dist_sq = torch.sum(dist_sq, 2)
#         # if torch.any(torch.isnan(dist_sq)):
#         #     raise RuntimeError("Nan at \"dist_sq = torch.sum(dist_sq, 2)\"")
#
#         dist_sq = torch.sum(diff, 2)  # initially set to zero
#         dist_sq = dist_sq - dist_sq
#
#
#         # Calculate each dist_sq entry separately
#         for k in range(self.num_classes):
#             curr_diff = diff[:, k]
#             curr_diff_t = torch.transpose(curr_diff, 0, 1)
#             Sig_inv_curr_diff_t = torch.mm(Sigma_inv[k], curr_diff_t)
#             Sig_inv_curr_diff_t_t = torch.transpose(Sig_inv_curr_diff_t, 0, 1)
#             curr_dist_sq = curr_diff * Sig_inv_curr_diff_t_t
#             curr_dist_sq = torch.sum(curr_dist_sq, 1)
#             dist_sq[:, k] = curr_dist_sq
#         if torch.any(torch.isnan(dist_sq)):
#             raise RuntimeError("Nan at \"dist_sq\"")
#
#         #   GMM
#         # dist_sq = (x-mu) Sigma_inv (x-mu)T
#         #   K-means
#         # dist_sq = (x-mu) dot (x-mu)
#
#         # Obtain the square distance to each cluster
#         #  of size [batch, dim]
#         # dist_sq = diff*diff
#         # dist_sq = torch.sum(dist_sq,2)
#
#         # Obtain the exponents
#         if self.isCauchy:
#             a = torch.add(dist_sq,1)
#             b = -((1+self.dim)/2)
#             expo = torch.pow(a, b)
#             if torch.any(torch.isnan(expo)):
#                 raise RuntimeError("Nan at \"-((1+self.dim)/2) * torch.add(dist_sq,1)\"")
#         else:
#             expo = -0.5 * dist_sq
#             if torch.any(torch.isnan(expo)):
#                 raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")
#         # if torch.any(torch.isnan(expo)):
#         #     raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")
#
#         det_scale_rep = det_scale.unsqueeze(0).repeat(batch, 1)
#
#
#         # # Calculate the true numerators and denominators
#         # #  (we don't use this directly for responsibility calculation
#         # #   we actually use the "safe" versions that are shifted
#         # #   for stability)
#         # # Note 0.00010211761 = (2*pi)^(-dim/2) where dim=10
#         # #
#         # numer = 0.00010211761 * torch.exp(expo)
#         # denom = torch.sum(numer, 1)
#         # denom = denom.unsqueeze(1).repeat(1, self.dim)
#
#         # Obtain the "safe" (numerically stable) versions of the
#         #  exponents.  These "safe" exponents produce fake numer and denom
#         #  but guarantee that resp = fake_numer / fake_denom = numer / denom
#         #  where fake_numer and fake_denom are numerically stable
#         # expo_safe_off = self.km_safe_pool(expo)
#         expo_safe_off, _ = torch.max(expo, dim=-1, keepdim=True)
#         expo_safe = expo - expo_safe_off  # use broadcast instead of the repeat
#         if torch.any(torch.isnan(expo_safe)):
#             raise RuntimeError("Nan at \"expo_safe = expo - expo_safe_off\"")
#
#         # TODO create a cauchy version of this resp
#
#         # Calculate the responsibilities
#         if self.isCauchy:
#             numer_safe = det_scale_rep * expo_safe
#             if torch.any(torch.isnan(numer_safe)):
#                 raise RuntimeError("Nan at \"numer_safe = det_scale_rep * expo_safe\"")
#
#         else:
#             numer_safe = det_scale_rep * torch.exp(expo_safe)
#             if torch.any(torch.isnan(numer_safe)):
#                 raise RuntimeError("Nan at \"numer_safe = det_scale_rep * torch.exp(expo_safe)\"")
#
#
#         denom_safe = torch.sum(numer_safe, 1, keepdim=True)
#         resp = numer_safe / denom_safe  # use broadcast
#
#
#         if torch.any(torch.isnan(denom_safe)):
#             raise RuntimeError("Nan at \"denom_safe = torch.sum(numer_safe, 1, keepdim=True)\"")
#         if torch.any(torch.isnan(resp)):
#             raise RuntimeError("Nan at \"resp = numer_safe / denom_safe\"")
#
#
#         # comment out
#         # output = torch.log(resp)
#         output = resp
#
#         return output
#
#
