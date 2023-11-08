import math
import copy
import os
import numpy as np
import torch.nn
import torch.nn.functional as F
import torchvision.models
from matplotlib import pyplot as plt
import gauss_moments



class kmeans(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        # TODO remove if not being used later
        # TODO need to scale this by the variance per cluster
        # self.D = torch.nn.Parameter(torch.ones(size=(self.num_classes, self.dim), requires_grad=True))

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

        resp_kmeans = torch.clip(resp_kmeans, min=1e-8)
        resp_kmeans = torch.log(resp_kmeans)

        return resp_kmeans, 0.0


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.centers = None

    def forward(self, x):
        return x


class axis_aligned_gmm_cmm_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True):
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

        resp_gmm = torch.clip(resp_gmm, min=1e-8)
        resp_cmm = torch.clip(resp_cmm, min=1e-8)
        resp_gmm = torch.log(resp_gmm)
        resp_cmm = torch.log(resp_cmm)

        # Which cluster to assign ?
        assign_gmm = torch.argmax(resp_gmm,dim=1)
        assign_gmm_onehot = F.one_hot(assign_gmm, self.num_classes)

        # Calculate the cross-entroy embedding constraint H(p(Y,Z)||Q)
        #class_contrib = math.log(1.0/self.num_classes)
        ln_2pi = 1.83787706641
        log_det_rep = torch.reshape(log_det, (1,self.num_classes)).repeat((batch,1))
        cross_entropy_embed = -0.5*self.dim*ln_2pi + log_det_rep + expo_gmm
        cross_entropy_embed *= assign_gmm_onehot
        cross_entropy_embed = (-1.0 / batch) * torch.sum(assign_gmm_onehot * cross_entropy_embed)

        if self.return_gmm and self.return_cmm:
            a = torch.nn.functional.softmax(resp_gmm, dim=-1)
            b = torch.nn.functional.softmax(resp_cmm, dim=-1)
            resp = torch.log((a + b) / 2.0)
        else:
            if self.return_gmm:
                resp = resp_gmm
            else:
                resp = resp_cmm

        return resp, cross_entropy_embed


class axis_aligned_gmm_cmm_D1_layer(torch.nn.Module):

    # this is identical to the kmeans layer, but with more complicated math doing the same work

    def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        self.return_gmm = return_gmm
        self.return_cmm = return_cmm

    def forward(self, x):
        batch = x.size()[0]  # batch size

        #
        #  Sigma  = L D Lt
        #
        #  Sigma_inv  = Lt-1 D-1 L-1
        #

        with torch.no_grad():
            # with D fixed at 1 diagonal, the determininat is the product of all ones, or just one
            det_scale = torch.ones(1, device=x.device, requires_grad=False)  #torch.rsqrt(det)  1/sqrt(1) is 1... skip the math
            Sigma_inv = torch.eye(self.dim, device=x.device, requires_grad=False)



        # ---
        # Calculate distance to cluster centers
        # ---
        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, num_classes, dim]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, num_classes, dim]
        diff = x_rep - centers_rep

        dist_sq = diff * diff
        dist_sq = torch.sum(dist_sq, 2)

        # # Calculate each dist_sq entry separately
        # dist_sq = torch.zeros_like(torch.sum(diff, 2))
        # for k in range(self.num_classes):
        #     curr_diff = diff[:, k]
        #     curr_diff_t = torch.transpose(curr_diff, 0, 1)
        #     Sig_inv_curr_diff_t = torch.mm(Sigma_inv, curr_diff_t)
        #     Sig_inv_curr_diff_t_t = torch.transpose(Sig_inv_curr_diff_t, 0, 1)
        #     curr_dist_sq = curr_diff * Sig_inv_curr_diff_t_t
        #     curr_dist_sq = torch.sum(curr_dist_sq, 1)
        #     dist_sq[:, k] = curr_dist_sq

        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the exponents
        expo_gmm = -0.5 * dist_sq
        a = torch.add(dist_sq, 1)
        b = -((1 + self.dim) / 2)
        expo_cmm = torch.pow(a, b)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_gmm, _ = torch.max(expo_gmm, dim=-1, keepdim=True)
        expo_safe_gmm = expo_gmm - expo_safe_off_gmm  # use broadcast instead of the repeat
        expo_safe_off_cmm, _ = torch.max(expo_cmm, dim=-1, keepdim=True)
        expo_safe_cmm = expo_cmm - expo_safe_off_cmm  # use broadcast instead of the repeat

        # Calculate the responsibilities
        numer_safe_gmm = det_scale * torch.exp(expo_safe_gmm)
        denom_safe_gmm = torch.sum(numer_safe_gmm, 1, keepdim=True)
        resp_gmm = numer_safe_gmm / denom_safe_gmm  # use broadcast

        numer_safe_cmm = det_scale * torch.exp(expo_safe_cmm)
        denom_safe_cmm = torch.sum(numer_safe_cmm, 1, keepdim=True)
        resp_cmm = numer_safe_cmm / denom_safe_cmm  # use broadcast

        resp_gmm = torch.clip(resp_gmm, min=1e-8)
        resp_cmm = torch.clip(resp_cmm, min=1e-8)
        resp_gmm = torch.log(resp_gmm)
        resp_cmm = torch.log(resp_cmm)

        if self.return_gmm and self.return_cmm:
            a = torch.nn.functional.softmax(resp_gmm, dim=-1)
            b = torch.nn.functional.softmax(resp_cmm, dim=-1)
            resp = torch.log((a + b) / 2.0)
        else:
            if self.return_gmm:
                resp = resp_gmm
            else:
                resp = resp_cmm

        return resp, 0.0

