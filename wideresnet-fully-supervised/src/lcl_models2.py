import copy
import os
import numpy as np
import torch.nn
import torchvision.models
from matplotlib import pyplot as plt
import gauss_moments

class moments_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

        # ----------
        # The moments
        # ----------
        moment_1 = gauss_moments.GaussMoments(self.dim, 1)  # position
        moment_2 = gauss_moments.GaussMoments(self.dim, 2)  # variance
        moment_3 = gauss_moments.GaussMoments(self.dim, 3)  # skew
        moment_4 = gauss_moments.GaussMoments(self.dim, 4)  # kutorsis

        # moment weights (for moment loss function)
        self.moment1_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)
        self.moment2_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)
        self.moment3_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)
        self.moment4_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)

        # gaussian moments
        self.gauss_moments1 = torch.nn.Parameter(torch.tensor(moment_1.joint_gauss_moments), requires_grad=False)
        self.gauss_moments2 = torch.nn.Parameter(torch.tensor(moment_2.joint_gauss_moments), requires_grad=False)
        self.gauss_moments3 = torch.nn.Parameter(torch.tensor(moment_3.joint_gauss_moments), requires_grad=False)
        self.gauss_moments4 = torch.nn.Parameter(torch.tensor(moment_4.joint_gauss_moments), requires_grad=False)

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

        # Obtain the exponents
        expo = -0.5 * dist_sq

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        dist_sq = diff * diff
        dist_sq = torch.sum(dist_sq, 2)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off = self.km_safe_pool(expo)
        expo_safe_off = expo_safe_off.repeat(1, self.num_classes)
        expo_safe = expo - expo_safe_off

        # Calculate the responsibilities
        numer_safe = torch.exp(expo_safe)
        denom_safe = torch.sum(numer_safe, 1)
        denom_safe = denom_safe.unsqueeze(1).repeat(1, self.num_classes)
        resp = numer_safe / denom_safe

        # Obtain cluster assignment from dist_sq directly
        cluster_assignment = torch.argmin(dist_sq, dim=-1)
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, dist_sq.shape[1])

        # -------------------------------------
        # The moments penalty
        # -------------------------------------
        # ----------------------------------------
        # Calculate the empirical moments
        #   OUTPUT:  moment1  [classes dim]
        #   OUTPUT:  moment2  [classes dim dim]
        #   OUTPUT:  moment3  [classes dim dim dim]
        #   OUTPUT:  moment4  [classes dim dim dim dim]
        # ----------------------------------------
        cluster_weight = torch.sum(cluster_assignment_onehot, axis=0)
        cluster_assignment_onehot_rep = cluster_assignment_onehot.unsqueeze(2).repeat(1, 1, self.dim)

        diff_onehot = diff * cluster_assignment_onehot_rep

        moment1 = torch.sum(diff_onehot, axis=0)
        moment1_count = cluster_weight.unsqueeze(1).repeat(1, self.dim)
        moment1 = moment1 / (moment1_count + 0.0000001)

        moment2_a = diff_onehot.unsqueeze(2)
        moment2_b = diff_onehot.unsqueeze(3)
        moment2_a_rep = moment2_a.repeat((1, 1, self.dim, 1))
        moment2_b_rep = moment2_b.repeat((1, 1, 1, self.dim))
        moment2 = moment2_a_rep * moment2_b_rep
        moment2 = torch.sum(moment2, axis=0)
        moment2_count = moment1_count.unsqueeze(2).repeat((1, 1, self.dim))
        moment2 = moment2 / (moment2_count + 0.0000001)

        moment3_a = moment2_a.unsqueeze(2)
        moment3_b = moment2_b.unsqueeze(2)
        moment3_c = moment2_b.unsqueeze(4)
        moment3_a_rep = moment3_a.repeat((1, 1, self.dim, self.dim, 1))
        moment3_b_rep = moment3_b.repeat((1, 1, self.dim, 1, self.dim))
        moment3_c_rep = moment3_c.repeat((1, 1, 1, self.dim, self.dim))
        moment3 = moment3_a_rep * moment3_b_rep * moment3_c_rep
        moment3 = torch.sum(moment3, axis=0)

        moment4_a = moment3_a.unsqueeze(2)
        moment4_b = moment3_b.unsqueeze(2)
        moment4_c = moment3_c.unsqueeze(2)
        moment4_d = moment3_c.unsqueeze(5)
        moment4_a_rep = moment4_a.repeat((1, 1, self.dim, self.dim, self.dim, 1))
        moment4_b_rep = moment4_b.repeat((1, 1, self.dim, self.dim, 1, self.dim))
        moment4_c_rep = moment4_c.repeat((1, 1, self.dim, 1, self.dim, self.dim))
        moment4_d_rep = moment4_d.repeat((1, 1, 1, self.dim, self.dim, self.dim))
        moment4 = moment4_a_rep * moment4_b_rep * moment4_c_rep * moment4_d_rep
        moment4 = torch.sum(moment4, axis=0)

        # ---------------------------------------
        # calculate the moment loss
        # ---------------------------------------

        # get the moment targets
        moment1_target = self.gauss_moments1
        moment2_target = self.gauss_moments2
        moment3_target = self.gauss_moments3
        moment4_target = self.gauss_moments4

        # normalize the moments with the "magic formula"
        #  that keeps the values from growing at the rate
        #  of x^2 x^3 .... etc
        #
        #  N(x) = sign(x)(abs(x) + c)^a - b
        #	 where
        #  c = pow(a, 1/(1-a))
        #  b = pow(a, a/(1-a))
        #
        #  precomputed values
        #	moment 1   no formula required, it's perfectly linear
        #	moment 2   a = 1/2  c = 0.25		   b = 0.5
        #	moment 3   a = 1/3  c = 0.19245008973  b = 0.57735026919
        #	moment 4   a = 1/4  c = 0.15749013123  b = 0.62996052494
        moment2 = torch.sign(torch.sign(moment2) + 0.1) * (torch.pow(torch.abs(moment2) + 0.25, 0.5) - 0.5)
        moment3 = torch.sign(torch.sign(moment3) + 0.1) * (torch.pow(torch.abs(moment3) + 0.19245008973, 0.3333333333) - 0.57735026919)
        moment4 = torch.sign(torch.sign(moment4) + 0.1) * (torch.pow(torch.abs(moment4) + 0.15749013123, 0.25) - 0.62996052494)
        moment2_target = torch.sign(torch.sign(moment2_target) + 0.1) * (torch.pow(torch.abs(moment2_target) + 0.25, 0.5) - 0.5)
        moment3_target = torch.sign(torch.sign(moment3_target) + 0.1) * (torch.pow(torch.abs(moment3_target) + 0.19245008973, 0.3333333333) - 0.57735026919)
        moment4_target = torch.sign(torch.sign(moment4_target) + 0.1) * (torch.pow(torch.abs(moment4_target) + 0.15749013123, 0.25) - 0.62996052494)

        # repeat the moment targets per class
        moment1_target = moment1_target.unsqueeze(0).repeat(self.num_classes, 1)
        moment2_target = moment2_target.unsqueeze(0).repeat(self.num_classes, 1, 1)
        moment3_target = moment3_target.unsqueeze(0).repeat(self.num_classes, 1, 1, 1)
        moment4_target = moment4_target.unsqueeze(0).repeat(self.num_classes, 1, 1, 1, 1)

        # repeat the moment penalty weights perclass
        cluster_weight_norm = cluster_weight / torch.sum(cluster_weight)

        cluster_weight_rep = cluster_weight_norm.unsqueeze(1).repeat((1, self.dim))
        moment1_weight = cluster_weight_rep * self.moment1_weight.unsqueeze(0).repeat((self.num_classes, 1))

        cluster_weight_rep = cluster_weight_rep.unsqueeze(2).repeat((1, 1, self.dim))
        moment2_weight = cluster_weight_rep * self.moment2_weight.unsqueeze(0).repeat((self.num_classes, 1, 1))

        cluster_weight_rep = cluster_weight_rep.unsqueeze(3).repeat((1, 1, 1, self.dim))
        moment3_weight = cluster_weight_rep * self.moment3_weight.unsqueeze(0).repeat((self.num_classes, 1, 1, 1))

        cluster_weight_rep = cluster_weight_rep.unsqueeze(4).repeat((1, 1, 1, 1, self.dim))
        moment4_weight = cluster_weight_rep * self.moment4_weight.unsqueeze(0).repeat((self.num_classes, 1, 1, 1, 1))

        # calculate the penalty loss function
        moment_penalty1 = torch.sum(moment1_weight * torch.pow((moment1 - moment1_target), 2))
        moment_penalty2 = torch.sum(moment2_weight * torch.pow((moment2 - moment2_target), 2))
        moment_penalty3 = torch.sum(moment3_weight * torch.pow((moment3 - moment3_target), 2))
        moment_penalty4 = torch.sum(moment4_weight * torch.pow((moment4 - moment4_target), 2))

        # MoM loss
        mom_penalty = 1.0 * moment_penalty1 + \
                      0.5 * moment_penalty2 + \
                      0.25 * moment_penalty3 + \
                      0.125 * moment_penalty4

        return resp, mom_penalty



class kmeans_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int):
        super().__init__()

        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

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

        # TODO build a version of this that is aa_gmm and aa_d1_gmm

        return resp_kmeans


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

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

        if self.return_gmm and self.return_cmm:
            a = torch.nn.functional.softmax(resp_gmm, dim=-1)
            b = torch.nn.functional.softmax(resp_cmm, dim=-1)
            resp = torch.log((a + b) / 2.0)
        else:
            if self.return_gmm:
                resp = resp_gmm
            else:
                resp = resp_cmm

        return resp


class axis_aligned_gmm_cmm_D1_layer(torch.nn.Module):
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

        return resp

