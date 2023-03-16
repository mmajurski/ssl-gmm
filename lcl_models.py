import copy
import numpy as np
import torch.nn
import torchvision.models


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class kMeans(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

    def forward(self, x):
        batch = x.size()[0]  # batch size

        # ---
        # Calculate distance to cluster centers
        # ---

        # Upsample the x-data to [batch, dim, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)
        if torch.any(torch.isnan(x_rep)):
            raise RuntimeError("Nan at \"x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)\"")

        # Upsample the clusters to [batch, 10, 10]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)
        if torch.any(torch.isnan(centers_rep)):
            raise RuntimeError("Nan at \"centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)\"")

        # Subtract to get diff of [batch, 10, 10]
        diff = x_rep - centers_rep
        if torch.any(torch.isnan(diff)):
            raise RuntimeError("Nan at \"diff = x_rep - centers_rep\"")

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        dist_sq = diff * diff
        dist_sq = torch.sum(dist_sq, 2)
        if torch.any(torch.isnan(dist_sq)):
            raise RuntimeError("Nan at \"dist_sq = torch.sum(dist_sq, 2)\"")

        # Obtain the exponents
        expo = -0.5 * dist_sq
        if torch.any(torch.isnan(expo)):
            raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")

        # # Calculate the true numerators and denominators
        # #  (we don't use this directly for responsibility calculation
        # #   we actually use the "safe" versions that are shifted
        # #   for stability)
        # # Note 0.00010211761 = (2*pi)^(-dim/2) where dim=10
        # #
        # numer = 0.00010211761 * torch.exp(expo)
        # denom = torch.sum(numer, 1)
        # denom = denom.unsqueeze(1).repeat(1, self.dim)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        # expo_safe_off = self.km_safe_pool(expo)
        expo_safe_off, _ = torch.max(expo, dim=-1, keepdim=True)
        expo_safe = expo - expo_safe_off  # use broadcast instead of the repeat
        if torch.any(torch.isnan(expo_safe)):
            raise RuntimeError("Nan at \"expo_safe = expo - expo_safe_off\"")

        # TODO create a cauchy version of this resp

        # Calculate the responsibilities
        numer_safe = torch.exp(expo_safe)
        denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        resp = numer_safe / denom_safe  # use broadcast

        if torch.any(torch.isnan(numer_safe)):
            raise RuntimeError("Nan at \"numer_safe = torch.exp(expo_safe)\"")
        if torch.any(torch.isnan(denom_safe)):
            raise RuntimeError("Nan at \"denom_safe = torch.sum(numer_safe, 1, keepdim=True)\"")
        if torch.any(torch.isnan(resp)):
            raise RuntimeError("Nan at \"resp = numer_safe / denom_safe\"")

        # comment out
        # output = torch.log(resp)
        output = resp

        return output


class axis_aligned_gmm_layer(torch.nn.Module):
    def __init__(self, embeddign_dim: int, num_classes: int, isCauchy=False):
        super().__init__()

        self.isCauchy = isCauchy
        self.dim = embeddign_dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        # this is roughly equivalent to init to identity (i.e. kmeans)
        a = torch.rand(size=(self.num_classes, self.dim), requires_grad=True)
        with torch.no_grad():
            a = 0.2 * a + 0.9  # rand of [0.9, 1.1]
        self.D = torch.nn.Parameter(a)

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
            D = torch.abs(D) + 1e-4

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

        dist_sq = torch.sum(diff, 2)  # initially set to zero
        dist_sq = dist_sq - dist_sq

        # Calculate each dist_sq entry separately
        # dist_sq = torch.zeros((diff.shape[0], diff.shape[1]), requires_grad=True)
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
        if self.isCauchy:
            a = torch.add(dist_sq, 1)
            b = -((1 + self.dim) / 2)
            expo = torch.pow(a, b)
        else:
            expo = -0.5 * dist_sq

        # Safe version
        det_scale_rep_safe = det_scale_safe.unsqueeze(0).repeat(batch, 1)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        # expo_safe_off = self.km_safe_pool(expo)
        expo_safe_off, _ = torch.max(expo, dim=-1, keepdim=True)
        expo_safe = expo - expo_safe_off  # use broadcast instead of the repeat

        # TODO verify the cauchy version

        # Calculate the responsibilities
        if self.isCauchy:
            numer_safe = det_scale_rep_safe * expo_safe
        else:
            numer_safe = det_scale_rep_safe * torch.exp(expo_safe)
        denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        resp = numer_safe / denom_safe  # use broadcast

        # output = torch.log(resp)
        output = resp

        return output


class gmm_layer(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int, isCauchy: bool = False):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        # does each of these need to start as identity?
        self.L = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.num_classes, self.dim), requires_grad=True))

        # self.L = torch.nn.Parameter(torch.ones(size=(self.num_classes, self.num_classes, self.dim), requires_grad=True))

        # self.L = torch.nn.Parameter(torch.zeros(size=(self.num_classes, self.num_classes, self.dim), requires_grad=True))
        # with torch.no_grad():
        #     for k1 in range(num_classes):
        #         self.L[k1, k1] = 1.0
        self.isCauchy = isCauchy

    def forward(self, x):
        batch = x.size()[0]  # batch size

        #
        #  Sigma  = L D Lt
        #
        #  Sigma_inv  = Lt-1 D-1 L-1
        #

        det = torch.zeros((self.num_classes), device=x.device, requires_grad=False)
        Sigma = [None] * self.num_classes
        Sigma_inv = [None] * self.num_classes
        for k in range(self.num_classes):
            raw_mat = self.L[k, ]

            # Construct the L matrix for LDL
            L = torch.tril(raw_mat, diagonal=-1)
            L = L.fill_diagonal_(1.0)  # L           (lower triangular)
            Lt = torch.transpose(L, 0, 1)  # L-transpose (upper triangular)

            # Construct diagonal D which must be positive
            root_D = torch.diag(raw_mat)
            D = root_D * root_D + 0.0001  # Diagonal D
            D_embed = torch.diag_embed(D)  # Upsample to NxN diagonal matrix
            if torch.any(torch.isnan(D_embed)):
                raise RuntimeError("Nan at \"D_embed = torch.diag_embed(D)\"")

            # ---
            # Construct the Covariance Matrix Sigma
            # ---
            LD = torch.mm(L, D_embed)
            if torch.any(torch.isnan(LD)):
                raise RuntimeError("Nan at \"LD = torch.mm(L, D_embed)\"")

            Sigma[k] = torch.mm(LD, Lt)  # k'th Sigma matrix
            if torch.any(torch.isnan(Sigma[k])):
                raise RuntimeError("Nan at \"Sigma[k] = torch.mm(LD, Lt)\"")

            # ---
            # Construct the inverse Covariance Matrix Sigma_inv
            # ---
            Identity = raw_mat - raw_mat
            Identity.fill_diagonal_(1.0)

            # L inverse
            L_inv = torch.linalg.solve_triangular(L, Identity, upper=False)
            if torch.any(torch.isnan(L_inv)):
                raise RuntimeError("Nan at \"L_inv = torch.linalg.solve_triangular(L, Identity, upper=False)\"")

            L_inv_t = torch.transpose(L_inv, 0, 1)

            # D inverse
            D_inv = 1.0 / D
            D_inv_embed = torch.diag_embed(D_inv)
            if torch.any(torch.isnan(D_inv_embed)):
                raise RuntimeError("Nan at \"D_inv_embed = torch.diag_embed(D_inv)\"")

            # Sigma inverse
            D_inv_L_inv = torch.mm(D_inv_embed, L_inv)
            if torch.any(torch.isnan(D_inv_L_inv)):
                raise RuntimeError("Nan at \"D_inv_L_inv = torch.mm(D_inv_embed, L_inv)\"")

            Sigma_inv[k] = torch.mm(L_inv_t, D_inv_L_inv)
            if torch.any(torch.isnan(Sigma_inv[k])):
                raise RuntimeError("Nan at \"Sigma_inv[k] = torch.mm(L_inv_t, D_inv_L_inv)\"")

            # Determinant
            det[k] = torch.prod(D, 0)
            if torch.any(torch.isnan(det[k])):
                raise RuntimeError("Nan at \"det[k] = torch.prod(D, 0)\"")

        det_scale = torch.rsqrt(det)
        if torch.any(torch.isnan(det_scale)):
            raise RuntimeError("Nan at \"det_scale = torch.rsqrt(det)\"")

        # ---
        # Calculate distance to cluster centers
        # ---

        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)
        if torch.any(torch.isnan(x_rep)):
            raise RuntimeError("Nan at \"x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)\"")

        # Upsample the clusters to [batch, num_classes, dim]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)
        if torch.any(torch.isnan(centers_rep)):
            raise RuntimeError("Nan at \"centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)\"")

        # Subtract to get diff of [batch, num_classes, dim]
        diff = x_rep - centers_rep
        if torch.any(torch.isnan(diff)):
            raise RuntimeError("Nan at \"diff = x_rep - centers_rep\"")

        # # Obtain the square distance to each cluster
        # #  of size [batch, num_classes]
        # dist_sq = diff * diff
        # dist_sq = torch.sum(dist_sq, 2)
        # if torch.any(torch.isnan(dist_sq)):
        #     raise RuntimeError("Nan at \"dist_sq = torch.sum(dist_sq, 2)\"")

        dist_sq = torch.sum(diff, 2)  # initially set to zero
        dist_sq = dist_sq - dist_sq


        # Calculate each dist_sq entry separately
        for k in range(self.num_classes):
            curr_diff = diff[:, k]
            curr_diff_t = torch.transpose(curr_diff, 0, 1)
            Sig_inv_curr_diff_t = torch.mm(Sigma_inv[k], curr_diff_t)
            Sig_inv_curr_diff_t_t = torch.transpose(Sig_inv_curr_diff_t, 0, 1)
            curr_dist_sq = curr_diff * Sig_inv_curr_diff_t_t
            curr_dist_sq = torch.sum(curr_dist_sq, 1)
            dist_sq[:, k] = curr_dist_sq
        if torch.any(torch.isnan(dist_sq)):
            raise RuntimeError("Nan at \"dist_sq\"")

        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        # dist_sq = diff*diff
        # dist_sq = torch.sum(dist_sq,2)

        # Obtain the exponents
        if self.isCauchy:
            a = torch.add(dist_sq,1)
            b = -((1+self.dim)/2)
            expo = torch.pow(a, b)
            if torch.any(torch.isnan(expo)):
                raise RuntimeError("Nan at \"-((1+self.dim)/2) * torch.add(dist_sq,1)\"")
        else:
            expo = -0.5 * dist_sq
            if torch.any(torch.isnan(expo)):
                raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")
        # if torch.any(torch.isnan(expo)):
        #     raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")

        det_scale_rep = det_scale.unsqueeze(0).repeat(batch, 1)


        # # Calculate the true numerators and denominators
        # #  (we don't use this directly for responsibility calculation
        # #   we actually use the "safe" versions that are shifted
        # #   for stability)
        # # Note 0.00010211761 = (2*pi)^(-dim/2) where dim=10
        # #
        # numer = 0.00010211761 * torch.exp(expo)
        # denom = torch.sum(numer, 1)
        # denom = denom.unsqueeze(1).repeat(1, self.dim)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        # expo_safe_off = self.km_safe_pool(expo)
        expo_safe_off, _ = torch.max(expo, dim=-1, keepdim=True)
        expo_safe = expo - expo_safe_off  # use broadcast instead of the repeat
        if torch.any(torch.isnan(expo_safe)):
            raise RuntimeError("Nan at \"expo_safe = expo - expo_safe_off\"")

        # TODO create a cauchy version of this resp

        # Calculate the responsibilities
        if self.isCauchy:
            numer_safe = det_scale_rep * expo_safe
            if torch.any(torch.isnan(numer_safe)):
                raise RuntimeError("Nan at \"numer_safe = det_scale_rep * expo_safe\"")

        else:
            numer_safe = det_scale_rep * torch.exp(expo_safe)
            if torch.any(torch.isnan(numer_safe)):
                raise RuntimeError("Nan at \"numer_safe = det_scale_rep * torch.exp(expo_safe)\"")


        denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        resp = numer_safe / denom_safe  # use broadcast


        if torch.any(torch.isnan(denom_safe)):
            raise RuntimeError("Nan at \"denom_safe = torch.sum(numer_safe, 1, keepdim=True)\"")
        if torch.any(torch.isnan(resp)):
            raise RuntimeError("Nan at \"resp = numer_safe / denom_safe\"")


        # comment out
        # output = torch.log(resp)
        output = resp

        return output


