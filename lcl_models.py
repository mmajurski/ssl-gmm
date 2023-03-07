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

        # TODO force the nan with high learning rate and debug whats causing the nan

        # comment out
        # output = torch.log(resp)
        output = resp

        return output


class axis_aligned_gmm_layer(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))
        a = torch.rand(size=(self.num_classes, self.dim), requires_grad=True)
        with torch.no_grad():
            a = 0.2*a + 0.9  # rand of 1.0+-0.1

        # a = torch.ones(size=(self.num_classes, self.dim), requires_grad=True)
        self.D = torch.nn.Parameter(a)

        # self.D = torch.nn.Parameter(torch.clip(a, min=0.8, max=1.2))

        self.counter = 0


    def forward(self, x):
        batch = x.size()[0]  # batch size

        #
        #  Sigma  = L D Lt
        #
        #  Sigma_inv  = Lt-1 D-1 L-1
        #

        self.counter += 1
        if self.counter % 1000 == 0:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.hist(self.D.detach().cpu().numpy().reshape(-1), bins=100, label='D')
            plt.savefig('D-hist.png')
            plt.close()



        # TODO do we need to add a compactness criteria to the loss to encourage high quality clusters?

        det = torch.zeros((self.num_classes), device=x.device, requires_grad=False)
        Sigma_inv = [None] * self.num_classes
        for k in range(self.num_classes):
            root_D = self.D[k, ]  # get the num_classes x 1 vector of covariances
            # ensure positive
            # D = root_D * root_D + 1e-4
            D = torch.abs(root_D) + 1e-4  # multiplying causes the value to get smaller if <1, which causes determinant problems. So let pytorch use a subgradient for abs
            # Upsample from Nx1 to NxN diagonal matrix
            # D_embed = torch.diag_embed(D)

            # create inverse of D
            D_inv = 1.0 / D
            D_inv_embed = torch.diag_embed(D_inv)

            # Sigma[k] = D_embed
            Sigma_inv[k] = D_inv_embed

            # Determinant
            det[k] = torch.prod(D, dim=0)  # difficulty keeping non 0 for large dim, as are multiplying many potentially small numbers together
            if torch.any(torch.isnan(det[k])):
                raise RuntimeError("Nan at \"det[k] = torch.prod(D, 0)\"")

        det_scale = torch.rsqrt(det)
        if torch.any(torch.isnan(det_scale)) or torch.any(torch.isinf(det_scale)):
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

        if self.counter % 1000 == 0:
            min_dist, _ = torch.min(dist_sq, dim=1)
            min_dist = min_dist.detach().cpu().numpy()

            avg_dist = torch.mean(dist_sq, dim=1)
            avg_dist = avg_dist.detach().cpu().numpy()
            from matplotlib import pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.hist(min_dist.reshape(-1), bins=100, label='Min Dist')
            plt.hist(avg_dist.reshape(-1), bins=100, label='Avg Dist')
            plt.legend(loc='upper right')
            plt.savefig('dist-hist.png')
            plt.close()

            fig, axs = plt.subplots(5, figsize=(8, 20))
            a = dist_sq.detach().cpu().numpy()
            for k in range(5):
                axs[k].hist(a[k,:].reshape(-1), bins=100, label='dist_sq[{}, :]'.format(k))
                axs[k].legend(loc='upper right')
            plt.savefig('dist-hist-0.png')
            plt.close()


        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        # dist_sq = diff*diff
        # dist_sq = torch.sum(dist_sq,2)

        # Obtain the exponents
        expo = -0.5 * dist_sq
        if torch.any(torch.isnan(expo)):
            raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")

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
        numer_safe = det_scale_rep * torch.exp(expo_safe)
        denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        resp = numer_safe / denom_safe  # use broadcast

        if torch.any(torch.isnan(numer_safe)):
            raise RuntimeError("Nan at \"numer_safe = det_scale_rep * torch.exp(expo_safe)\"")
        if torch.any(torch.isnan(denom_safe)):
            raise RuntimeError("Nan at \"denom_safe = torch.sum(numer_safe, 1, keepdim=True)\"")
        if torch.any(torch.isnan(resp)):
            raise RuntimeError("Nan at \"resp = numer_safe / denom_safe\"")

        # comment out
        # output = torch.log(resp)
        output = resp

        return output



class gmm_layer(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int):
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
        expo = -0.5 * dist_sq
        if torch.any(torch.isnan(expo)):
            raise RuntimeError("Nan at \"expo = -0.5 * dist_sq\"")

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
        numer_safe = det_scale_rep * torch.exp(expo_safe)
        denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        resp = numer_safe / denom_safe  # use broadcast

        if torch.any(torch.isnan(numer_safe)):
            raise RuntimeError("Nan at \"numer_safe = det_scale_rep * torch.exp(expo_safe)\"")
        if torch.any(torch.isnan(denom_safe)):
            raise RuntimeError("Nan at \"denom_safe = torch.sum(numer_safe, 1, keepdim=True)\"")
        if torch.any(torch.isnan(resp)):
            raise RuntimeError("Nan at \"resp = numer_safe / denom_safe\"")

        # TODO force the nan with high learning rate and debug whats causing the nan

        # comment out
        # output = torch.log(resp)
        output = resp

        return output


class kMeansResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # TODO work out how to cluster in the 512 dim second to last layer
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.model.fc = Identity()  # replace the fc layer with an identity to ensure it does nothing, preserving the 512 len embedding

        self.dim = 512  # dim of the last FC layer in the model, I just manually specified it here out of lazyness.
        self.kmeans = kMeans(dim=self.dim, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)  # get the feature embedding of 512 dim
        output = self.kmeans(x)
        return output


class GmmResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # TODO work out how to cluster in the 512 dim second to last layer
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        # self.model.fc = Identity()  # replace the fc layer with an identity to ensure it does nothing, preserving the 512 len embedding

        self.dim = num_classes  # dim of the last FC layer in the model, I just manually specified it here out of lazyness.
        self.gmm_layer = gmm_layer(dim=self.dim, num_classes=num_classes)


    def forward(self, x):
        x = self.model(x)  # get the feature embedding of X dim
        output = self.gmm_layer(x)
        return output
