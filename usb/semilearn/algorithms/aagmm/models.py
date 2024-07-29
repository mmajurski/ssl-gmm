import torch


class aagmm_kl_layer(torch.nn.Module):
    def __init__(self, nDim: int, nClass:int, embedding_constraint:int=0):
        super(aagmm_kl_layer, self).__init__()
        self.n_dim = nDim
        self.n_class = nClass
        self.mu = torch.nn.Parameter(torch.randn((self.n_class, self.n_dim)))
        self.sigma = torch.nn.Parameter(torch.ones((self.n_class, self.n_dim)))
        self.embedding_constraint = embedding_constraint

        # Note 0.00010211761 = (2*pi)^(-dim/2) where dim=10
        self.numer_const = (2*torch.pi)**(-self.n_dim/2)

    def forward(self, x):
        delta = 1e-8
        # extract batch size and dimensions
        B = x.shape[0]  # B is batch size
        D = x.shape[1]  # D is encoded dimension
        C = self.n_class  # C is num classes
        if (D != self.n_dim):
            raise Exception('number of dimension D' + str(D) + 'is not expected' + str(self.n_dim))

        # extract parameters for AAGMM layer
        mu = self.mu  # cluster centers [C D]
        sigma = torch.abs(self.sigma)  # cluster stdev   [C D]
        inv_sigma = torch.reciprocal(sigma + delta)  # [C D]

        # calculate the determinent
        det = torch.prod(inv_sigma, dim=1)  # [C]

        # upscale all tensors to the same dimensions
        x = torch.reshape(x, (B, 1, D)).repeat((1, C, 1))  # shape [B C D]
        mu_rep = torch.reshape(mu, (1, C, D)).repeat((B, 1, 1))  # shape [B C D]
        inv_sigma = torch.reshape(inv_sigma, (1, C, D)).repeat((B, 1, 1))  # shape [B C D]
        det = torch.reshape(det, (1, C)).repeat((B, 1))  # shape [B C]

        # subtract mu and multiply by inv_sigma
        diff = (x - mu_rep) * inv_sigma  # [B C D]

        # obtain Mahalanobis distance
        dist_sq = torch.sum(diff * diff, dim=2)  # [B C]

        # Obtain the exponents
        expo = -0.5 * dist_sq  # [B C]

        # # Calculate the true numerators and denominators
        # #  (we don't use this directly for responsibility calculation
        # #   we actually use the "safe" versions that are shifted
        # #   for stability)
        # numer = self.numer_const * det * torch.exp(expo)  # [B C]
        # denom = torch.sum(numer, 1)  # [B]
        # denom = torch.reshape(denom, (B, 1)).repeat((1, C))  # [B C]
        # # resp = numer / (denom + delta)
        #
        # log('numer', numer)
        # log('denom', denom)

        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off = torch.max(expo, dim=1)[0]
        expo_safe_off = torch.reshape(expo_safe_off, (B, 1)).repeat((1, C))
        expo_safe = expo - expo_safe_off

        # Calculate the responsibilities
        numer_safe = self.numer_const * det * torch.exp(expo_safe)  # [B C]
        denom_safe = torch.sum(numer_safe, 1)  # [B]
        denom_safe = torch.reshape(denom_safe, (B, 1)).repeat((1, C))  # [B C]
        resp = numer_safe / denom_safe

        return resp

    def kl_penalty(self, embedding, logits):
        kl_penalty = 0.0
        if self.embedding_constraint == 0:
            return kl_penalty

        mu = self.mu  # cluster centers [C D]
        sigma = torch.abs(self.sigma)  # cluster stdev   [C D]

        delta = 1e-8
        # extract batch size and dimensions
        B = embedding.shape[0]  # B is batch size
        D = embedding.shape[1]  # D is encoded dimension
        C = self.n_class  # C is num classes
        if (D != self.n_dim):
            raise Exception('number of dimension D' + str(D) + 'is not expected' + str(self.n_dim))

        # upscale all tensors to the same dimensions
        embedding = torch.reshape(embedding, (B, 1, D)).repeat((1, C, 1))  # shape [B C D]

        # What is the cluster assignment  (one_hot)
        cluster_assignment = torch.argmax(logits, dim=1)  # [B]
        cluster_assignment = torch.nn.functional.one_hot(cluster_assignment, C)  # [B C]

        # Calculate cluster weight (how many samples per cluster)
        cluster_weight = torch.sum(cluster_assignment, dim=0)  # [C]
        cluster_weight_CD = torch.reshape(cluster_weight, (C, 1)).repeat((1, D))  # [C D]

        if self.embedding_constraint == 1:
            # Calculate the empirical mean
            cluster_mask = torch.reshape(cluster_assignment, (B, C, 1)).repeat((1, 1, D))  # [B C D]
            mu_Bc = torch.sum(embedding * cluster_mask, dim=0)  # [C D]
            mu_Bc = mu_Bc * torch.reciprocal(cluster_weight_CD + delta)  # [C D]

            diff = self.mu - mu_Bc
            mean_penalty = torch.linalg.norm(diff, ord=2, dim=1)
            kl_penalty = torch.sum(mean_penalty)

        if self.embedding_constraint == 2:
            # Calculate corrected cluster weight (N-1) because variance requires at least two samples
            cluster_weight_corr = torch.clamp(cluster_weight - 1.0, min=0.0)  # [C]
            cluster_weight_corr_CD = torch.reshape(cluster_weight_corr, (C, 1)).repeat((1, D))  # [C D]

            # Corrected batch size
            batch_size_corr = torch.sum(cluster_weight_corr)

            # Calculate the empirical mean
            cluster_mask = torch.reshape(cluster_assignment, (B, C, 1)).repeat((1, 1, D))  # [B C D]
            mu_Bc = torch.sum(embedding * cluster_mask, dim=0)  # [C D]
            mu_Bc = mu_Bc * torch.reciprocal(cluster_weight_CD + delta)  # [C D]

            # Calculate the empirical standard deviation
            sigma_Bc = torch.reshape(mu_Bc, (1, C, D)).repeat((B, 1, 1))  # [B C D]
            sigma_Bc = (embedding - sigma_Bc) * cluster_mask  # [B C D]
            sigma_Bc = sigma_Bc * sigma_Bc  # [B C D]
            sigma_Bc = torch.sum(sigma_Bc, dim=0)  # [C D]
            sigma_Bc = sigma_Bc * torch.reciprocal(cluster_weight_corr_CD + delta)  # [C D]

            # Calculate the kl-divergence
            sigma_Bc = sigma_Bc + 1e-8  # [C D]

            # sigma of target distribution
            kl_div = 0.5 * (  # [C D]
                    torch.log(sigma.pow(2) / sigma_Bc.pow(2)) - 1 +
                    (1 / sigma.pow(2)) * (sigma_Bc.pow(2) + (mu_Bc - mu).pow(2)))
            kl_div = (kl_div * cluster_weight_corr_CD) / (batch_size_corr + delta)
            kl_penalty = kl_div  # [C D]

            # add it up to get a full kl-divergence penalty
            kl_penalty = torch.sum(kl_penalty)
        return kl_penalty


class aagmm_layer(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()

        self.dim = embedding_dim
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
        det_scale_unsafe = torch.exp(det_scale_factor)

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

        cluster_dist_mat = torch.sqrt(dist_sq)  # in sigma
        # TOOD filter for > 2.0 for the nearest assigned class
        # TODO if min per col? > 2.0, not use


        #   GMM
        # dist_sq = (x-mu) Sigma_inv (x-mu)T
        #   K-means
        # dist_sq = (x-mu) dot (x-mu)

        # Obtain the exponents
        expo_gmm = -0.5 * dist_sq

        # TODO for aagmm dist_sq is in units of std for the clusters, maybe use this for outlier thresholding
        # TODO use min of sqrt(dist_sq) as the outlier thresholding. I.e. whats it min distance to any cluster. If data point is x std away from all clusters, then its an outlier. So threshold this value with x=2.


        # Safe version
        det_scale_rep_safe = det_scale_safe.unsqueeze(0).repeat(batch, 1)
        det_scale_rep_unsafe = det_scale_unsafe.unsqueeze(0).repeat(batch, 1)


        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_gmm, _ = torch.max(expo_gmm, dim=-1, keepdim=True)
        expo_safe_gmm = expo_gmm - expo_safe_off_gmm  # use broadcast instead of the repeat

        numer_unsafe_gmm = det_scale_rep_unsafe * torch.exp(expo_gmm)
        denom_unsafe_gmm = torch.sum(numer_unsafe_gmm, 1, keepdim=True)


        # Calculate the responsibilities
        numer_safe_gmm = det_scale_rep_safe * torch.exp(expo_safe_gmm)
        denom_safe_gmm = torch.sum(numer_safe_gmm, 1, keepdim=True)
        resp_gmm = numer_safe_gmm / denom_safe_gmm  # use broadcast

        resp_gmm = torch.clip(resp_gmm, min=1e-8)
        resp_gmm = torch.log(resp_gmm)

        # # Which cluster to assign ?
        # assign_gmm = torch.argmax(resp_gmm,dim=1)
        # assign_gmm_onehot = torch.nn.functional.one_hot(assign_gmm, self.num_classes)
        #
        # # Calculate the cross-entroy embedding constraint H(p(Y,Z)||Q)
        # #class_contrib = math.log(1.0/self.num_classes)
        # ln_2pi = 1.83787706641
        # log_det_rep = torch.reshape(log_det, (1,self.num_classes)).repeat((batch,1))
        # cross_entropy_embed = -0.5*self.dim*ln_2pi + log_det_rep + expo_gmm
        # cross_entropy_embed *= assign_gmm_onehot
        # cross_entropy_embed = (-1.0 / batch) * torch.sum(assign_gmm_onehot * cross_entropy_embed)

        return resp_gmm


class kmeans_layer(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()

        self.dim = embedding_dim
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

        return resp_kmeans


class AagmmModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_classes: int, last_layer: str = 'aagmm', embedding_dim: int = None, embedding_constraint:int = 0):
        super(AagmmModelWrapper, self).__init__()
        self.num_classes = num_classes

        self.emb_linear = None
        if embedding_dim is None or embedding_dim == 0:
            self.embedding_dim = model.classifier.in_features
        else:
            self.embedding_dim = embedding_dim
            # create a layer to convert from model.channels to embedding_dim
            self.emb_linear = torch.nn.Linear(model.classifier.in_features, self.embedding_dim)

        self.model = model

        self.last_layer_name = last_layer
        if self.last_layer_name == 'linear':
            self.last_layer = torch.nn.Linear(self.embedding_dim, self.num_classes)
        elif self.last_layer_name == 'aagmm' or self.last_layer_name == 'aa_gmm':
            self.last_layer = aagmm_layer(self.embedding_dim, self.num_classes)
        elif self.last_layer_name == 'aagmm_kl':
            self.last_layer = aagmm_kl_layer(nDim=self.embedding_dim, nClass=self.num_classes, embedding_constraint=embedding_constraint)
        elif self.last_layer_name == 'kmeans':
            self.last_layer = kmeans_layer(self.embedding_dim, self.num_classes)
        else:
            raise RuntimeError("Invalid last layer type: {}".format(self.last_layer_name))

    def forward(self, x):
        embedding = self.model(x, only_feat=True)

        kl_loss = 0.0
        if self.emb_linear is not None:
            embedding = self.emb_linear(embedding)
        if self.last_layer_name == 'linear':
            logits = self.last_layer(embedding)
        else:
            logits = self.last_layer(embedding)

        outputs = {'logits': logits, 'feat': embedding}
        return outputs





