import torch





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
        # det_scale_unsafe = torch.exp(det_scale_factor)

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
        # det_scale_rep_unsafe = det_scale_unsafe.unsqueeze(0).repeat(batch, 1)


        # Obtain the "safe" (numerically stable) versions of the
        #  exponents.  These "safe" exponents produce fake numer and denom
        #  but guarantee that resp = fake_numer / fake_denom = numer / denom
        #  where fake_numer and fake_denom are numerically stable
        expo_safe_off_gmm, _ = torch.max(expo_gmm, dim=-1, keepdim=True)
        expo_safe_gmm = expo_gmm - expo_safe_off_gmm  # use broadcast instead of the repeat

        # numer_unsafe_gmm = det_scale_rep_unsafe * torch.exp(expo_gmm)
        # denom_unsafe_gmm = torch.sum(numer_unsafe_gmm, 1, keepdim=True)


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

        cluster_assignment = torch.argmax(resp_gmm, dim=1)
        cluster_dist = cluster_dist_mat[range(cluster_dist_mat.shape[0]), cluster_assignment]
        #cluster_dist2 = torch.min(cluster_dist_mat, dim=1).values

        return resp_gmm, cluster_dist  #denom_unsafe_gmm  #, cross_entropy_embed


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

        return resp_kmeans, denom_safe_kmeans  #, 0.0


class AagmmModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_classes: int, last_layer: str = 'aa_gmm', embedding_dim: int = None):
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
        self.outlier_threshold = torch.nn.Parameter(5.0 * torch.ones(size=(1,), requires_grad=True))

        self.last_layer_name = last_layer
        if self.last_layer_name == 'linear':
            self.last_layer = torch.nn.Linear(self.embedding_dim, self.num_classes)
        elif self.last_layer_name == 'aa_gmm':
            self.last_layer = aagmm_layer(self.embedding_dim, self.num_classes)
        elif self.last_layer_name == 'kmeans':
            self.last_layer = kmeans_layer(self.embedding_dim, self.num_classes)
        else:
            raise RuntimeError("Invalid last layer type: {}".format(self.last_layer_name))

    def forward(self, x):
        embedding = self.model(x, only_feat=True)

        if self.emb_linear is not None:
            embedding = self.emb_linear(embedding)
        if self.last_layer_name == 'linear':
            logits = self.last_layer(embedding)
            denom = torch.ones((logits.shape[0]))
        else:
            logits, denom = self.last_layer(embedding)
            denom = denom.squeeze()  # get rid of singleton second dimension, is originally [batch_size,1]

        outputs = {'logits': logits, 'feat': embedding, 'denom': denom}
        return outputs





