import torch


def l2_cluster_centroid(embedding, centers, logits):
    num_classes = logits.shape[-1]
    # argmax resp to assign to cluster
    # optimize CE over resp + L2 loss
    cluster_assignment = torch.argmax(logits, dim=-1)

    # version of cluster_dist which is based on the centroid distance from self.centers
    cluster_dist = torch.zeros_like(logits[0, :])
    for c in range(num_classes):
        if torch.any(c == cluster_assignment):
            x_centroid = torch.mean(embedding[c == cluster_assignment, :], dim=0)
            delta = centers[c, :] - x_centroid
            delta = torch.sqrt(torch.sum(torch.pow(delta, 2), dim=-1))
            cluster_dist[c] = delta

    return cluster_dist


def mean_covar(embedding, centers, logits):
    num_classes = logits.shape[-1]
    dim = embedding.shape[-1]
    batch = embedding.shape[0]

    # Upsample the x-data to [batch, num_classes, dim]
    embedding_rep = embedding.unsqueeze(1).repeat(1, num_classes, 1)

    # Obtain cluster assignment from dist_sq directly
    cluster_assignment = torch.argmax(logits, dim=-1)
    cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, logits.shape[1])

    # ----------------------------------------
    # Calculate the empirical cluster mean / covariance
    #   OUTPUT:  empirical_mean  [classes dim]
    #                  cluster centers for the current minibatch
    #   OUTPUT:  empirical_covar [classes dim dim]
    #                  gaussian covariance matrices for the current minibatch
    #   OUTPUT:  cluster_weight  [classes]
    #                  number of samples for each class
    # ----------------------------------------
    cluster_weight = torch.sum(cluster_assignment_onehot, dim=0)
    cluster_assignment_onehot_rep = cluster_assignment_onehot.unsqueeze(2).repeat(1, 1, dim)
    x_onehot_rep = embedding_rep * cluster_assignment_onehot_rep

    #
    # Calculate the empirical mean
    #
    empirical_total = torch.sum(x_onehot_rep, dim=0)
    empirical_count = cluster_weight.unsqueeze(1).repeat(1, dim)
    empirical_mean = empirical_total / (empirical_count + 1e-8)

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
    empirical_covar_count = empirical_count.unsqueeze(2).repeat(1, 1, dim)

    empirical_covar = empirical_covar_total / (empirical_covar_count + 1e-8)

    # ----------------------------------------
    # Calculate a loss distance of the empirical measures from ideal
    # ----------------------------------------

    # ------
    # calculate empirical_mean_loss
    #  weighted L2 loss of each empirical mean from the cluster centers
    # ------

    # calculate empirical weighted dist squares (for means)
    empirical_diff = empirical_mean - centers
    empirical_diff_sq = empirical_diff * empirical_diff
    empirical_dist_sq = torch.sum(empirical_diff_sq, dim=1)
    empirical_wei_dist_sq = cluster_weight * empirical_dist_sq

    # create identity covariance of size [class dim dim]
    # identity_covar = torch.eye(self.dim).unsqueeze(0).repeat(self.num_classes,1,1)
    zeros = empirical_covar - empirical_covar
    zeros = torch.sum(zeros, dim=0)
    identity_covar = zeros.fill_diagonal_(1.0)
    identity_covar = identity_covar.unsqueeze(0).repeat(num_classes, 1, 1)

    # separate diagonal and off diagonal elements for covar loss
    empirical_covar_diag = empirical_covar * identity_covar
    empirical_covar_off_diag = empirical_covar * (1.0 - identity_covar)

    # calculate diagonal distance squared
    empirical_covar_diag_dist_sq = empirical_covar_diag - identity_covar
    empirical_covar_diag_dist_sq = empirical_covar_diag_dist_sq * empirical_covar_diag_dist_sq
    empirical_covar_diag_dist_sq = torch.sum(empirical_covar_diag_dist_sq, dim=2)
    empirical_covar_diag_dist_sq = torch.sum(empirical_covar_diag_dist_sq, dim=1)

    # calculate diagonal weighted distance squared
    empirical_covar_diag_wei_dist_sq = cluster_weight * empirical_covar_diag_dist_sq / (batch * dim)

    # calculate off diagonal distance squared
    empirical_covar_off_diag_dist_sq = empirical_covar_off_diag * empirical_covar_off_diag
    empirical_covar_off_diag_dist_sq = torch.sum(empirical_covar_off_diag_dist_sq, dim=2)
    empirical_covar_off_diag_dist_sq = torch.sum(empirical_covar_off_diag_dist_sq, dim=1)

    # Calculate off-diagonal weighted distance squared
    empirical_covar_off_diag_wei_dist_sq = cluster_weight * empirical_covar_off_diag_dist_sq / (batch * dim * (dim - 1.0))

    # Add together to get covariance loss
    empirical_covar_dist_sq = empirical_covar_diag_wei_dist_sq + empirical_covar_off_diag_wei_dist_sq

    # ------------------
    # return mean and covariance weighted mse loss
    # ------------------
    empirical_mean_mse = torch.sum(empirical_wei_dist_sq) / (batch * dim)
    empirical_covar_mse = torch.sum(empirical_covar_dist_sq)

    return empirical_mean_mse + empirical_covar_mse