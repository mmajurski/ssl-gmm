import torch
from .gauss_moments import GaussMoments


class GaussianMoments(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, moment_order: int):
        super(GaussianMoments, self).__init__()
        self.dim = embedding_dim
        self.num_classes = num_classes
        self.moment_order = moment_order
        assert 1 <= self.moment_order <= 4

        moment_1 = GaussMoments(self.dim, 1)  # position
        self.moment1_weight = torch.tensor(moment_1.moment_weights, requires_grad=False)
        self.gauss_moments1 = torch.tensor(moment_1.joint_gauss_moments, requires_grad=False)

        if self.moment_order >= 2:
            moment_2 = GaussMoments(self.dim, 2)  # variance
            self.moment2_weight = torch.tensor(moment_2.moment_weights, requires_grad=False)
            self.gauss_moments2 = torch.tensor(moment_2.joint_gauss_moments, requires_grad=False)

        if self.moment_order >= 3:
            moment_3 = GaussMoments(self.dim, 3)  # skew
            self.moment3_weight = torch.tensor(moment_3.moment_weights, requires_grad=False)
            self.gauss_moments3 = torch.tensor(moment_3.joint_gauss_moments, requires_grad=False)

        if self.moment_order >= 4:
            moment_4 = GaussMoments(self.dim, 4)  # kutorsis
            self.moment4_weight = torch.tensor(moment_4.moment_weights, requires_grad=False)
            self.gauss_moments4 = torch.tensor(moment_4.joint_gauss_moments, requires_grad=False)


    def forward(self, embedding, centers, logits):
        if centers is None:
            return 0.0

        if centers.device != self.gauss_moments1.device:
            self.gauss_moments1 = self.gauss_moments1.to(centers.device)
            self.moment1_weight = self.moment1_weight.to(centers.device)
            if self.moment_order >= 2:
                self.gauss_moments2 = self.gauss_moments2.to(centers.device)
                self.moment2_weight = self.moment2_weight.to(centers.device)
            if self.moment_order >= 3:
                self.gauss_moments3 = self.gauss_moments3.to(centers.device)
                self.moment3_weight = self.moment3_weight.to(centers.device)
            if self.moment_order >= 4:
                self.gauss_moments4 = self.gauss_moments4.to(centers.device)
                self.moment4_weight = self.moment4_weight.to(centers.device)

        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss
        cluster_assignment = torch.argmax(logits, dim=-1)
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, logits.shape[1])


        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = embedding.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, 10, 10]
        batch = logits.shape[0]  # batch size
        centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

        #		print(centers)
        #		print("centers")
        #		input("enter")

        # Subtract to get diff of [batch, 10, 10]
        diff = x_rep - centers_rep

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

        moment1 = torch.sum(diff_onehot, dim=0)
        moment1_count = cluster_weight.unsqueeze(1).repeat(1, self.dim)
        moment1 = moment1 / (moment1_count + 0.0000001)

        if self.moment_order >= 2:
            moment2_a = diff_onehot.unsqueeze(2)
            moment2_b = diff_onehot.unsqueeze(3)
            moment2_a_rep = moment2_a.repeat((1, 1, self.dim, 1))
            moment2_b_rep = moment2_b.repeat((1, 1, 1, self.dim))
            moment2 = moment2_a_rep * moment2_b_rep
            moment2 = torch.sum(moment2, dim=0)
            moment2_count = moment1_count.unsqueeze(2).repeat((1, 1, self.dim))
            moment2 = moment2 / (moment2_count + 0.0000001)

        if self.moment_order >= 3:
            moment3_a = moment2_a.unsqueeze(2)
            moment3_b = moment2_b.unsqueeze(2)
            moment3_c = moment2_b.unsqueeze(4)
            moment3_a_rep = moment3_a.repeat((1, 1, self.dim, self.dim, 1))
            moment3_b_rep = moment3_b.repeat((1, 1, self.dim, 1, self.dim))
            moment3_c_rep = moment3_c.repeat((1, 1, 1, self.dim, self.dim))
            moment3 = moment3_a_rep * moment3_b_rep * moment3_c_rep
            moment3 = torch.sum(moment3, dim=0)

        if self.moment_order >= 4:
            moment4_a = moment3_a.unsqueeze(2)
            moment4_b = moment3_b.unsqueeze(2)
            moment4_c = moment3_c.unsqueeze(2)
            moment4_d = moment3_c.unsqueeze(5)
            moment4_a_rep = moment4_a.repeat((1, 1, self.dim, self.dim, self.dim, 1))
            moment4_b_rep = moment4_b.repeat((1, 1, self.dim, self.dim, 1, self.dim))
            moment4_c_rep = moment4_c.repeat((1, 1, self.dim, 1, self.dim, self.dim))
            moment4_d_rep = moment4_d.repeat((1, 1, 1, self.dim, self.dim, self.dim))
            moment4 = moment4_a_rep * moment4_b_rep * moment4_c_rep * moment4_d_rep
            moment4 = torch.sum(moment4, dim=0)

        # ---------------------------------------
        # calculate the moment loss
        # ---------------------------------------

        # get the moment targets
        moment1_target = self.gauss_moments1
        if self.moment_order >= 2:
            moment2_target = self.gauss_moments2
        if self.moment_order >= 3:
            moment3_target = self.gauss_moments3
        if self.moment_order >= 4:
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
        moment1_target = moment1_target.unsqueeze(0).repeat(self.num_classes, 1)

        if self.moment_order >= 2:
            moment2 = torch.sign(torch.sign(moment2) + 0.1) * (torch.pow(torch.abs(moment2) + 0.25, 0.5) - 0.5)
            moment2_target = torch.sign(torch.sign(moment2_target) + 0.1) * (torch.pow(torch.abs(moment2_target) + 0.25, 0.5) - 0.5)
            moment2_target = moment2_target.unsqueeze(0).repeat(self.num_classes, 1, 1)

            # moment2_target2 = torch.diag_embed(torch.abs(D) + 1e-8)


        if self.moment_order >= 3:
            moment3 = torch.sign(torch.sign(moment3) + 0.1) * (torch.pow(torch.abs(moment3) + 0.19245008973, 0.3333333333) - 0.57735026919)
            moment3_target = torch.sign(torch.sign(moment3_target) + 0.1) * (torch.pow(torch.abs(moment3_target) + 0.19245008973, 0.3333333333) - 0.57735026919)
            moment3_target = moment3_target.unsqueeze(0).repeat(self.num_classes, 1, 1, 1)

        if self.moment_order >= 4:
            moment4 = torch.sign(torch.sign(moment4) + 0.1) * (torch.pow(torch.abs(moment4) + 0.15749013123, 0.25) - 0.62996052494)
            moment4_target = torch.sign(torch.sign(moment4_target) + 0.1) * (torch.pow(torch.abs(moment4_target) + 0.15749013123, 0.25) - 0.62996052494)
            moment4_target = moment4_target.unsqueeze(0).repeat(self.num_classes, 1, 1, 1, 1)

        # repeat the moment penalty weights perclass
        cluster_weight_norm = cluster_weight / torch.sum(cluster_weight)

        cluster_weight_rep = cluster_weight_norm.unsqueeze(1).repeat((1, self.dim))
        moment1_weight = cluster_weight_rep * self.moment1_weight.unsqueeze(0).repeat((self.num_classes, 1))
        moment_penalty1 = torch.sum(moment1_weight * torch.pow((moment1 - moment1_target), 2))

        if self.moment_order >= 2:
            cluster_weight_rep = cluster_weight_rep.unsqueeze(2).repeat((1, 1, self.dim))
            moment2_weight = cluster_weight_rep * self.moment2_weight.unsqueeze(0).repeat((self.num_classes, 1, 1))

            # set the diagonal elements weight to 0
            mask = torch.eye(*moment2.shape[-2:])
            mask = mask.repeat((self.num_classes, 1, 1))
            moment2_weight[mask > 0] = 0.0

            moment_penalty2 = torch.sum(moment2_weight * torch.pow((moment2 - moment2_target), 2))

        if self.moment_order >= 3:
            cluster_weight_rep = cluster_weight_rep.unsqueeze(3).repeat((1, 1, 1, self.dim))
            moment3_weight = cluster_weight_rep * self.moment3_weight.unsqueeze(0).repeat((self.num_classes, 1, 1, 1))
            moment_penalty3 = torch.sum(moment3_weight * torch.pow((moment3 - moment3_target), 2))

        if self.moment_order >= 4:
            cluster_weight_rep = cluster_weight_rep.unsqueeze(4).repeat((1, 1, 1, 1, self.dim))
            moment4_weight = cluster_weight_rep * self.moment4_weight.unsqueeze(0).repeat((self.num_classes, 1, 1, 1, 1))

            # set the diagonal elements weight to 0
            mask = torch.eye(*moment4.shape[-2:])
            mask = mask.repeat((self.num_classes, self.dim, self.dim, 1, 1))
            moment4_weight[mask > 0] = 0.0

            moment_penalty4 = torch.sum(moment4_weight * torch.pow((moment4 - moment4_target), 2))

        mom_penalty = 1.0 * moment_penalty1
        if self.moment_order >= 2:
            mom_penalty += 0.5 * moment_penalty2
        if self.moment_order >= 3:
            mom_penalty += 0.25 * moment_penalty3
        if self.moment_order >= 4:
            mom_penalty += 0.125 * moment_penalty4

        return mom_penalty


class GaussianMoments4(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super(GaussianMoments4, self).__init__()
        self.dim = embedding_dim
        self.num_classes = num_classes

        moment_1 = GaussMoments(self.dim, 1)  # position
        moment_2 = GaussMoments(self.dim, 2)  # variance
        moment_3 = GaussMoments(self.dim, 3)  # skew
        moment_4 = GaussMoments(self.dim, 4)  # kutorsis

        # moment weights (for moment loss function)
        self.moment1_weight = torch.tensor(moment_1.moment_weights, requires_grad=False)
        self.moment2_weight = torch.tensor(moment_2.moment_weights, requires_grad=False)
        self.moment3_weight = torch.tensor(moment_3.moment_weights, requires_grad=False)
        self.moment4_weight = torch.tensor(moment_4.moment_weights, requires_grad=False)

        # gaussian moments
        self.gauss_moments1 = torch.tensor(moment_1.joint_gauss_moments, requires_grad=False)
        self.gauss_moments2 = torch.tensor(moment_2.joint_gauss_moments, requires_grad=False)
        self.gauss_moments3 = torch.tensor(moment_3.joint_gauss_moments, requires_grad=False)
        self.gauss_moments4 = torch.tensor(moment_4.joint_gauss_moments, requires_grad=False)

    def forward(self, embedding, centers, logits):
        if centers is None:
            return 0.0

        if centers.device != self.gauss_moments1.device:
            self.gauss_moments1 = self.gauss_moments1.to(centers.device)
            self.gauss_moments2 = self.gauss_moments2.to(centers.device)
            self.gauss_moments3 = self.gauss_moments3.to(centers.device)
            self.gauss_moments4 = self.gauss_moments4.to(centers.device)

            self.moment1_weight = self.moment1_weight.to(centers.device)
            self.moment2_weight = self.moment2_weight.to(centers.device)
            self.moment3_weight = self.moment3_weight.to(centers.device)
            self.moment4_weight = self.moment4_weight.to(centers.device)

        # if centers.device != 'cpu':
        #     embedding = embedding.cpu()
        #     centers = centers.cpu()
        #     logits = logits.cpu()

        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss
        cluster_assignment = torch.argmax(logits, dim=-1)
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, logits.shape[1])


        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = embedding.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, 10, 10]
        batch = logits.shape[0]  # batch size
        centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

        #		print(centers)
        #		print("centers")
        #		input("enter")

        # Subtract to get diff of [batch, 10, 10]
        diff = x_rep - centers_rep

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

        moment1 = torch.sum(diff_onehot, dim=0)
        moment1_count = cluster_weight.unsqueeze(1).repeat(1, self.dim)
        moment1 = moment1 / (moment1_count + 0.0000001)

        moment2_a = diff_onehot.unsqueeze(2)
        moment2_b = diff_onehot.unsqueeze(3)
        moment2_a_rep = moment2_a.repeat((1, 1, self.dim, 1))
        moment2_b_rep = moment2_b.repeat((1, 1, 1, self.dim))
        moment2 = moment2_a_rep * moment2_b_rep
        moment2 = torch.sum(moment2, dim=0)
        moment2_count = moment1_count.unsqueeze(2).repeat((1, 1, self.dim))
        moment2 = moment2 / (moment2_count + 0.0000001)

        moment3_a = moment2_a.unsqueeze(2)
        moment3_b = moment2_b.unsqueeze(2)
        moment3_c = moment2_b.unsqueeze(4)
        moment3_a_rep = moment3_a.repeat((1, 1, self.dim, self.dim, 1))
        moment3_b_rep = moment3_b.repeat((1, 1, self.dim, 1, self.dim))
        moment3_c_rep = moment3_c.repeat((1, 1, 1, self.dim, self.dim))
        moment3 = moment3_a_rep * moment3_b_rep * moment3_c_rep
        moment3 = torch.sum(moment3, dim=0)

        moment4_a = moment3_a.unsqueeze(2)
        moment4_b = moment3_b.unsqueeze(2)
        moment4_c = moment3_c.unsqueeze(2)
        moment4_d = moment3_c.unsqueeze(5)
        moment4_a_rep = moment4_a.repeat((1, 1, self.dim, self.dim, self.dim, 1))
        moment4_b_rep = moment4_b.repeat((1, 1, self.dim, self.dim, 1, self.dim))
        moment4_c_rep = moment4_c.repeat((1, 1, self.dim, 1, self.dim, self.dim))
        moment4_d_rep = moment4_d.repeat((1, 1, 1, self.dim, self.dim, self.dim))
        moment4 = moment4_a_rep * moment4_b_rep * moment4_c_rep * moment4_d_rep
        moment4 = torch.sum(moment4, dim=0)

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

        return mom_penalty

class GaussianMoments3(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super(GaussianMoments3, self).__init__()
        self.dim = embedding_dim
        self.num_classes = num_classes

        moment_1 = GaussMoments(self.dim, 1)  # position
        moment_2 = GaussMoments(self.dim, 2)  # variance
        moment_3 = GaussMoments(self.dim, 3)  # skew

        # moment weights (for moment loss function)
        self.moment1_weight = torch.tensor(moment_1.moment_weights, requires_grad=False)
        self.moment2_weight = torch.tensor(moment_2.moment_weights, requires_grad=False)
        self.moment3_weight = torch.tensor(moment_3.moment_weights, requires_grad=False)

        # gaussian moments
        self.gauss_moments1 = torch.tensor(moment_1.joint_gauss_moments, requires_grad=False)
        self.gauss_moments2 = torch.tensor(moment_2.joint_gauss_moments, requires_grad=False)
        self.gauss_moments3 = torch.tensor(moment_3.joint_gauss_moments, requires_grad=False)

    def forward(self, embedding, centers, logits):
        if centers is None:
            return 0.0

        if centers.device != self.gauss_moments1.device:
            self.gauss_moments1 = self.gauss_moments1.to(centers.device)
            self.gauss_moments2 = self.gauss_moments2.to(centers.device)
            self.gauss_moments3 = self.gauss_moments3.to(centers.device)

            self.moment1_weight = self.moment1_weight.to(centers.device)
            self.moment2_weight = self.moment2_weight.to(centers.device)
            self.moment3_weight = self.moment3_weight.to(centers.device)

        # if centers.device != 'cpu':
        #     embedding = embedding.cpu()
        #     centers = centers.cpu()
        #     logits = logits.cpu()

        # argmax resp to assign to cluster
        # optimize CE over resp + L2 loss
        cluster_assignment = torch.argmax(logits, dim=-1)
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, logits.shape[1])


        # Upsample the x-data to [batch, num_classes, dim]
        x_rep = embedding.unsqueeze(1).repeat(1, self.num_classes, 1)

        # Upsample the clusters to [batch, 10, 10]
        batch = logits.shape[0]  # batch size
        centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

        #		print(centers)
        #		print("centers")
        #		input("enter")

        # Subtract to get diff of [batch, 10, 10]
        diff = x_rep - centers_rep

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

        moment1 = torch.sum(diff_onehot, dim=0)
        moment1_count = cluster_weight.unsqueeze(1).repeat(1, self.dim)
        moment1 = moment1 / (moment1_count + 0.0000001)

        moment2_a = diff_onehot.unsqueeze(2)
        moment2_b = diff_onehot.unsqueeze(3)
        moment2_a_rep = moment2_a.repeat((1, 1, self.dim, 1))
        moment2_b_rep = moment2_b.repeat((1, 1, 1, self.dim))
        moment2 = moment2_a_rep * moment2_b_rep
        moment2 = torch.sum(moment2, dim=0)
        moment2_count = moment1_count.unsqueeze(2).repeat((1, 1, self.dim))
        moment2 = moment2 / (moment2_count + 0.0000001)

        moment3_a = moment2_a.unsqueeze(2)
        moment3_b = moment2_b.unsqueeze(2)
        moment3_c = moment2_b.unsqueeze(4)
        moment3_a_rep = moment3_a.repeat((1, 1, self.dim, self.dim, 1))
        moment3_b_rep = moment3_b.repeat((1, 1, self.dim, 1, self.dim))
        moment3_c_rep = moment3_c.repeat((1, 1, 1, self.dim, self.dim))
        moment3 = moment3_a_rep * moment3_b_rep * moment3_c_rep
        moment3 = torch.sum(moment3, dim=0)

        # ---------------------------------------
        # calculate the moment loss
        # ---------------------------------------

        # get the moment targets
        moment1_target = self.gauss_moments1
        moment2_target = self.gauss_moments2
        moment3_target = self.gauss_moments3

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

        moment2_target = torch.sign(torch.sign(moment2_target) + 0.1) * (torch.pow(torch.abs(moment2_target) + 0.25, 0.5) - 0.5)
        moment3_target = torch.sign(torch.sign(moment3_target) + 0.1) * (torch.pow(torch.abs(moment3_target) + 0.19245008973, 0.3333333333) - 0.57735026919)

        # repeat the moment targets per class
        moment1_target = moment1_target.unsqueeze(0).repeat(self.num_classes, 1)
        moment2_target = moment2_target.unsqueeze(0).repeat(self.num_classes, 1, 1)
        moment3_target = moment3_target.unsqueeze(0).repeat(self.num_classes, 1, 1, 1)

        # repeat the moment penalty weights perclass
        cluster_weight_norm = cluster_weight / torch.sum(cluster_weight)

        cluster_weight_rep = cluster_weight_norm.unsqueeze(1).repeat((1, self.dim))
        moment1_weight = cluster_weight_rep * self.moment1_weight.unsqueeze(0).repeat((self.num_classes, 1))

        cluster_weight_rep = cluster_weight_rep.unsqueeze(2).repeat((1, 1, self.dim))
        moment2_weight = cluster_weight_rep * self.moment2_weight.unsqueeze(0).repeat((self.num_classes, 1, 1))

        cluster_weight_rep = cluster_weight_rep.unsqueeze(3).repeat((1, 1, 1, self.dim))
        moment3_weight = cluster_weight_rep * self.moment3_weight.unsqueeze(0).repeat((self.num_classes, 1, 1, 1))

        # calculate the penalty loss function
        moment_penalty1 = torch.sum(moment1_weight * torch.pow((moment1 - moment1_target), 2))
        moment_penalty2 = torch.sum(moment2_weight * torch.pow((moment2 - moment2_target), 2))
        moment_penalty3 = torch.sum(moment3_weight * torch.pow((moment3 - moment3_target), 2))

        # MoM loss
        mom_penalty = 1.0 * moment_penalty1 + \
                      0.5 * moment_penalty2 + \
                      0.25 * moment_penalty3

        return mom_penalty
            

class MeanCovar(torch.nn.Module):
    def __init__(self):
        super(MeanCovar, self).__init__()

    def forward(self, embedding, centers, logits):
        if centers is None:
            return 0.0

        num_classes = logits.shape[-1]
        dim = embedding.shape[-1]
        batch = embedding.shape[0]

        # Upsample the x-data to [batch, num_classes, dim]
        embedding_rep = embedding.unsqueeze(1).repeat(1, num_classes, 1)
        # x_rep = embedding_rep

        # Obtain cluster assignment from dist_sq directly
        cluster_assignment = torch.argmax(logits, dim=-1)
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, logits.shape[1])

        # Upsample the x-data to [batch, num_classes, dim]
        centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, dim, dim]
        x_mu_rep = embedding_rep - centers_rep
        # diff = x_mu_rep

        # ----------------------------------------
        # Calculate the mean / stdev of the whole sample
        # ----------------------------------------

        
        #embedding_mean = torch.mean(embedding, axis=0)
        
        #embedding_stdev = torch.reshape(embedding_mean, (1, dim)).repeat(batch,1)
        #embedding_stdev = embedding - embedding_stdev
        embedding_stdev = embedding
        embedding_stdev = embedding_stdev*embedding_stdev
        embedding_stdev = torch.mean(embedding_stdev, axis=0)
        
        embedding_stdev_target = 10.0
        embedding_stdev_penalty = embedding_stdev_target - embedding_stdev
        embedding_stdev_penalty = torch.mean(embedding_stdev_penalty * embedding_stdev_penalty)

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

        diff_onehot = x_mu_rep * cluster_assignment_onehot_rep

        #
        # Calculate the empirical mean
        #
        empirical_total = torch.sum(diff_onehot, dim=0)
        empirical_count = cluster_weight.unsqueeze(1).repeat(1, dim)
        moment1 = empirical_total / (empirical_count + 1e-8)

        #
        # Calculate the empirical covariance
        #
        moment2_a = diff_onehot.unsqueeze(2)
        moment2_b = diff_onehot.unsqueeze(3)
        moment2_a_rep = moment2_a.repeat((1, 1, dim, 1))
        moment2_b_rep = moment2_b.repeat((1, 1, 1, dim))
        moment2 = moment2_a_rep * moment2_b_rep
        moment2 = torch.sum(moment2, dim=0)
        moment2_count = empirical_count.unsqueeze(2).repeat((1, 1, dim))
        moment2 = moment2 / (moment2_count + 1e-8)

        # TODO? moment2 = torch.sign(torch.sign(moment2) + 0.1) * (torch.pow(torch.abs(moment2) + 0.25, 0.5) - 0.5)

        # repeat the moment targets per class
        moment1_target = torch.zeros_like(moment1, requires_grad=False)
        moment1_weight = torch.ones_like(moment1, requires_grad=False) * (1.0 / dim)
        moment2_target = torch.eye(dim, dtype=moment2.dtype, requires_grad=False, device=moment2.device)
        # TODO? moment2_target = torch.sign(torch.sign(moment2_target) + 0.1) * (torch.pow(torch.abs(moment2_target) + 0.25, 0.5) - 0.5)

        moment2_target = moment2_target.repeat(num_classes, 1, 1)

        diag_terms     = moment2_target
        off_diag_terms = 1.0 - diag_terms

        a = 1.0 / float(2 * dim)
        b = 1.0 / float(2 * dim * (dim - 1))
        moment2_weight = (a - b) * torch.eye(dim, dtype=moment2.dtype, requires_grad=False, device=moment2.device)
        moment2_weight = moment2_weight + b

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
#        moment2        = torch.sign(torch.sign(moment2) + 0.1) * (torch.pow(torch.abs(moment2) + 0.25, 0.5) - 0.5)
#        moment2_target = torch.sign(torch.sign(moment2_target) + 0.1) * (torch.pow(torch.abs(moment2_target) + 0.25, 0.5) - 0.5)

        # repeat the moment penalty weights perclass
        cluster_weight_norm = cluster_weight / torch.sum(cluster_weight)

        cluster_weight_rep = cluster_weight_norm.unsqueeze(1).repeat((1, dim))
        moment1_weight = cluster_weight_rep * moment1_weight

        cluster_weight_rep = cluster_weight_rep.unsqueeze(2).repeat((1, 1, dim))
        moment2_weight = cluster_weight_rep * moment2_weight.unsqueeze(0).repeat((num_classes, 1, 1))

        # calculate the penalty loss function
        moment_penalty1 = torch.sum(moment1_weight * torch.pow((moment1 - moment1_target), 2))
#        moment_penalty2 = torch.sum(moment2_weight * torch.pow((moment2 - moment2_target), 2))

        # use cosine similarity
        #moment_penalty2_ab = torch.sum(moment2_weight * moment2 * moment2_target)
        #moment_penalty2_a  = torch.sqrt(torch.sum(moment2_weight * moment2 * moment2) + 0.000001)
        #moment_penalty2_b  = torch.sqrt(torch.sum(moment2_weight * moment2_target * moment2_target) + 0.000001)
        #moment_penalty2    = moment_penalty2_ab / (moment_penalty2_a * moment_penalty2_b + 0.000001)

        # use cosine similarity
        moment_similarity2_ab = moment2_weight * moment2 * moment2_target
        moment_similarity2_a  = moment2_weight * moment2 * moment2
        moment_similarity2_b  = moment2_weight * moment2_target * moment2_target
        moment_similarity2_ab =            torch.sum(torch.sum(moment_similarity2_ab, axis=2), axis=1)
        moment_similarity2_a  = torch.sqrt(torch.sum(torch.sum(moment_similarity2_a,  axis=2), axis=1) + 0.000001)
        moment_similarity2_b  = torch.sqrt(torch.sum(torch.sum(moment_similarity2_b,  axis=2), axis=1) + 0.000001)


        moment_similarity2    = moment_similarity2_ab / (moment_similarity2_a * moment_similarity2_b + 0.000001)


        moment_penalty2 = 1 - moment_similarity2


        moment_penalty2 = torch.mean(moment_penalty2)


        # separate into diagonal and off-diagonal terms
        moment_penalty2_diag     = torch.sum(    diag_terms * (moment2_weight * torch.pow((moment2 - moment2_target), 2)))
        moment_penalty2_off_diag = torch.sum(off_diag_terms * (moment2_weight * torch.pow((moment2 - moment2_target), 2)))

        # MoM loss
        #mom_penalty = 1.0 * moment_penalty1 + \
        #              0.5 * moment_penalty2
        # MoM loss
        mom_penalty = 1.0 * moment_penalty1 + \
                      0.05 * moment_penalty2
                      #0.5 * moment_penalty2

        return mom_penalty


class Mean(torch.nn.Module):
    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, embedding, centers, logits):
        if centers is None:
            return 0.0

        num_classes = logits.shape[-1]
        cluster_assignment = torch.argmax(logits, dim=-1)

        # Upsample the x-data to [batch, num_classes, dim]
        embedding_rep = embedding.unsqueeze(1).repeat(1, num_classes, 1)

        # Obtain cluster assignment
        cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, logits.shape[1])

        # Subtract to get diff of [batch, num_classes, dim]
        x_mu_rep = embedding_rep - centers.unsqueeze(0)

        cluster_weight = torch.sum(cluster_assignment_onehot, dim=0)

        # use broadcast
        diff_onehot = x_mu_rep * cluster_assignment_onehot.unsqueeze(2)

        #
        # Calculate the empirical mean
        #
        empirical_total = torch.sum(diff_onehot, dim=0)
        moment1 = empirical_total / (cluster_weight.unsqueeze(1) + 1e-8)

        # calculate the penalty loss function
        moment_penalty = torch.linalg.norm(moment1, ord=2, dim=1)  # grad stable, whereas below isn't (thou its the same math)
        # sqrt causes significant gradient instability (i.e. straight to nan in 1 step)
        # moment_penalty = torch.sqrt(torch.sum(torch.pow(moment1, 2), dim=1))

        return moment_penalty


