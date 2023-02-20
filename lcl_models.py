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

        # Upsample the clusters to [batch, 10, 10]
        centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

        # Subtract to get diff of [batch, 10, 10]
        diff = x_rep - centers_rep

        # Obtain the square distance to each cluster
        #  of size [batch, dim]
        dist_sq = diff * diff
        dist_sq = torch.sum(dist_sq, 2)

        # Obtain the exponents
        expo = -0.5 * dist_sq

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

        # TODO create a cauchy version of this resp

        # Calculate the responsibilities
        numer_safe = torch.exp(expo_safe)
        denom_safe = torch.sum(numer_safe, 1, keepdim=True)
        resp = numer_safe / denom_safe  # use broadcast

        output = torch.log(resp)

        return output


class kMeansResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # TODO work out how to cluster in the 512 dim second to last layer
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.model.fc = Identity()  # replace the fc layer with an identity to ensure it does nothing, preserving the 512 len embedding

        self.dim = 512  # dim of the last FC layer in the model, I just manually specified it here out of lazyness.
        # kmeans layer
        self.kmeans = kMeans(dim=self.dim, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)  # get the feature embedding of 512 dim
        output = self.kmeans(x)
        return output


class GmmResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        raise NotImplementedError()
        # super().__init__()
        #
        # # TODO work out how to cluster in the 512 dim second to last layer
        # self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        # self.model.fc = Identity()  # replace the fc layer with an identity to ensure it does nothing, preserving the 512 len embedding
        #
        # self.dim = 512  # dim of the last FC layer in the model, I just manually specified it here out of lazyness.
        # # kmeans layer
        # self.kmeans = kMeans(dim=self.dim, num_classes=num_classes)

    def forward(self, x):
        raise NotImplementedError()
        # x = self.model(x)  # get the feature embedding of 512 dim
        # output = self.kmeans(x)
        # return output

