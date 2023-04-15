import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import lcl_models

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, activate_before_residual=False):
        super(BasicBlock, self).__init__()

        #self.relu = nn.ReLU(inplace=True)  # published wideresnet network
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = None
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if self.equalInOut:
            out = self.relu(self.bn1(x))
            out = self.relu(self.bn2(self.conv1(out)))
        else:
            if self.activate_before_residual:
                x = self.relu(self.bn1(x))
            out = self.relu(self.bn2(self.conv1(x)))

        out = self.conv2(out)

        if self.equalInOut:
            return torch.add(x, out)
        else:
            return torch.add(self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, activate_before_residual=False):
        super(NetworkBlock, self).__init__()

        layers = []
        for i in range(int(nb_layers)):
            cur_in_planes = in_planes if i == 0 else out_planes
            cur_stride = stride if i == 0 else 1
            block = BasicBlock(cur_in_planes, out_planes, cur_stride, activate_before_residual)
            layers.append(block)

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, last_layer:str='fc', depth=28, width=2, embedding_dim=10):
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'

        super(WideResNet, self).__init__()
        channels = [16, 16*width, 32*width, 64*width]
        n = (depth - 4) / 6
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.width = width
        self.channels = channels[3]
        self.last_layer = last_layer

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], 1, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], 2)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], 2)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        # self.relu = nn.ReLU(inplace=True)  # published wideresnet network
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if self.last_layer == 'fc':
            self.fc = nn.Linear(channels[3], num_classes)
        elif self.last_layer == 'gmm':
            self.fc = nn.Linear(channels[3], embedding_dim)
            self.gmm_layer = lcl_models.axis_aligned_gmm_cmm_layer(embedding_dim, num_classes, return_gmm=True, return_cmm=False, return_cluster_dist=False)
        elif self.last_layer == 'cauchy':
            self.fc = nn.Linear(channels[3], embedding_dim)
            self.cmm_layer = lcl_models.axis_aligned_gmm_cmm_layer(embedding_dim, num_classes, return_gmm=False, return_cmm=True, return_cluster_dist=False)
        elif self.last_layer == 'aa_gmm':
            self.fc = nn.Linear(channels[3], embedding_dim)
            self.gmm_layer = lcl_models.axis_aligned_gmm_cmm_layer(embedding_dim, num_classes, return_gmm=True, return_cmm=True, return_cluster_dist=True)
        elif self.last_layer == 'aa_gmm_d1':
            self.fc = nn.Linear(channels[3], embedding_dim)
            self.gmm_layer = lcl_models.axis_aligned_gmm_cmm_D1_layer(embedding_dim, num_classes, return_gmm=True, return_cmm=True, return_cluster_dist=True)
        else:
            raise RuntimeError("Invalid last layer type: {}".format(self.last_layer))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        out = self.fc(out)
        if self.last_layer == 'fc':
            pass
        elif self.last_layer == 'gmm':
            out = self.gmm_layer(out)
        elif self.last_layer == 'cauchy':
            out = self.cmm_layer(out)
        elif self.last_layer == 'aa_gmm':
            return self.gmm_layer(out)
        elif self.last_layer == 'aa_gmm_d1':
            return self.gmm_layer(out)
        return out

