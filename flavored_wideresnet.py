import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import lcl_models2

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
    def __init__(self, num_classes, depth=28, width=2):
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'

        super(WideResNet, self).__init__()
        channels = [16, 16*width, 32*width, 64*width]
        n = (depth - 4) / 6
        self.num_classes = num_classes
        self.depth = depth
        self.width = width
        self.channels = channels[3]

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
        self.fc = nn.Linear(channels[3], num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        out = self.fc(out)
        return out

            

class WideResNetMajurski(nn.Module):
    def __init__(self, num_classes, last_layer:str='aa_gmm_d1', depth=28, width=2, embedding_dim=None, output_folder=None):
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'

        super(WideResNetMajurski, self).__init__()
        channels = [16, 16*width, 32*width, 64*width]
        n = (depth - 4) / 6
        self.num_classes = num_classes

        self.emb_linear = None
        if embedding_dim is None or embedding_dim == 0:
            self.embedding_dim = channels[3]
        else:
            self.embedding_dim = embedding_dim
            # create a layer to conver from channels[3] to embedding_dim
            self.emb_linear = nn.Linear(channels[3], self.embedding_dim)
        self.depth = depth
        self.width = width
        self.channels = channels[3]
        self.last_layer_name = last_layer
        self.output_folder = output_folder

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

        self.count = 0

        if self.last_layer_name == 'fc':
            self.last_layer = nn.Linear(self.embedding_dim, self.num_classes)
        elif self.last_layer_name == 'aa_gmm':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=False)
        elif self.last_layer_name == 'aa_cmm':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_layer(self.embedding_dim, self.num_classes, return_gmm=False, return_cmm=True)
        elif self.last_layer_name == 'aa_gmmcmm':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=True)
        elif self.last_layer_name == 'aa_gmm_d1':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_D1_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=False)
        elif self.last_layer_name == 'aa_cmm_d1':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_D1_layer(self.embedding_dim, self.num_classes, return_gmm=False, return_cmm=True)
        elif self.last_layer_name == 'aa_gmmcmm_d1':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_D1_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=True)
        elif self.last_layer_name == 'kmeans':
            self.last_layer = lcl_models2.kmeans(self.embedding_dim, self.num_classes)
        else:
            raise RuntimeError("Invalid last layer type: {}".format(self.last_layer_name))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        embedding = out.view(-1, self.channels)

        if self.emb_linear is not None:
            embedding = self.emb_linear(embedding)

        if self.last_layer_name == 'fc':
            logits = self.last_layer(embedding)
            latent_cross_entropy = 0.0
        else:
            logits, latent_cross_entropy = self.last_layer(embedding)

        return embedding, logits, latent_cross_entropy


class WideResNetVarMajurski(nn.Module):
    def __init__(self, num_classes, last_layer:str='aa_gmm', depth=28, width=2, embedding_dim=None):
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'

        super(WideResNetVarMajurski, self).__init__()
        channels = [16, 16 * width, 32 * width, 64 * width, 128 * width]
        n = (depth - 4) / 6
        self.num_classes = num_classes

        self.emb_linear = None
        if embedding_dim is None or embedding_dim == 0:
            self.embedding_dim = channels[3]
        else:
            self.embedding_dim = embedding_dim
            # create a layer to conver from channels[4] to embedding_dim
            self.emb_linear = nn.Linear(channels[4], self.embedding_dim)
        self.depth = depth
        self.width = width
        self.channels = channels[4]
        self.last_layer_name = last_layer

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], 1, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], 2)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], 2)
        # 4rd block
        self.block4 = NetworkBlock(n, channels[3], channels[4], 2)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001)
        # self.relu = nn.ReLU(inplace=True)  # published wideresnet network
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.count = 0

        if self.last_layer_name == 'fc':
            self.last_layer = nn.Linear(self.embedding_dim, self.num_classes)
        elif self.last_layer_name == 'aa_gmm':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=False)
        elif self.last_layer_name == 'aa_cmm':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_layer(self.embedding_dim, self.num_classes, return_gmm=False, return_cmm=True)
        elif self.last_layer_name == 'aa_gmmcmm':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=True)
        elif self.last_layer_name == 'aa_gmm_d1':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_D1_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=False)
        elif self.last_layer_name == 'aa_cmm_d1':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_D1_layer(self.embedding_dim, self.num_classes, return_gmm=False, return_cmm=True)
        elif self.last_layer_name == 'aa_gmmcmm_d1':
            self.last_layer = lcl_models2.axis_aligned_gmm_cmm_D1_layer(self.embedding_dim, self.num_classes, return_gmm=True, return_cmm=True)
        elif self.last_layer_name == 'kmeans':
            self.last_layer = lcl_models2.kmeans(self.embedding_dim, self.num_classes)
        else:
            raise RuntimeError("Invalid last layer type: {}".format(self.last_layer_name))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        embedding = out.view(-1, self.channels)

        if self.emb_linear is not None:
            embedding = self.emb_linear(embedding)

        if self.last_layer_name == 'fc':
            logits = self.last_layer(embedding)
            latent_cross_entropy = 0.0
        else:
            logits, latent_cross_entropy = self.last_layer(embedding)

        return embedding, logits, latent_cross_entropy

