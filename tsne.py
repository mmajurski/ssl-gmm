import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import random
from sklearn.manifold import TSNE
matplotlib.use('agg')


print('----------------------------')
print(' Load embedding')
print('----------------------------')
Z = np.load("./model/test_embedding.npy")
labels = np.load("./model/test_labels.npy")

print('Z.shape', Z.shape)
print('labels.shape', labels.shape)

print('----------------------------')
print(' Fit TSNE')
print('----------------------------')


tsne = TSNE(n_components=2, random_state=42)
Y = tsne.fit_transform(Z)

print('Y.shape', Y.shape)

print('----------------------------')
print(' Plot TSNE')
print('----------------------------')

plt.figure(0, figsize=(12,12))
plt.scatter(Y[:, 0], Y[:, 1], c=labels)
plt.savefig("./model/tsne.png")

print(' Success!')
