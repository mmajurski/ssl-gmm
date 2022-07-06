import numpy as np
import torch
from torch.distributions import MultivariateNormal #Just for test data
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mmm import GMM

def density_plot(data, mu, cov, n=100):
    '''
    :param data: type: numpy.array(num_samples x features)
    :param mu: type: torch.Tensor (features)
    :param cov: type: torch.Tensor (features)
    :param n: number of samples to generate, type: int
    :return likelihoods: type: torch.Tensor(num_samples, features)
    '''
    def vis(xx, yy, z):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, z, rstride=3, cstride=3, linewidth=1,
                        antialiased=True,
                        cmap=cm.inferno)
        cset = ax.contourf(xx, yy, z, zdir='z', offset=-0.15, cmap=cm.inferno)
        ax.set_zlim(-0.15, 0.2)
        ax.view_init(27, -21)
        plt.show()

    # Extract x and y
    x, y = data[:, 0], data[:, 1]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    # Create meshgrid
    xx, yy = np.meshgrid(xx, yy)
    # get the design matrix
    samples = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    samples = torch.from_numpy(samples).double()
    # compute the densities under each mixture
    k = len(mu)
    likelihoods = torch.zeros((n**2, k)).double()
    for i, data in enumerate(samples):
        for j in range(k):
            likelihoods[i, j] = MultivariateNormal(mu[j], cov[j]).log_prob(data).exp_().double()
    # sum the densities to get mixture density
    likelihoods = torch.sum(likelihoods, dim=1).data.numpy().reshape([n, n])
    vis(xx, yy, likelihoods)

def main():
    '''
    Unit test on a synthetic dataset generated from 3 2D Gaussians
    '''
    def synthetic_data(mu, var, num_samples=500):
        """
        :param mu: type: torch.Tensor (features)
        :param var: type: torch.Tensor (features)
        :param num_samples: number of samples to be drawn, type: int
        :return: type: torch.Tensor (num_samples, features)
        """
        data = []
        for i in range(num_samples):
            data += [torch.normal(mu, var.sqrt())]
        return torch.stack(data, dim=0)

    # generate some clusters (uneven number of samples to test robustness)
    n1, n2, n3 = 300, 500, 1000
    cluster1 = synthetic_data(torch.Tensor([2.5, 2.5]), torch.Tensor([1.2, .8]),
                              num_samples=n1).double()
    cluster2 = synthetic_data(torch.Tensor([7.5, 7.5]), torch.Tensor([.75, .5]),
                              num_samples=n2).double()
    cluster3 = synthetic_data(torch.Tensor([8, 15]), torch.Tensor([.6, .8]),
                              num_samples=n3).double()

    # cluster1 = synthetic_data(torch.Tensor([5.0, 5.0]), torch.Tensor([1.2, .8]),
    #                           num_samples=n1).double()
    # cluster2 = synthetic_data(torch.Tensor([-3.0, -2.0]), torch.Tensor([.75, .5]),
    #                           num_samples=n2).double()
    # cluster3 = synthetic_data(torch.Tensor([-3.0, 5.0]), torch.Tensor([.6, .8]),
    #                           num_samples=n3).double()

    x = torch.cat([cluster1, cluster2, cluster3]).double()
    
    gmm = GMM(x, num_clusters=3, num_iterations=50, tolerance=1e-4)
    mu, cov, posteriors, likelihoods = gmm.convergence()

    # plt.scatter(x.numpy()[:n1, 0], x.numpy()[:n1, 1], color='red')
    # plt.scatter(x.numpy()[n1:n1+n2, 0], x.numpy()[n1:n1+n2, 1], color='blue')
    # plt.scatter(x.numpy()[n1+n2:, 0], x.numpy()[n1+n2:, 1], color='green')
    # plt.title('Generated Data from 3 2D Gaussians')
    # plt.show()

    # density_plot(x.numpy(), mu, cov)
    # to test a predict_probs function.
    if False:
        print(gmm.predict_probs(x[:10], mu, cov))
        print(gmm.predict_probs(x[:10], mu, cov, False))

main()