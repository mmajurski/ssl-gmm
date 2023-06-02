from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
import torchvision.models.resnet as resnet
import copy
import os
import numpy as np

import metadata
import lr_scheduler


class kmeans_distribution_layer(torch.nn.Module):
	def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True, return_cluster_dist: bool = True, return_mean_mse: bool = True, return_covar_mse: bool = True):
		super().__init__()

		self.dim = embeddign_dim
		self.num_classes = num_classes
		self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

		self.return_gmm = return_gmm
		self.return_cmm = return_cmm
		self.return_cluster_dist = return_cluster_dist
		self.return_mean_mse = return_mean_mse
		self.return_covar_mse = return_covar_mse

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
		dist_sq = diff*diff
		dist_sq = torch.sum(dist_sq,2)

		#   GMM
		# dist_sq = (x-mu) Sigma_inv (x-mu)T
		#   K-means
		# dist_sq = (x-mu) dot (x-mu)

		# Obtain the exponents
		expo_kmeans = -0.5 * dist_sq
		a = torch.add(dist_sq, 1)
		b = -((1 + self.dim) / 2)
		expo_cmm = torch.pow(a, b)

		# Obtain the "safe" (numerically stable) versions of the
		#  exponents.  These "safe" exponents produce fake numer and denom
		#  but guarantee that resp = fake_numer / fake_denom = numer / denom
		#  where fake_numer and fake_denom are numerically stable
		expo_safe_off_kmeans, _ = torch.max(expo_kmeans, dim=-1, keepdim=True)
		expo_safe_kmeans = expo_kmeans - expo_safe_off_kmeans  # use broadcast instead of the repeat
		expo_safe_off_cmm, _ = torch.max(expo_cmm, dim=-1, keepdim=True)
		expo_safe_cmm = expo_cmm - expo_safe_off_cmm  # use broadcast instead of the repeat
		
		# Calculate the responsibilities kmeans
		numer_safe_kmeans = torch.exp(expo_safe_kmeans)
		denom_safe_kmeans = torch.sum(numer_safe_kmeans, 1, keepdim=True)
		resp_kmeans = numer_safe_kmeans / denom_safe_kmeans  # use broadcast

		# Calculate the responsibilities cmm
		numer_safe_cmm = torch.exp(expo_safe_cmm)
		denom_safe_cmm = torch.sum(numer_safe_cmm, 1, keepdim=True)
		resp_cmm = numer_safe_cmm / denom_safe_cmm  # use broadcast

		# # Build the kmeans resp
		# dist_sq_kmeans = torch.sum(diff * diff, dim=2)
		# # Obtain the exponents
		# expo = -0.5 * dist_sq_kmeans
		#
		# # Obtain the "safe" (numerically stable) versions of the
		# #  exponents.  These "safe" exponents produce fake numer and denom
		# #  but guarantee that resp = fake_numer / fake_denom = numer / denom
		# #  where fake_numer and fake_denom are numerically stable
		# expo_safe_off = torch.mean(expo, dim=-1, keepdim=True)
		# expo_safe = expo - expo_safe_off
		#
		# # Calculate the responsibilities
		# numer_safe = torch.exp(expo_safe)
		# denom_safe = torch.sum(numer_safe, 1, keepdim=True)
		# resp_kmeans = numer_safe / denom_safe

		# argmax resp to assign to cluster
		# optimize CE over resp + L2 loss

		#
		# Vectorized version of cluster_dist
		#
		
		# Obtain cluster assignment from dist_sq directly
		cluster_assignment = torch.argmin(dist_sq, dim=-1)

		# # This computes the L2 loss over all the points, as opposed to the required L2 loss for the centroids
		# # Use one-hot encoding trick to extract the dist_sq
		cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, dist_sq.shape[1])
		# cluster_dist_sq_onehot	= cluster_assignment_onehot * dist_sq
		# cluster_dist_sq		   = torch.sum(cluster_dist_sq_onehot, dim=-1)
		# # Take square root of dist_sq to get L2 norm
		# cluster_dist = torch.sqrt(cluster_dist_sq)

		# version of cluster_dist which is based on the centroid distance from self.centers
		cluster_dist = torch.zeros_like(resp_kmeans[0, :])
		for c in range(self.num_classes):
			if torch.any(c == cluster_assignment):
				x_centroid = torch.mean(x[c == cluster_assignment, :], dim=0)
				delta = self.centers[c, :] - x_centroid
				delta = torch.sqrt(torch.sum(torch.pow(delta, 2), dim=-1))
				cluster_dist[c] = delta

		resp_kmeans = torch.clip(resp_kmeans, min=1e-8)
		resp_cmm = torch.clip(resp_cmm, min=1e-8)
		resp_kmeans = torch.log(resp_kmeans)
		resp_cmm = torch.log(resp_cmm)


		#----------------------------------------
		# Calculate the empirical cluster mean / covariance
		#   OUTPUT:  empirical_mean  [classes dim]
		#                  cluster centers for the current minibatch
		#   OUTPUT:  empirical_covar [classes dim dim]
		#                  gaussian covariance matrices for the current minibatch
		#   OUTPUT:  cluster_weight  [classes]
		#                  number of samples for each class
		#----------------------------------------
		cluster_weight = torch.sum(cluster_assignment_onehot, axis=0)
		cluster_assignment_onehot_rep = cluster_assignment_onehot.unsqueeze(2).repeat(1, 1, self.dim)
		x_onehot_rep = x_rep * cluster_assignment_onehot_rep

		#
		# Calculate the empirical mean		
		#
		empirical_total = torch.sum(x_onehot_rep, axis=0)
		empirical_count = cluster_weight.unsqueeze(1).repeat(1,self.dim)
		empirical_mean  = empirical_total / (empirical_count + 0.0000001)
		
		#
		# Calculate the empirical covariance
		#
		empirical_mean_rep = empirical_mean.unsqueeze(0).repeat(batch,1,1)
		empirical_mean_rep = empirical_mean_rep * cluster_assignment_onehot_rep
		x_mu_rep = x_onehot_rep - empirical_mean_rep
				
		# perform batch matrix multiplication
		x_mu_rep_B = torch.transpose(x_mu_rep,0,1)
		x_mu_rep_A = torch.transpose(x_mu_rep_B,1,2)
		
		empirical_covar_total = torch.bmm(x_mu_rep_A,x_mu_rep_B)		
		empirical_covar_count = empirical_count.unsqueeze(2).repeat(1,1,self.dim)
		
		empirical_covar = empirical_covar_total / (empirical_covar_count + 0.0000001)
		
		#----------------------------------------
		# Calculate a loss distance of the empirical measures from ideal
		#----------------------------------------
		
		#------
		# calculate empirical_mean_loss
		#  weighted L2 loss of each empirical mean from the cluster centers
		#------
		
		# calculate empirical weighted dist squares (for means)
		empirical_diff         = empirical_mean - self.centers
		empirical_diff_sq      = empirical_diff * empirical_diff
		empirical_dist_sq      = torch.sum(empirical_diff_sq,axis=1)
		empirical_wei_dist_sq  = cluster_weight * empirical_dist_sq
		
		# create identity covariance of size [class dim dim]
		#identity_covar = torch.eye(self.dim).unsqueeze(0).repeat(self.num_classes,1,1)
		zeros = empirical_covar - empirical_covar
		zeros = torch.sum(zeros, axis=0)
		identity_covar = zeros.fill_diagonal_(1.0)
		identity_covar = identity_covar.unsqueeze(0).repeat(self.num_classes,1,1)

		# separate diagonal and off diagonal elements for covar loss
		empirical_covar_diag     = empirical_covar * identity_covar
		empirical_covar_off_diag = empirical_covar * (1.0 - identity_covar)

		# calculate diagonal distance squared
		empirical_covar_diag_dist_sq = empirical_covar_diag - identity_covar
		empirical_covar_diag_dist_sq = empirical_covar_diag_dist_sq * empirical_covar_diag_dist_sq
		empirical_covar_diag_dist_sq = torch.sum(empirical_covar_diag_dist_sq, axis=2)
		empirical_covar_diag_dist_sq = torch.sum(empirical_covar_diag_dist_sq, axis=1)

		# calculate diagonal weighted distance squared
		empirical_covar_diag_wei_dist_sq = cluster_weight * empirical_covar_diag_dist_sq / (batch*self.dim)


		# calculate off diagonal distance squared
		empirical_covar_off_diag_dist_sq = empirical_covar_off_diag * empirical_covar_off_diag
		empirical_covar_off_diag_dist_sq = torch.sum(empirical_covar_off_diag_dist_sq, axis=2)
		empirical_covar_off_diag_dist_sq = torch.sum(empirical_covar_off_diag_dist_sq, axis=1)

		# Calculate off-diagonal weighted distance squared
		empirical_covar_off_diag_wei_dist_sq = cluster_weight * empirical_covar_off_diag_dist_sq / (batch*self.dim*(self.dim-1.0))
		
		# Add together to get covariance loss
		empirical_covar_dist_sq = empirical_covar_diag_wei_dist_sq + empirical_covar_off_diag_wei_dist_sq

	
		#------------------
		# return mean and covariance weighted mse loss
		#------------------
		empirical_mean_mse  = torch.sum(empirical_wei_dist_sq) / (batch*self.dim)
		empirical_covar_mse = torch.sum(empirical_covar_dist_sq)
	
	
		outputs = list()
		if self.return_gmm:
			outputs.append(resp_kmeans)
		if self.return_cmm:
			outputs.append(resp_cmm)
		if self.return_cluster_dist:
			outputs.append(cluster_dist)
		if self.return_mean_mse:
			outputs.append(empirical_mean_mse)
		if self.return_covar_mse:
			outputs.append(empirical_covar_mse)

		return outputs
		

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		#----------
		# The architecture
		#----------
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 8)
		
		#----------
		# The k-means layer
		#----------
		self.last_layer = kmeans_distribution_layer(8, 10, return_gmm=True, return_cmm=True, return_cluster_dist=True)

	def forward(self, x):

		#----------
		# The architecture
		#----------
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)	

		#---------
		# The k-means layer
		#---------
		resp_list = self.last_layer(x)

		# Use the k-means layer
		#	 not we take log, because this model
		#	 outputs log softmax
#		output = torch.log(resp_list[0])
		output = resp_list[1]

		return resp_list



def train(args, model, device, train_loader, optimizer, epoch, train_stats):
	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		target_size = target.size()
		batch_size = target_size[0]
		num_class  = 10

		optimizer.zero_grad()
		output = model(data)
		output_kmeans    = output[0][:,:num_class]
		output_cmm       = output[1][:,:num_class]
		output_L2        = output[2][:]
		output_mean_mse  = output[3]
		output_covar_mse = output[4]

		pred = torch.argmax(output_kmeans, dim=-1)
		accuracy = torch.sum(pred == target) / len(pred)

		# calculate y and yhat
		y=F.one_hot(target,num_class)
		yhat=torch.softmax(output_kmeans,dim=1)

		# implement log_loss by hand
		minus_one_over_N = (-1.0 / (batch_size*num_class))
				
		log_yhat=torch.log(torch.clamp(yhat,min=0.0001))
		log_one_minus_yhat=torch.log(torch.clamp(1.0-yhat,min=0.0001))
		presum=(y * log_yhat + (1.0-y)*log_one_minus_yhat) * minus_one_over_N
		loss_bce = torch.sum(  presum  )
		loss = loss_bce + output_mean_mse + output_covar_mse
		
		#print("loss:", loss, "bce", loss_bce, "output_empirical", output_empirical)
		#input("press enter")
		
#		if (epoch < 2):
#			loss = loss + torch.sum( output_L2 )

		if not np.isnan(loss.item()):
			train_stats.append_accumulate('train_loss', loss.item())
			train_stats.append_accumulate('train_accuracy', accuracy.item())
			train_stats.append_accumulate('train_bce_loss', loss_bce.item())
			train_stats.append_accumulate('train_mean_mse_loss', output_mean_mse.item())
			train_stats.append_accumulate('train_covar_mse_loss', output_covar_mse.item())
		
		loss.backward()
		optimizer.step()
		
		if batch_idx % args.log_interval == 0:
			#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			#	epoch, batch_idx * len(data), len(train_loader.dataset),
			#	100. * batch_idx / len(train_loader), loss.item()))
			print("Train Epoch: %d [%d/%d] (%.0f)    Loss: %.6f    Bce: %.6f    Mean_mse %.6f  Covar_mse %.6f" %
				(epoch, batch_idx*len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item(), loss_bce.item(), output_mean_mse.item(), output_covar_mse.item() ) )
			if args.dry_run:
				break

	train_stats.add(epoch, 'learning_rate', optimizer.param_groups[0]['lr'])
	train_stats.close_accumulate(epoch, 'train_loss', method='avg')
	train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')
	train_stats.close_accumulate(epoch, 'train_bce_loss', method='avg')
	train_stats.close_accumulate(epoch, 'train_mean_mse_loss', method='avg')
	train_stats.close_accumulate(epoch, 'train_covar_mse_loss', method='avg')



def test(model, device, test_loader, epoch, train_stats):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)

			target_size = target.size()
			batch_size = target_size[0]
			num_class  = 10

			output = model(data)
			output_kmeans = output[0][:,:num_class]
			output_cmm    = output[1][:,:num_class]
			output_L2     = output[2][:]

			test_loss += F.nll_loss(output_kmeans, target, reduction='sum').item()  # sum up batch loss
			pred = output_kmeans.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			pred = torch.argmax(output_kmeans, dim=-1)
			accuracy = torch.sum(pred == target) / len(pred)

			if not np.isnan(test_loss):
				# train_stats.append_accumulate('test_loss', test_loss)
				train_stats.append_accumulate('test_accuracy', accuracy.item())

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	# train_stats.close_accumulate(epoch, 'test_loss', method='avg')
	train_stats.close_accumulate(epoch, 'test_accuracy', method='avg')


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of epochs to train (default: 14)')
	parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
						help='learning rate (default: 1.0)')
	parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
						help='Learning rate step gamma (default: 0.7)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--no-mps', action='store_true', default=False,
						help='disables macOS GPU training')
	parser.add_argument('--dry-run', action='store_true', default=False,
						help='quickly check a single pass')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	parser.add_argument('--num-lr-reductions', default=2, type=int)
	parser.add_argument('--lr-reduction-factor', default=0.2, type=float)
	parser.add_argument('--patience', default=4, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')
	parser.add_argument('--loss-eps', default=1e-4, type=float, help='loss value eps for determining early stopping loss equivalence.')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	use_mps = not args.no_mps and torch.backends.mps.is_available()

	torch.manual_seed(args.seed)

	if use_cuda:
		device = torch.device("cuda")
	elif use_mps:
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.test_batch_size}
	if use_cuda:
		cuda_kwargs = {'num_workers': 0,
					   'pin_memory': True,
					   'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])
	dataset1 = datasets.MNIST('../data', train=True, download=True,
					   transform=transform)
	dataset2 = datasets.MNIST('../data', train=False,
					   transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
	test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

	#input("10")

	model = Net().to(device)
	#model = resnet.resnet18().to(device)
	
	#input("20")

	#layers = dict(model.named_children())
	layers = dict(model.named_modules())
	# print(layers)
	
#	sys.exit(0)
#	#input ("press enter")
	
	
	optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
	train_stats = metadata.TrainingStats()

	output_dirpath = './covar-model'
	if not os.path.exists(output_dirpath):
		os.makedirs(output_dirpath)

	# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
	# for epoch in range(1, args.epochs + 1):
	# 	train(args, model, device, train_loader, optimizer, epoch)
	# 	torch.cuda.empty_cache()
	# 	torch.cuda.synchronize()
	# 	test(model, device, test_loader)
	# 	scheduler.step()


	plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions)

	epoch = -1
	MAX_EPOCHS = args.epochs+1
	best_model = copy.deepcopy(model)
	best_epoch = 0

	while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
		epoch += 1
		print("Epoch: {}".format(epoch))

		train_stats.plot_all_metrics(output_dirpath=output_dirpath)
		train(args, model, device, train_loader, optimizer, epoch, train_stats)

		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		test(model, device, test_loader, epoch, train_stats)

		test_accuracy = train_stats.get_epoch('test_accuracy', epoch=epoch)
		plateau_scheduler.step(test_accuracy)

		if plateau_scheduler.is_equiv_to_best_epoch:
			print('Updating best model with epoch: {}'.format(epoch))
			best_model = copy.deepcopy(model)

			# update the global metrics with the best epoch
			train_stats.update_global(epoch)


	train_stats.export(output_dirpath)  # update metrics data on disk
	best_model.cpu()  # move to cpu before saving to simplify loading the model
	torch.save(best_model, os.path.join(output_dirpath, 'model.pt'))



if __name__ == '__main__':
	main()
	
