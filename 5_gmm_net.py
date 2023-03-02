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
import sys

import lcl_models


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		dim=10
		
		#----------
		# The architecture
		#----------
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, dim)

		self.gmm_layer = lcl_models.gmm_layer(dim, dim)
		
		#----------
		# The k-means layer
		#----------
		self.km_linear    = nn.Linear(1,dim*dim)
		
		# A list of linear layers for parameters for L/D for k'th class
		self.chol_linear = [None]*dim
		for k in range(dim):
			self.chol_linear[k]  = nn.Linear(1,dim*dim)

			# At first assign Sigma to identity matrix (for stability)
			self.chol_linear[k].weight.requires_grad=False
			self.chol_linear[k].bias.requires_grad=False
			i=0
			for y in range(dim):
				for x in range(dim):
					self.chol_linear[k].weight[i,0] = 0
					self.chol_linear[k].bias[i] = float(y==x)
					i+=1
			self.chol_linear[k].weight.requires_grad=True
			self.chol_linear[k].bias.requires_grad=True
		
		
		self.chol_linear = torch.nn.ModuleList( self.chol_linear )   # Convert list to module list

		self.km_safe_pool = nn.MaxPool1d(dim)

	def forward(self, x):

		#
		#  Sigma  = L D Lt
		#
		#  Sigma_inv  = Lt-1 D-1 L-1
		#

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

		output = self.gmm_layer(x)
		return output

		#---------
		# The k-means layer
		#---------

		x_size = x.size()
		batch  = x_size[0]    # batch size
		dim    = x_size[1]    # number of classes (usually)


		#---
		# Construct cluster centers
		#---

		# Create an all-zeros tensor of size [1]
		zeros = x-x
		zeros = torch.sum(zeros)
		zeros = torch.reshape(zeros,(1,))

		# Create the trainable cluster centers of size [dim,dim]
		centers = self.km_linear(zeros)
		centers = torch.reshape(centers,(dim,dim))

		centers = self.gmm_layer.centers

		#---
		# Construct the Sigma matrix through
		#  LDL decomposition
		#---

		det       = x-x
		det       = torch.sum(det,0)
		Sigma     = [None] * dim
		Sigma_inv = [None] * dim

		for k in range(dim):

			# Hack, the raw weights are stored in chol_linear[k]
			raw_mat = self.chol_linear[k](zeros)
			raw_mat = torch.reshape(raw_mat,(dim,dim))
			raw_mat = self.gmm_layer.L[k,]
	
			# Construct the L matrix for LDL
			L       = torch.tril(raw_mat,diagonal=-1)
			L       = L.fill_diagonal_(1.0)              # L           (lower triangular)
			Lt      = torch.transpose(L,0,1)             # L-transpose (upper triangular)

			# Construct diagonal D which must be positive
			root_D  = torch.diag(raw_mat)
			D       = root_D * root_D + 0.0001           # Diagonal D
			D_embed = torch.diag_embed(D)                # Upsample to NxN diagonal matrix

			#---
			# Construct the Covariance Matrix Sigma
			#---
			LD = torch.mm(L, D_embed)
			Sigma[k] = torch.mm(LD,Lt)                   # k'th Sigma matrix
		
			#---
			# Construct the inverse Covariance Matrix Sigma_inv
			#---
			Identity = raw_mat-raw_mat
			Identity.fill_diagonal_(1.0)
			
			# L inverse
			L_inv = torch.linalg.solve_triangular(L,Identity,upper=False)
			L_inv_t = torch.transpose(L_inv,0,1)	
			
			# D inverse
			D_inv = 1.0 / D
			D_inv_embed = torch.diag_embed(D_inv)

			# Sigma inverse
			D_inv_L_inv = torch.mm(D_inv_embed,L_inv)
			Sigma_inv[k] = torch.mm(L_inv_t, D_inv_L_inv)

			#Identity22 = torch.mm(Sigma,Sigma_inv)
			#print("Identity22", Identity21)
			#print("Identity22", Identity21.size())
			#input("press enter")

			# Determinant
			det[k] = torch.prod(D,0)
			
		det_scale = torch.rsqrt(det)

		#---
		# Calculate distance to cluster centers
		#---

		# Upsample the x-data to [batch, dim, dim]
		x_rep = x.unsqueeze(1).repeat(1, dim, 1)

		# Upsample the clusters to [batch, dim, dim
		centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

		# Subtract to get diff of [batch, dim, dim]
		diff = x_rep - centers_rep
		
		#---
		# Switch it up to get the differences
		#---

		dist_sq = torch.sum(diff,2)  # initially set to zero
		dist_sq = dist_sq - dist_sq
		
		# Calculate each dist_sq entry separately
		for k in range(dim):
			curr_diff   = diff[:,k]
			curr_diff_t = torch.transpose(curr_diff,0,1)
			Sig_inv_curr_diff_t = torch.mm(Sigma_inv[k], curr_diff_t)
			Sig_inv_curr_diff_t_t = torch.transpose(Sig_inv_curr_diff_t,0,1)
			curr_dist_sq = curr_diff * Sig_inv_curr_diff_t_t
			curr_dist_sq = torch.sum(curr_dist_sq, 1)			
			dist_sq[:,k] = curr_dist_sq

		#   GMM
		# dist_sq = (x-mu) Sigma_inv (x-mu)T
		#   K-means
		# dist_sq = (x-mu) dot (x-mu)
		
		# Obtain the square distance to each cluster
		#  of size [batch, dim]
		#dist_sq = diff*diff
		#dist_sq = torch.sum(dist_sq,2)
		

		# Obtain the exponents
		expo = -0.5*dist_sq

		det_scale_rep = det_scale.unsqueeze(0).repeat(batch, 1)
		
		# Calculate the true numerators and denominators
		#  (we don't use this directly for responsibility calculation
		#   we actually use the "safe" versions that are shifted
		#   for stability)
		# Note 0.00010211761 = (2*pi)^(-dim/2) where dim=10
		#
		numer = 0.00010211761 * det_scale_rep * torch.exp(expo)
		denom = torch.sum(numer,1)
		denom = denom.unsqueeze(1).repeat(1, dim)		
		
		# Obtain the "safe" (numerically stable) versions of the
		#  exponents.  These "safe" exponents produce fake numer and denom
		#  but guarantee that resp = fake_numer / fake_denom = numer / denom
		#  where fake_numer and fake_denom are numerically stable
		expo_safe_off = self.km_safe_pool(expo)
		expo_safe_off = expo_safe_off.repeat(1,dim)
		expo_safe = expo - expo_safe_off

		# Calculate the responsibilities
		numer_safe = det_scale_rep * torch.exp(expo_safe)
		denom_safe = torch.sum(numer_safe,1)
		denom_safe = denom_safe.unsqueeze(1).repeat(1, dim)
		resp = numer_safe / denom_safe

		# Use the k-means layer
		#     not we take log, because this model
		#     outputs log softmax
		# output = torch.log(resp)
		
		# Uncomment for regular log-softmax layer
#		output = F.log_softmax(x, dim=1)

		output = resp

		return output



def train(args, model, device, train_loader, optimizer, epoch):
	model.train()

	criterion = torch.nn.CrossEntropyLoss()

	import time
	start_time = time.time()



	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		target_size = target.size()
		batch_size = target_size[0]
		num_class  = 10

		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)

		# output = output[:,:num_class]
		#
		# # calculate y and yhat
		# y=F.one_hot(target,num_class)
		# yhat=torch.softmax(output,dim=1)
		#
		# # implement log_loss by hand
		# minus_one_over_N = (-1.0 / (batch_size*num_class))
		#
		# log_yhat=torch.log(torch.clamp(yhat,min=0.0001))
		# log_one_minus_yhat=torch.log(torch.clamp(1.0-yhat,min=0.0001))
		# presum=(y * log_yhat + (1.0-y)*log_one_minus_yhat) * minus_one_over_N
		# loss = torch.sum(  presum  )
		
		loss.backward()
		optimizer.step()
		
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
			if args.dry_run:
				break
				
		# if batch_idx * len(data) >= len(train_loader.dataset)/4:
		# 	break

	print("Epoch took {}s".format(time.time() - start_time))



def test(model, device, test_loader):
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
			output = output[:,:num_class]

			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=14, metavar='N',
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
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
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
		cuda_kwargs = {'num_workers': 10,
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

	# layers = dict(model.named_modules())
	# print(layers)
	
#	sys.exit(0)
#	#input ("press enter")
	
	
	optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

	scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		test(model, device, test_loader)
		scheduler.step()

	if args.save_model:
		torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
	main()
	
