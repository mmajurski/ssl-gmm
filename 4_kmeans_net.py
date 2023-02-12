from typing import Optional, Union
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
import torchvision.models
import torchvision.models.resnet as resnet
import sys


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class kMeans(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.dim = dim
		self.centers = nn.Parameter(torch.rand(size=(self.dim, self.dim), requires_grad=True))


	def forward(self, x):
		batch = x.size()[0]  # batch size

		# TODO make this work with num_classes x feature embedding length elements

		# ---
		# Calculate distance to cluster centers
		# ---

		# Upsample the x-data to [batch, dim, dim]
		x_rep = x.unsqueeze(1).repeat(1, self.dim, 1)

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

		# Calculate the responsibilities
		numer_safe = torch.exp(expo_safe)
		denom_safe = torch.sum(numer_safe, 1)
		denom_safe = denom_safe.unsqueeze(1).repeat(1, self.dim)
		resp = numer_safe / denom_safe

		output = torch.log(resp)

		return output



class kMeansResNet18(nn.Module):
	def __init__(self, num_classes):
		super(kMeansResNet18, self).__init__()

		# TODO work out how to cluster in the 512 dim second to last layer
		self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
		# self.model.fc = Identity()  # replace the fc layer with an identity to ensure it does nothing


		self.dim = num_classes
		# kmeans layer
		self.kmeans = kMeans(dim=self.dim)


	def forward(self, x):
		x = self.model(x)  # get the feature embedding of 512 dim
		output = self.kmeans(x)
		return output


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
		self.fc2 = nn.Linear(128, 10)
		
		#----------
		# The k-means layer
		#----------
		self.km_linear    = nn.Linear(1,100)
		self.km_safe_pool = nn.MaxPool1d(10)

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

		#---
		# Calculate distance to cluster centers
		#---
		
		# Upsample the x-data to [batch, dim, dim]
		x_rep = x.unsqueeze(1).repeat(1, dim, 1)

		# Upsample the clusters to [batch, 10, 10]
		centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

		# Subtract to get diff of [batch, 10, 10]
		diff = x_rep - centers_rep
		
		# Obtain the square distance to each cluster
		#  of size [batch, dim]
		dist_sq = diff*diff
		dist_sq = torch.sum(dist_sq,2)

		# Obtain the exponents
		expo = -0.5*dist_sq
		
		# Calculate the true numerators and denominators
		#  (we don't use this directly for responsibility calculation
		#   we actually use the "safe" versions that are shifted
		#   for stability)
		# Note 0.00010211761 = (2*pi)^(-dim/2) where dim=10
		#
		numer = 0.00010211761 * torch.exp(expo)
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
		numer_safe = torch.exp(expo_safe)
		denom_safe = torch.sum(numer_safe,1)
		denom_safe = denom_safe.unsqueeze(1).repeat(1, dim)
		resp = numer_safe / denom_safe

		# Use the k-means layer
		#     not we take log, because this model
		#     outputs log softmax
		output = torch.log(resp)
		
		# Uncomment for regular log-softmax layer
#		output = F.log_softmax(x, dim=1)

		return output



def train(args, model, device, train_loader, optimizer, epoch):
	model.train()

	# Setup loss criteria
	criterion = torch.nn.CrossEntropyLoss()


	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		target_size = target.size()
		batch_size = target_size[0]
		num_class  = 10

		optimizer.zero_grad()
		output = model(data)

		batch_loss = criterion(output, target)
		batch_loss.backward()
		optimizer.step()


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
		# batch_loss = torch.sum(  presum  )
		#
		# batch_loss.backward()
		# optimizer.step()
		
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), batch_loss.item()))
			if args.dry_run:
				break



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
	return test_loss


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=14, metavar='N',
						help='number of epochs to train (default: 14)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
		# check if IDE is in debug mode, and set the args debug flag and set num parallel worker to 0
		import utils
		if utils.is_ide_debug():
			cuda_kwargs['num_workers'] = 0

		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	use_MNIST = False
	if use_MNIST:
		model = Net().to(device)

		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
			])
		train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
		test_dataset = datasets.MNIST('../data', train=False, transform=transform)
	else:
		# model = kMeansResNet18(num_classes=10).to(device)
		model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)


		import cifar_datasets
		train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_WEAK_TRAIN, train=True)
		test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

	train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
	test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

	#input("10")

	# model = Net().to(device)
	# model = kMeansResNet18().to(device)

	
	#input("20")

	#layers = dict(model.named_children())

	# layers = dict(model.named_modules())
	# print(layers)
	
#	sys.exit(0)
#	#input ("press enter")
	
	
	# optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	import lr_scheduler
	plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, threshold=1e-4, max_num_lr_reductions=2)

	# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
	epoch = -1
	MAX_EPOCHS = 200
	while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
		epoch += 1
		train(args, model, device, train_loader, optimizer, epoch)
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		test_loss = test(model, device, test_loader)
		plateau_scheduler.step(test_loss)

	if args.save_model:
		torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
	main()
	
