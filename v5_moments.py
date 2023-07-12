import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy

import numpy as np
import scipy.misc
import psutil
import utils
import cifar_datasets
import gauss_moments
import lr_scheduler
import metadata


class Net(nn.Module):
	def __init__(self, sY=32, sX=32, chan=3, num_classes=10, dim=64, use_tanh=False):
		super(Net, self).__init__()
		
		self.num_classes = num_classes
		self.dim	= dim
		self.use_tanh = use_tanh
	
		#----------
		# The architecture
		#----------
		self.conv1 = nn.Conv2d(chan, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.fc1 = nn.Linear((sY-4)*(sX-4)*64//4, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, dim)
	
		#----------
		# The k-means layer
		#----------
		#centers = np.zeros([num_classes,dim], dtype=np.float32)
		#centers = np.random.normal(0.0, 1.0, size=[num_classes,dim]).astype(np.float32)
		centers = np.eye(num_classes,dim).astype(np.float32)
		self.centers = torch.nn.Parameter(torch.tensor(centers), requires_grad=True)
		# self.centers = torch.tensor(centers, requires_grad=True).to(device)
		self.km_safe_pool = nn.MaxPool1d(10)
		
		#----------
		# The moments
		#----------
		moment_1 = gauss_moments.GaussMoments(dim,1)   # position
		moment_2 = gauss_moments.GaussMoments(dim,2)   # variance
		moment_3 = gauss_moments.GaussMoments(dim,3)   # skew
		moment_4 = gauss_moments.GaussMoments(dim,4)   # kutorsis
		
		# moment weights (for moment loss function)
		self.moment1_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)
		self.moment2_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)
		self.moment3_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)
		self.moment4_weight = torch.nn.Parameter(torch.tensor(moment_1.moment_weights), requires_grad=False)

		# gaussian moments
		self.gauss_moments1 = torch.nn.Parameter(torch.tensor(moment_1.joint_gauss_moments), requires_grad=False)
		self.gauss_moments2 = torch.nn.Parameter(torch.tensor(moment_2.joint_gauss_moments), requires_grad=False)
		self.gauss_moments3 = torch.nn.Parameter(torch.tensor(moment_3.joint_gauss_moments), requires_grad=False)
		self.gauss_moments4 = torch.nn.Parameter(torch.tensor(moment_4.joint_gauss_moments), requires_grad=False)

		# # moment weights (for moment loss function)
		# self.moment1_weight = torch.tensor(moment_1.moment_weights, requires_grad=False).to(device)
		# self.moment2_weight = torch.tensor(moment_2.moment_weights, requires_grad=False).to(device)
		# self.moment3_weight = torch.tensor(moment_3.moment_weights, requires_grad=False).to(device)
		# self.moment4_weight = torch.tensor(moment_4.moment_weights, requires_grad=False).to(device)
		#
		# # gaussian moments
		# self.gauss_moments1 = torch.tensor(moment_1.joint_gauss_moments, requires_grad=False).to(device)
		# self.gauss_moments2 = torch.tensor(moment_2.joint_gauss_moments, requires_grad=False).to(device)
		# self.gauss_moments3 = torch.tensor(moment_3.joint_gauss_moments, requires_grad=False).to(device)
		# self.gauss_moments4 = torch.tensor(moment_4.joint_gauss_moments, requires_grad=False).to(device)


	def forward(self, x, y_onehot=None):

		#----------
		# The architecture
		#----------		

		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		if self.use_tanh:
			x = torch.tanh(x)
		else:
			x = torch.relu(x)
		x = self.fc2(x)	
		if self.use_tanh:
			x = torch.tanh(x)
		else:
			x = torch.relu(x)
		x = self.fc3(x)	
		
		
		#-------------------------------------
		# The k-means layer
		#-------------------------------------
		x_size = x.size()
		batch  = x_size[0]	# batch size
		dim	= self.dim	 # number of internal dimensinos
		num_classes = self.num_classes  # number of classes

		# Upsample the x-data to [batch, num_classes, dim]
		x_rep = x.unsqueeze(1).repeat(1, num_classes, 1)

		# Upsample the clusters to [batch, 10, 10]
		centers = self.centers
		centers_rep = centers.unsqueeze(0).repeat(batch, 1, 1)

#		print(centers)
#		print("centers")
#		input("enter")

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
		denom = denom.unsqueeze(1).repeat(1, num_classes)		
		
		# Obtain the "safe" (numerically stable) versions of the
		#  exponents.  These "safe" exponents produce fake numer and denom
		#  but guarantee that resp = fake_numer / fake_denom = numer / denom
		#  where fake_numer and fake_denom are numerically stable
		expo_safe_off = self.km_safe_pool(expo)
		expo_safe_off = expo_safe_off.repeat(1,num_classes)
		expo_safe = expo - expo_safe_off

		# Calculate the responsibilities
		numer_safe = torch.exp(expo_safe)
		denom_safe = torch.sum(numer_safe,1)
		denom_safe = denom_safe.unsqueeze(1).repeat(1, num_classes)
		resp = numer_safe / denom_safe

		#-------------------------------------
		# The moments penalty
		#-------------------------------------
		
		#-----
		# Vectorized version of cluster_dist
		#-----
		
		# Obtain cluster assignment from dist_sq directly
		cluster_assignment = torch.argmin(dist_sq, dim=-1)

		# Use one-hot encoding trick to extract the dist_sq
		if y_onehot is not None:
			cluster_assignment_onehot = y_onehot
		else:
			cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, dist_sq.shape[1])


		cluster_dist_sq_onehot	= cluster_assignment_onehot * dist_sq
		cluster_dist_sq		   = torch.sum(cluster_dist_sq_onehot, dim=-1)

		# Take square root of dist_sq to get L2 norm
		cluster_dist = torch.sqrt(cluster_dist_sq)

		#----------------------------------------
		# Calculate the empirical moments
		#   OUTPUT:  moment1  [classes dim]
		#   OUTPUT:  moment2  [classes dim dim]
		#   OUTPUT:  moment3  [classes dim dim dim]
		#   OUTPUT:  moment4  [classes dim dim dim dim]
		#----------------------------------------
		cluster_weight = torch.sum(cluster_assignment_onehot,axis=0)
		cluster_assignment_onehot_rep = cluster_assignment_onehot.unsqueeze(2).repeat(1, 1, self.dim)

		diff_onehot = diff * cluster_assignment_onehot_rep

		moment1	   = torch.sum(diff_onehot, axis=0)
		moment1_count = cluster_weight.unsqueeze(1).repeat(1,self.dim)
		moment1	   = moment1 / (moment1_count + 0.0000001)

		moment2_a	 = diff_onehot.unsqueeze(2)
		moment2_b	 = diff_onehot.unsqueeze(3)
		moment2_a_rep = moment2_a.repeat((1,1,dim,1))		
		moment2_b_rep = moment2_b.repeat((1,1,1,dim))		
		moment2 = moment2_a_rep * moment2_b_rep
		moment2 = torch.sum(moment2, axis=0)
		moment2_count = moment1_count.unsqueeze(2).repeat((1,1,dim))
		moment2	   = moment2 / (moment2_count + 0.0000001)
		
		moment3_a = moment2_a.unsqueeze(2)
		moment3_b = moment2_b.unsqueeze(2)
		moment3_c = moment2_b.unsqueeze(4)
		moment3_a_rep = moment3_a.repeat((1,1,dim,dim,1))
		moment3_b_rep = moment3_b.repeat((1,1,dim,1,dim))
		moment3_c_rep = moment3_c.repeat((1,1,1,dim,dim))		
		moment3 = moment3_a_rep * moment3_b_rep * moment3_c_rep
		moment3 = torch.sum(moment3, axis=0)
		
		moment4_a = moment3_a.unsqueeze(2)
		moment4_b = moment3_b.unsqueeze(2)
		moment4_c = moment3_c.unsqueeze(2)
		moment4_d = moment3_c.unsqueeze(5)
		moment4_a_rep = moment4_a.repeat((1,1,dim,dim,dim,1))
		moment4_b_rep = moment4_b.repeat((1,1,dim,dim,1,dim))
		moment4_c_rep = moment4_c.repeat((1,1,dim,1,dim,dim))
		moment4_d_rep = moment4_d.repeat((1,1,1,dim,dim,dim))
		moment4 = moment4_a_rep * moment4_b_rep * moment4_c_rep * moment4_d_rep
		moment4 = torch.sum(moment4, axis=0)
		
		
		#---------------------------------------
		# calculate the moment loss
		#---------------------------------------
		
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
		moment2 = torch.sign(torch.sign(moment2)+0.1)*(torch.pow(torch.abs(moment2)+0.25,0.5) - 0.5)
		moment3 = torch.sign(torch.sign(moment3)+0.1)*(torch.pow(torch.abs(moment3)+0.19245008973,0.3333333333) - 0.57735026919)
		moment4 = torch.sign(torch.sign(moment4)+0.1)*(torch.pow(torch.abs(moment4)+0.15749013123,0.25) - 0.62996052494)
		moment2_target = torch.sign(torch.sign(moment2_target)+0.1)*(torch.pow(torch.abs(moment2_target)+0.25,0.5) - 0.5)
		moment3_target = torch.sign(torch.sign(moment3_target)+0.1)*(torch.pow(torch.abs(moment3_target)+0.19245008973,0.3333333333) - 0.57735026919)
		moment4_target = torch.sign(torch.sign(moment4_target)+0.1)*(torch.pow(torch.abs(moment4_target)+0.15749013123,0.25) - 0.62996052494)


#		print(moment2_target)
#		print("moment2_target")
#		input("press enter")
	
		# repeat the moment targets per class
		moment1_target = moment1_target.unsqueeze(0).repeat(num_classes,1)
		moment2_target = moment2_target.unsqueeze(0).repeat(num_classes,1,1)
		moment3_target = moment3_target.unsqueeze(0).repeat(num_classes,1,1,1)
		moment4_target = moment4_target.unsqueeze(0).repeat(num_classes,1,1,1,1)
		
		# repeat the moment penalty weights perclass
		cluster_weight_norm = cluster_weight / torch.sum(cluster_weight)
#		print(cluster_weight_norm)
#		print("cluster_weight_norm")
#		input("enter")

		cluster_weight_rep = cluster_weight_norm.unsqueeze(1).repeat((1,dim))
		moment1_weight = cluster_weight_rep * self.moment1_weight.unsqueeze(0).repeat((num_classes,1))

		cluster_weight_rep = cluster_weight_rep.unsqueeze(2).repeat((1,1,dim))
		moment2_weight = cluster_weight_rep * self.moment2_weight.unsqueeze(0).repeat((num_classes,1,1))

		cluster_weight_rep = cluster_weight_rep.unsqueeze(3).repeat((1,1,1,dim))
		moment3_weight = cluster_weight_rep * self.moment3_weight.unsqueeze(0).repeat((num_classes,1,1,1))

		cluster_weight_rep = cluster_weight_rep.unsqueeze(4).repeat((1,1,1,1,dim))
		moment4_weight = cluster_weight_rep * self.moment4_weight.unsqueeze(0).repeat((num_classes,1,1,1,1))

		# calculate the penalty loss function
		moment_penalty1 = torch.sum( moment1_weight*torch.pow( (moment1 - moment1_target), 2 ) )
		moment_penalty2 = torch.sum( moment2_weight*torch.pow( (moment2 - moment2_target), 2 ) )
		moment_penalty3 = torch.sum( moment3_weight*torch.pow( (moment3 - moment3_target), 2 ) )
		moment_penalty4 = torch.sum( moment4_weight*torch.pow( (moment4 - moment4_target), 2 ) )

		resp = torch.clip(resp, min=1e-8)
		resp = torch.log(resp)

		return [resp, moment_penalty1, moment_penalty2, moment_penalty3, moment_penalty4]


def train(args, model, device, train_loader, optimizer, epoch, train_stats):
	model.train()

	# Setup loss criteria
	criterion = torch.nn.CrossEntropyLoss()

	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		data, target = data.to(device), target.to(device)
		target_onehot = torch.nn.functional.one_hot(target, num_classes=10)
		# model_out = model(data, target_onehot)
		model_out = model(data)
		yhat = model_out[0]
		mp1 = model_out[1]
		mp2 = model_out[2]
		mp3 = model_out[3]
		mp4 = model_out[4]

		bce_loss = criterion(yhat, target)
		# MoM loss
		mom_loss = 1.0 * (mp1 + 0.5 * mp2 + 0.25 * mp3 + 0.125 * mp4)

		if args.disable_moments:
			batch_loss = bce_loss
		else:
			batch_loss = bce_loss + mom_loss
			train_stats.append_accumulate('train_ce_loss_comp', bce_loss.item())
			train_stats.append_accumulate('train_weighted_mom_loss_comp', mom_loss.item())
			train_stats.append_accumulate('train_1_mom_loss_comp', mp1.item())
			train_stats.append_accumulate('train_2_mom_loss_comp', mp2.item())
			train_stats.append_accumulate('train_3_mom_loss_comp', mp3.item())
			train_stats.append_accumulate('train_4_mom_loss_comp', mp4.item())

		if torch.isnan(batch_loss):
			print("nan loss")

		train_stats.append_accumulate('train_loss', batch_loss.item())
		acc = torch.argmax(yhat, dim=-1) == target
		train_stats.append_accumulate('train_accuracy', torch.mean(acc, dtype=torch.float32).item())

		batch_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)

		if batch_idx % args.log_interval == 0:
			# log loss and current GPU utilization
			cpu_mem_percent_used = psutil.virtual_memory().percent
			gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
			gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
			print('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, len(train_loader), batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

	train_stats.close_accumulate(epoch, 'train_loss', 'avg')
	if not args.disable_moments:
		train_stats.close_accumulate(epoch, 'train_ce_loss_comp', 'avg')
		train_stats.close_accumulate(epoch, 'train_weighted_mom_loss_comp', 'avg')
		train_stats.close_accumulate(epoch, 'train_1_mom_loss_comp', 'avg')
		train_stats.close_accumulate(epoch, 'train_2_mom_loss_comp', 'avg')
		train_stats.close_accumulate(epoch, 'train_3_mom_loss_comp', 'avg')
		train_stats.close_accumulate(epoch, 'train_4_mom_loss_comp', 'avg')
	train_stats.close_accumulate(epoch, 'train_accuracy', 'avg')


def test(model, device, test_loader, epoch, train_stats, args):
	model.eval()
	criterion = torch.nn.CrossEntropyLoss()

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)

			target_onehot = torch.nn.functional.one_hot(target, num_classes=10)

			# model_out = model(data, target_onehot)
			model_out = model(data)
			yhat = model_out[0]
			mp1 = model_out[1]
			mp2 = model_out[2]
			mp3 = model_out[3]
			mp4 = model_out[4]

			bce_loss = criterion(yhat, target)
			# MoM loss
			mom_loss = 1.0 * (mp1 + 0.5 * mp2 + 0.25 * mp3 + 0.125 * mp4)

			if args.disable_moments:
				batch_loss = bce_loss
			else:
				batch_loss = bce_loss + mom_loss
			# train_stats.append_accumulate('test_ce_loss_comp', bce_loss.item())
			# train_stats.append_accumulate('test_weighted_mom_loss_comp', mom_loss.item())
			# train_stats.append_accumulate('test_1_mom_loss_comp', mp1.item())
			# train_stats.append_accumulate('test_2_mom_loss_comp', mp2.item())
			# train_stats.append_accumulate('test_3_mom_loss_comp', mp3.item())
			# train_stats.append_accumulate('test_4_mom_loss_comp', mp4.item())

			train_stats.append_accumulate('test_loss', batch_loss.item())
			acc = torch.argmax(yhat, dim=-1) == target
			train_stats.append_accumulate('test_accuracy', torch.mean(acc, dtype=torch.float32).item())

	train_stats.close_accumulate(epoch, 'test_loss', 'avg')
	# train_stats.close_accumulate(epoch, 'test_ce_loss_comp', 'avg')
	# train_stats.close_accumulate(epoch, 'test_weighted_mom_loss_comp', 'avg')
	# train_stats.close_accumulate(epoch, 'test_1_mom_loss_comp', 'avg')
	# train_stats.close_accumulate(epoch, 'test_2_mom_loss_comp', 'avg')
	# train_stats.close_accumulate(epoch, 'test_3_mom_loss_comp', 'avg')
	# train_stats.close_accumulate(epoch, 'test_4_mom_loss_comp', 'avg')
	train_stats.close_accumulate(epoch, 'test_accuracy', 'avg')

	test_loss = train_stats.get_epoch('test_loss', epoch)
	test_acc = train_stats.get_epoch('test_accuracy', epoch)

	print('Test set: Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))
	return test_acc


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 1.0)')
	parser.add_argument('--ofn', type=str)
	parser.add_argument('--weight-decay', default=5e-4, type=float)
	parser.add_argument('--disable-moments', action='store_true', default=False)
	parser.add_argument('--tanh', action='store_true', default=False)
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	args = parser.parse_args()

	if torch.cuda.is_available():
		use_cuda = True
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.batch_size}
	if use_cuda:
		cuda_kwargs = {'num_workers': 2,
					   'pin_memory': True,
					   'shuffle': True}
		# check if IDE is in debug mode, and set the args debug flag and set num parallel worker to 0
		if utils.is_ide_debug():
			print("setting num_workers to 0")
			cuda_kwargs['num_workers'] = 0

		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	model = Net(32, 32, 3, 10, 10, use_tanh=args.tanh).to(device)
	# transform = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	transforms.Normalize((0.1307,), (0.3081,))
	# ])
	# train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
	# test_dataset = datasets.MNIST('../data', train=False, transform=transform)
	train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_WEAK_TRAIN, train=True)
	test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

	train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
	test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

	optimizer = utils.configure_optimizer(model, args.weight_decay, args.learning_rate, 'sgd')
	plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20, threshold=1e-4, max_num_lr_reductions=2)
	output_folder = './models-20230709/{}'.format(args.ofn)
	os.makedirs(output_folder, exist_ok=True)
	train_stats = metadata.TrainingStats()
	epoch = -1

	while not plateau_scheduler.is_done():
		epoch += 1
		train(args, model, device, train_loader, optimizer, epoch, train_stats)
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		test_acc = test(model, device, test_loader, epoch, train_stats, args)
		plateau_scheduler.step(test_acc)

		if plateau_scheduler.is_equiv_to_best_epoch:
			print('Updating best model with epoch: {} accuracy: {}'.format(epoch, test_acc))
			model.cpu()
			torch.save(model, os.path.join(output_folder, "model.pt"))
			model.to(device)

			# update the global metrics with the best epoch
			train_stats.update_global(epoch)

		train_stats.export(output_folder)  # update metrics data on disk
		train_stats.plot_all_metrics(output_folder)
	train_stats.export(output_folder)  # update metrics data on disk
	train_stats.plot_all_metrics(output_folder)




if __name__ == '__main__':
	main()

