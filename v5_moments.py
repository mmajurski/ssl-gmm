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
import os
from PIL import Image
import numpy as np
import scipy.misc
import cifar10
import gauss_moments

class Net(nn.Module):
	def __init__(self, sY=32, sX=32, chan=3, num_classes=10, dim=64):
		super(Net, self).__init__()
		
		self.num_classes = num_classes
		self.dim    = dim
	
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
		self.centers = torch.tensor(centers, requires_grad=True).to(g_device)
		self.km_safe_pool = nn.MaxPool1d(10)
		
		#----------
		# The moments
		#----------
		moment_1 = gauss_moments.GaussMoments(dim,1)   # position
		moment_2 = gauss_moments.GaussMoments(dim,2)   # variance
		moment_3 = gauss_moments.GaussMoments(dim,3)   # skew
		moment_4 = gauss_moments.GaussMoments(dim,4)   # kutorsis
		
		# moment weights (for moment loss function)
		self.moment1_weight = torch.tensor(moment_1.moment_weights, requires_grad=False).to(g_device)
		self.moment2_weight = torch.tensor(moment_2.moment_weights, requires_grad=False).to(g_device)
		self.moment3_weight = torch.tensor(moment_3.moment_weights, requires_grad=False).to(g_device)
		self.moment4_weight = torch.tensor(moment_4.moment_weights, requires_grad=False).to(g_device)

		# gaussian moments
		self.gauss_moments1 = torch.tensor(moment_1.joint_gauss_moments, requires_grad=False).to(g_device)
		self.gauss_moments2 = torch.tensor(moment_2.joint_gauss_moments, requires_grad=False).to(g_device)
		self.gauss_moments3 = torch.tensor(moment_3.joint_gauss_moments, requires_grad=False).to(g_device)
		self.gauss_moments4 = torch.tensor(moment_4.joint_gauss_moments, requires_grad=False).to(g_device)


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
		x = torch.tanh(x)
		x = self.fc2(x)	
		x = torch.tanh(x)
		x = self.fc3(x)	
		
		
		#-------------------------------------
		# The k-means layer
		#-------------------------------------
		x_size = x.size()
		batch  = x_size[0]    # batch size
		dim    = self.dim     # number of internal dimensinos
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


		cluster_dist_sq_onehot    = cluster_assignment_onehot * dist_sq
		cluster_dist_sq           = torch.sum(cluster_dist_sq_onehot, dim=-1)

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

		moment1       = torch.sum(diff_onehot, axis=0)
		moment1_count = cluster_weight.unsqueeze(1).repeat(1,self.dim)
		moment1       = moment1 / (moment1_count + 0.0000001)

		moment2_a     = diff_onehot.unsqueeze(2)
		moment2_b     = diff_onehot.unsqueeze(3)
		moment2_a_rep = moment2_a.repeat((1,1,dim,1))		
		moment2_b_rep = moment2_b.repeat((1,1,1,dim))		
		moment2 = moment2_a_rep * moment2_b_rep
		moment2 = torch.sum(moment2, axis=0)
		moment2_count = moment1_count.unsqueeze(2).repeat((1,1,dim))
		moment2       = moment2 / (moment2_count + 0.0000001)
		
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
		#     where
		#  c = pow(a, 1/(1-a))
		#  b = pow(a, a/(1-a))
		#
		#  precomputed values
		#    moment 1   no formula required, it's perfectly linear
		#    moment 2   a = 1/2  c = 0.25           b = 0.5
		#    moment 3   a = 1/3  c = 0.19245008973  b = 0.57735026919
		#    moment 4   a = 1/4  c = 0.15749013123  b = 0.62996052494
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

#		print("moment_penalty")
#		print(moment_penalty1)
#		print(moment_penalty2)
#		print(moment_penalty3)
#		print(moment_penalty4)

		return [resp, moment_penalty1, moment_penalty2, moment_penalty3, moment_penalty4]


#-------------------------------------------------
#-------------------------------------------------
# Utilities to write out images
#-------------------------------------------------
#-------------------------------------------------

def to_img(A, cmin=-9999, cmax=-9999):
	cmin=np.amin(A) if cmin==-9999 else cmin
	cmax=np.amax(A) if cmax==-9999 else cmax
	#print("to_img cmin", cmin, "cmax", cmax)
	img=(A-cmin) * (255.0/(cmax-cmin))
	img[img<0]=0
	img[img>255]=255
	img=img.astype(np.uint8)
	return img

def mkdir(outdir):
	if not os.path.exists(outdir):
		os.mkdir(outdir)

#-------------------------------------------------
#-------------------------------------------------
# Main
#-------------------------------------------------
#-------------------------------------------------

#----------------
# Pytorch models and data
#----------------
g_train_data   = None
g_test_data    = None
g_device       = None
g_model        = None
g_train_loader = None
g_test_loader  = None
g_optimizer    = None

#----------------
# Read Command Args
#----------------
arg_parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
arg_parser.add_argument('--batch-size', type=int, default=48, metavar='N', help='input batch size for training (default: 64)')
arg_parser.add_argument('--blocks', type=int, default=8, metavar='N', help='number of blocks to add to the autoencoder (default: 8)')
arg_parser.add_argument('--epochs', type=int, default=32, metavar='N', help='number of epochs to train (default: 3)')
arg_parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
arg_parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
arg_parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
arg_parser.add_argument('--no-mps', action='store_true', default=False,  help='disables macOS GPU training')
arg_parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
arg_parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
arg_parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
arg_parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
args = arg_parser.parse_args()

# command arguments are of course globally acessible
arg_batch_size   = args.batch_size
arg_blocks       = args.blocks
arg_epochs       = args.epochs
arg_lr           = args.lr
arg_gamma        = args.gamma
arg_use_cuda     = not args.no_cuda and torch.cuda.is_available()
arg_use_mps      = not args.no_mps and torch.backends.mps.is_available()
arg_dry_run      = args.dry_run
arg_seed         = args.seed
arg_log_interval = args.log_interval
arg_save_model   = args.save_model


#----------------
# Initialize PyTorch
#----------------
torch.manual_seed(arg_seed)

if arg_use_cuda:
	g_device = torch.device("cuda")
elif arg_use_mps:
	g_device = torch.device("mps")
else:
	g_device = torch.device("cpu")


#----------------
# Read the dataset
#----------------
((g_train_x_numpy,g_train_y_numpy),(g_test_x_numpy,g_test_y_numpy)) = cifar10.readCifar10()

g_num_train = g_train_y_numpy.shape[0]
g_num_test  = g_test_y_numpy.shape[0]
g_num_class = 10

g_train_pred_numpy = np.zeros([g_num_train], dtype=np.int32)
g_test_pred_numpy  = np.zeros([g_num_test], dtype=np.int32)


#----------------
# Send dataset to the device
#----------------
g_train_x = torch.tensor(g_train_x_numpy,requires_grad=False).float().to(g_device)
g_train_y = torch.tensor(g_train_y_numpy,requires_grad=False).long().to(g_device)
g_test_x  = torch.tensor(g_test_x_numpy,requires_grad=False).float().to(g_device)
g_test_y  = torch.tensor(g_test_y_numpy,requires_grad=False).long().to(g_device)

# normalize the dataset to 0-1
g_train_x /= 255.0
g_test_x  /= 255.0



#----------------
# Create the model
#----------------
g_model = Net(32,32,3,10,10).to(g_device)
g_model_params = g_model.parameters()

#----------------
# Create the optimizer and scheduler
#----------------
g_optimizer = optim.Adadelta(g_model_params, lr=arg_lr)
torch.nn.utils.clip_grad_norm_(g_model_params, max_norm=1.0, norm_type=2.0)
g_scheduler = StepLR(g_optimizer, step_size=1000, gamma=arg_gamma)


#------------------------------------------------------
# Main training loop
#------------------------------------------------------

#-------------------------------------
# For every epoch
#-------------------------------------
for epoch in range(1, arg_epochs+1):

	print("+----------------------------------------+")
	print("| epoch %3d                              |" % epoch)
	print("+----------------------------------------+")

	epoch_loss = 0.0
	num_batch = g_num_train // arg_batch_size

	#--------------------------------------
	# For every minibatch
	#--------------------------------------
	for batch_idx in range(num_batch):
	
		# Grab the minibatch from the dataset
		b0 = batch_idx * arg_batch_size
		b1 = b0 + arg_batch_size
		real_data  = g_train_x[b0:b1]
		real_label = g_train_y[b0:b1]			

		#if (batch_idx % 100 == 0):
		#	print("batch", batch_idx, "of", g_num_train / arg_batch_size)

		# no partial batches
		batch_size  = real_label.size()[0]
		if (batch_size != arg_batch_size):
			break

		# One_hot encode
		y = F.one_hot(real_label,g_num_class)

		#---------
		# Train the model
		#---------
		model_out = g_model(real_data,y)
		yhat = model_out[0]
		mp1 = model_out[1]
		mp2 = model_out[2]
		mp3 = model_out[3]
		mp4 = model_out[4]
		
		# Cross entropy loss (by hand)
		minus_one_over_N = (-1.0 / (batch_size*g_num_class))
		log_yhat=torch.log(torch.clamp(yhat,min=0.0001))
		log_one_minus_yhat=torch.log(torch.clamp(1.0-yhat,min=0.0001))
		presum=(y * log_yhat + (1.0-y)*log_one_minus_yhat) * minus_one_over_N
		bce_loss = torch.sum(  presum  )

		# MoM loss
		mom_loss = 1.0 * (mp1 + 0.5*mp2 + 0.25*mp3 + 0.125*mp4)
		
		loss = bce_loss + mom_loss

		# gradient descent !
		loss.backward()
		g_optimizer.step()			
		
		# calculate accuracy
		ypred = torch.argmax(yhat, dim=1)
		#print(yhat)
		#print("yhat")
		#input("enter")

		#print(ypred)
		#print("ypred")
		#input("enter")

		g_train_pred_numpy[b0:b1] = ypred.detach().cpu().numpy()
		
		#------------
		# Print diagnostics
		#------------
		bce_loss = bce_loss.detach().cpu().numpy()
		mom_loss = mom_loss.detach().cpu().numpy()
		loss     = loss.detach().cpu().numpy()
		
		if (batch_idx % 100 == 0):
			print("batch", batch_idx, "of", g_num_train / arg_batch_size,
				"loss", loss, "bce_loss", bce_loss, "mom_loss", mom_loss)		

		epoch_loss += loss
		
	#---------------------------
	# End of epoch, print out the loss
	#---------------------------
	
	n_train = num_batch * arg_batch_size
	n_train_correct = 0
	for i in range(n_train):
		print("true %d pred %d" % (g_train_y_numpy[i], g_train_pred_numpy[i]))
		if g_train_pred_numpy[i] == g_train_y_numpy[i]:
			n_train_correct+=1
	train_accuracy = 100.0 * n_train_correct / n_train
	
	epoch_loss /= num_batch
	print("epoch ", epoch, "loss", epoch_loss, "train_accuracy", train_accuracy)


print("Success!")