import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cifar10


import torch_randaugment as tra
import orig_randaugment as ora

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

import os

#-------------------------------------------------
#-------------------------------------------------
# Main
#-------------------------------------------------
#-------------------------------------------------
g_device = torch.device("cuda")

print('read cifar10')
((g_train_x_numpy,g_train_y_numpy),(g_test_x_numpy,g_test_y_numpy)) = cifar10.readCifar10()

# Send dataset to the device
g_train_x = torch.tensor(g_train_x_numpy,requires_grad=False).float().to(g_device)
g_train_y = torch.tensor(g_train_y_numpy,requires_grad=False).long().to(g_device)
g_test_x  = torch.tensor(g_test_x_numpy,requires_grad=False).float().to(g_device)
g_test_y  = torch.tensor(g_test_y_numpy,requires_grad=False).long().to(g_device)

print('g_train_x', g_train_x.shape)
print('g_train_y', g_train_y.shape)
print('g_test_x',  g_test_x.shape)
print('g_test_y',  g_test_y.shape)
print('transform image')

#--------------------------
# Create a rand augmenter
#--------------------------

traug = tra.TorchRandAugment(clear_chance=0.0, store_choices=True, device=g_device)

oraug = ora.OrigRandAugmentTester(traug)

# make the output directory
try:
	os.mkdir("outimgs") 
except:
	print('warning outimgs exists')

# For every minibatch
batch_size = 1

for sidx in range(0, g_train_x.shape[0], batch_size):
	eidx = sidx+batch_size
	imgs = g_train_x[sidx:eidx]
	
	nimg = imgs.shape[0]
	chan = imgs.shape[1]
	sY   = imgs.shape[2]
	sX   = imgs.shape[3]
	
	imgs_traug = traug(imgs)
	imgs_oraug = oraug(imgs)

	# Convert images from tensor to numpy to pillow
	imgs_traug = imgs_traug.detach().cpu().numpy()
	imgs_oraug = imgs_oraug.detach().cpu().numpy()
	
	# write the image outputs
	for j in range(batch_size):
		oidx = sidx + j
		
		# create a numpy array of the appropriate size
		np_buf = np.zeros((chan,sY,2*sX),dtype=np.float32)
		np_buf[:,:,0:sX]      = imgs_traug[j,:,:,:]
		np_buf[:,:,sX:(2*sX)] = imgs_oraug[j,:,:,:]
		
		# clamp
		np_buf[np_buf<0.0] = 0.0
		np_buf[np_buf>255.0] = 255.0
		
		# convert to pillow
		img_rgb = np.transpose(np_buf, (1,2,0))
		PIL_img = Image.fromarray(np.uint8(img_rgb)).convert('RGB')

		# write the image
		outname = "outimgs/%05d.png" % (oidx)
		print("save", outname)
		PIL_img.save(outname)

print('Success')

