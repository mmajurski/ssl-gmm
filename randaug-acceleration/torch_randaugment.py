import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

#------------------------------
# Augmentation ops
#------------------------------
TRAUG_IDENTITY      = 0
TRAUG_AUTO_CONTRAST = 1
TRAUG_BRIGHTNESS    = 2
TRAUG_COLOR         = 3
TRAUG_CONTRAST      = 4
TRAUG_EQUALIZE      = 5
TRAUG_INVERT        = 6
TRAUG_POSTERIZE     = 7
TRAUG_ROTATE        = 8
TRAUG_SHARPNESS     = 9
TRAUG_SHEAR_X       = 10
TRAUG_SHEAR_Y       = 11
TRAUG_SOLARIZE      = 12
TRAUG_SOLARIZE_ADD  = 13
TRAUG_TRANSLATE_X   = 14
TRAUG_TRANSLATE_Y   = 15
TRAUG_CUTOUT        = 16

#------------------------------
# Maximum number of discreet buckets
#  for the augmentation parameter
#------------------------------
TRAUG_PARAMETER_MAX = 10

#-----------------------------------------------------------------
# Tiny wrapper around the most important pytorch routine
#   allows passing of multi-dim input and index
#   but the actually "indices" are still 1D ints
#-----------------------------------------------------------------
def traug_index_select(input_tensor, index_tensor):
	index_tensor_shape = index_tensor.shape
	input_tensor_flat  = torch.flatten(input_tensor)
	index_tensor_flat  = torch.flatten(index_tensor)
	
	result = torch.index_select(input_tensor_flat, 0, index_tensor_flat)
	result = torch.reshape(result, index_tensor_shape)
	return result

#-----------------------------------------------------------------
# torch_randaug_batch_affine
#   Perform affine transformation on a batch of images
#
#   Note: a separate affine transform matrix
#         is applied to each image
#
# input:
#   imgs     [nimg,chan,sY,sX]    batch of images BEFORE transformation
#   aff      [nimg,6]             batch of affine transform matrices
#   fillval   default:127         grayscale fill value for blank pixels
#   device    default:None        which pytorch device to run on?
# output:
#   imgs2    [nimg,chan,sY,sX]    batch of images AFTER transformation
#
#   Note: the format of each 'aff' matrix is
#            [m11, m12, b1, m21, m22, b2]
#               where
#             xout = m11*xpos + m12*ypos + b1
#             yout = m21*xpos + m22*ypos + b2
#-----------------------------------------------------------------
def torch_randaug_batch_affine(imgs, aff, fillval=0, device=None):
	nimg = imgs.shape[0]
	chan = imgs.shape[1]
	sY   = imgs.shape[2]
	sX   = imgs.shape[3]

	#--------------
	# calculate starting positions xpos ypos
	#--------------

	xpos = torch.arange(sX, device=device, requires_grad=False)
	xpos = xpos.type(torch.float32) + 0.5
	
	ypos = torch.arange(sY, device=device, requires_grad=False)
	ypos = ypos.type(torch.float32) + 0.5
	
	# upsample the positions to the full scale of the images
	xpos = torch.reshape(xpos, (1,1,1,sX))
	xpos = xpos.repeat((nimg,chan,sY,1))
	
	ypos = torch.reshape(ypos, (1,1,sY,1))
	ypos = ypos.repeat((nimg,chan,1,sX))
	
	#--------------
	# upsample affine hyperparameters
	#--------------

	# extract affine parameters
	m11 = aff[:,0]
	m12 = aff[:,1]
	b1  = aff[:,2]
	m21 = aff[:,3]
	m22 = aff[:,4]
	b2  = aff[:,5]
	
	# upsample the affine transformations to
	#  the full scale of the images
	m11 = torch.reshape(m11, (nimg,1,1,1))
	m12 = torch.reshape(m12, (nimg,1,1,1))
	b1  = torch.reshape(b1,  (nimg,1,1,1))
	m21 = torch.reshape(m21, (nimg,1,1,1))
	m22 = torch.reshape(m22, (nimg,1,1,1))
	b2  = torch.reshape(b2,  (nimg,1,1,1))
	
	m11 = m11.repeat((1,chan,sY,sX))
	m12 = m12.repeat((1,chan,sY,sX))
	b1  = b1.repeat ((1,chan,sY,sX))
	m21 = m21.repeat((1,chan,sY,sX))
	m22 = m22.repeat((1,chan,sY,sX))
	b2  = b2.repeat ((1,chan,sY,sX))
	
	#--------------
	# more bookkeeping,  image index, and channel index
	#--------------
	img_idx = torch.arange(nimg, device=device, requires_grad=False)
	chan_idx = torch.arange(chan, device=device, requires_grad=False)
	
	img_idx  = torch.reshape(img_idx,  (nimg,1,1,1))
	chan_idx = torch.reshape(chan_idx, (1,chan,1,1))
	
	img_idx  =  img_idx.repeat((1,chan,sY,sX))
	chan_idx = chan_idx.repeat((nimg,1,sY,sX))

	#--------------
	# calculate the output image coordinates
	#--------------
	xout = m11*xpos + m12*ypos + b1
	yout = m21*xpos + m22*ypos + b2

	#--------------
	# create a mask for safe assignment
	#--------------
	mask =                         (xout>=0.0)
	mask = torch.logical_and(mask, (xout<sX))
	mask = torch.logical_and(mask, (yout>=0.0))
	mask = torch.logical_and(mask, (yout<sY))
	mask = mask.type(torch.int32)
	
	inv_mask = 1 - mask
	
	#--------------
	# calculate the 1D addressing index
	#  NOTE: this is nearest neighbor interpolation
	#--------------
	
	# nearest neighbor filtering of xout and yout
	xout = torch.floor(xout)
	xout = xout.long()
	
	yout = torch.floor(yout)
	yout = yout.long()
	
	# addressing offsets
	y_off    = sX
	chan_off = y_off*sY
	img_off  = chan_off*chan
	nvals    = img_off * nimg

	# 1D indexing address
	address = img_idx*img_off + chan_idx*chan_off + yout*y_off + xout

	# apply masking for safe addressing
	address = mask*address

	#-------------
	# Apply the memory lookup
	#-------------
	
	# flatten
	flat_imgs = torch.reshape(imgs, (nvals,))
	address   = torch.reshape(address, (nvals,))
	
	# table lookup  (nearest neighbor)
	imgs2 = torch.index_select(flat_imgs,0,address)
	
	# unflatten
	imgs2 = torch.reshape(imgs2, (nimg,chan,sY,sX))

	# apply fillval
	imgs2 = imgs2*mask + fillval*inv_mask
	
	return imgs2


#-----------------------------------------------------------------
#   torch_randaug_reduce_histo
#      part of a parallel algorithm to calculate the
#      comulative density function (cdf)
#      from the probability density function (pdf)
#      required by EQUALIZATION
#
#   this functions takes as input
#      a high-resolution histogram (in_pdf)  with num_bin bins
#   and produces as output
#      a low-resolution histogram (out_pdf)  with num_bin // 2 bins
#      
#   input:
#       in_pdf    the input (high resolution) pdf
#                 [nimg, num_bin]
#       nimg      the number of single-channel images (or filters) in the batch
#       num_bin   the number of histogram bins of the pdf
#       device    which pytorch device to run on?
#   output:
#       out_pdf   the output (low resolution) pdf
#                 [nimg, num_bin//2]
#-----------------------------------------------------------------
def torch_randaug_reduce_histo(nimg,num_bin,in_pdf,device=None):
	out_bin = num_bin // 2
	out_size = nimg*out_bin
	in_size  = nimg*num_bin
	
	# calculate the indices of the input bins
	oB = torch.arange(out_bin, device=device, requires_grad=False)
	iB_L = 2*oB
	
	# upsample to our parallel format of
	#  [nimg, out_bin]
	iB_L = torch.reshape(iB_L, (1,out_bin)).repeat((nimg,1))
	
	# iN is the index of which 'image' the processor is working with
	iN = torch.arange(nimg, device=device, requires_grad=False)  # the level index
	iN = torch.reshape(iN, (nimg,1)).repeat((1,out_bin))

	#-------
	# Perform memory lookup for left and right bins
	#-------
	addr_L = iN * num_bin + iB_L
	addr_R = addr_L + 1
	
	# flatten everything
	addr_L = torch.reshape(addr_L, (out_size,))
	addr_R = torch.reshape(addr_R, (out_size,))
	in_pdf_flat = torch.reshape(in_pdf, (in_size,))
	
	# memory lookup
	val_L = torch.index_select(in_pdf_flat,0,addr_L)
	val_R = torch.index_select(in_pdf_flat,0,addr_R)
	
	# store result
	result = val_L + val_R
	result = torch.reshape(result, (nimg,out_bin))
	return result

#-----------------------------------------------------------------
#   torch_randaug_calc_cdf
#      part of a parallel algorithm to calculate the
#      comulative density function (cdf)
#      from the probability density function (pdf)
#      required by EQUALIZATION
#
#   this is needed to calculate the
#        summation term (cdf)
#        of the histogram (pdf)
#        in log_N parallel time
#
#   input:
#       nimg       the number of images
#       num_bin    the number of bins in this level of hierarchy
#       curr_pdf   the pdf that we are converting to a cdf
#                  [nimg, num_bin]
#       prev_cdf   the cdf of one level prior
#                  [nimg, num_bin//2]
#       device     which pytorch device to run on?
#   output:
#       curr_cdf   the cdf of the current level
#                  [nimg, num_bin]
#-----------------------------------------------------------------
def torch_randaug_calc_cdf(nimg, num_bin, curr_pdf, prev_cdf, device=None):
	prev_bin = num_bin // 2
	curr_size = nimg * num_bin
	prev_size = nimg * prev_bin
	
	# All parallel variables have size [nimg,num_bin]

	# iB is the index of which 'bin' the processor is working with
	iB = torch.arange(num_bin, device=device, requires_grad=False)  # the level index
	iB = torch.reshape(iB, (1,num_bin)).repeat((nimg,1))
	
	# iN is the index of which 'image' the processor is working with
	iN = torch.arange(nimg, device=device, requires_grad=False)  # the level index
	iN = torch.reshape(iN, (nimg,1)).repeat((1,num_bin))
		
	# is this an odd pixel ?
	is_odd = iB % 2
	
	#----
	# Perform memory lookup of the 'cdf value'
	#   result:  cdf_val
	#----

	# calculate the index of which value to use
	#  for the cdf lookup
	cdf_iB = iB // 2 - 1   # one to the left of the current value

	# one dimensional memory lookup
	cdf_address  =  iN * prev_bin  +  cdf_iB
	cdf_address  = torch.clamp(cdf_address,0,prev_size-1)  # restrict to memory bounds
	cdf_address  = torch.reshape(cdf_address, (curr_size,))
	prev_cdf_flat = torch.reshape(prev_cdf, (prev_size,))
	cdf_val = torch.index_select(prev_cdf_flat,0,cdf_address)
	cdf_val = torch.reshape(cdf_val, (nimg,num_bin))
	
	# Default to zero value, if memory lookup is unsafe
	cdf_is_nonzero = (cdf_iB >= 0).type(torch.int32)
	cdf_val *= cdf_is_nonzero
	
	#----
	# Possibly perform memory lookup for the
	#  'even' previous value
	#  (if working with an odd index)
	#----
	
	# calculatethe index of which value to use
	even_iB = iB - 1

	# one dimensional memory lookup
	even_address = iN * num_bin + even_iB
	even_address = torch.clamp(even_address, 0, curr_size-1)
	even_address = torch.reshape(even_address, (curr_size,))
	curr_pdf_flat = torch.reshape(curr_pdf, (curr_size,))
	even_val = torch.index_select(curr_pdf_flat,0,even_address)
	even_val = torch.reshape(even_val, (nimg,num_bin))
	
	# Default is zero value for 'even' indices
	#  we only add the even neighbor, if we are
	#  an odd pixel to begin with
	even_val *= is_odd
	
	#----
	# Add the results together
	#----
	curr_cdf = cdf_val + even_val + curr_pdf
	return curr_cdf

#-----------------------------------------------------------------
#  torch_randaug_cutout_abs
#    Performs a single cutout_abs operation
#
#   input tensors
#     imgs             batch of images             [nimg, chan, sY, sX]  float
#     v                randaugment parameter       [nimg]                float
#     x                center of cutout            [nimg]                float
#     y                center of cutout            [nimg]                float
#     fillval          fillval (scalar default:127)
#     device           which pytorch device to run on?
#   output tensor
#     out_imgs         augmented images
#-----------------------------------------------------------------

def torch_randaug_cutout_abs(imgs, v, x, y, fillval=127, device=None):
	
	# get the image dimensions
	nimg = imgs.shape[0]
	chan = imgs.shape[1]
	sY   = imgs.shape[2]
	sX   = imgs.shape[3]
	
	# get the boundaries of the cutout
	half_v = 0.5 * v
	x0 = torch.clamp(x - half_v, 0, sX-1)
	y0 = torch.clamp(y - half_v, 0, sY-1)
	x1 = x0 + v + 1
	y1 = y0 + v + 1
	
	# set the starting positions xpos ypos
	xpos = torch.arange(sX, device=device, requires_grad=False)
	xpos = xpos.type(torch.float32) + 0.5

	ypos = torch.arange(sY, device=device, requires_grad=False)
	ypos = ypos.type(torch.float32) + 0.5
	
	# upsample all of the SIMD variables
	xpos = torch.reshape(xpos, (1,1,1,sX)).repeat((nimg,chan,sY,1))
	ypos = torch.reshape(ypos, (1,1,sY,1)).repeat((nimg,chan,1,sX))
	x0   = torch.reshape(x0, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	x1   = torch.reshape(x1, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	y0   = torch.reshape(y0, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	y1   = torch.reshape(y1, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	
	# create a cutout mask
	mask = (xpos < x0)
	mask = torch.logical_or(mask, (xpos >= x1))
	mask = torch.logical_or(mask, (ypos < y0))
	mask = torch.logical_or(mask, (ypos >= y1))
	mask = mask.type(torch.int32)

	# run masking
	start_time = time.time()
	result_cutout = mask * imgs + (1-mask)*fillval
	return result_cutout
	
#-----------------------------------------------------------------
#  torch_randaug_run_ops  
#    Performs a single rand-aug operation
#     on a batch of images.
#
#    Note: every image may have a
#    different operation specified
#
#   input tensors
#     imgs             batch of images             [nimg, chan, sY, sX]  float
#     ops              list of operations          [nimg]                long
#     v                randaugment parameter       [nimg]                float
#     max_v            randaugment parameter       [nimg]                float
#     bias             randaugment parameter       [nimg]                float
#
#   optional input:
#     rand_rand  (optional) tensor of random numbers  [nimg]
#     rand_x     (optional) tensor of random x positions for cutout  [nimg]
#     rand_y     (optional) tensor of random y positions for cutout  [nimg]
#     device   (optional) which pytorch device to run on?
#
#   output tensor
#     out_imgs         augmented images            [nimg, chan, sY, sX]
#-----------------------------------------------------------------
g_torch_randaug_run_ops_padlayer    = None
g_torch_randaug_run_ops_smoothlayer = None

def torch_randaug_run_ops(imgs, ops, v, max_v, bias, rand_rand=None, rand_x=None, rand_y=None,device=None):
	
	#----
	# image batch dimensions
	#----
	nimg = imgs.shape[0]
	chan = imgs.shape[1]
	sY   = imgs.shape[2]
	sX   = imgs.shape[3]

	# float_parameter and int_parameter from randaugment [nimg]
	float_param   =  v * max_v / TRAUG_PARAMETER_MAX
	int_param     = (v * max_v / TRAUG_PARAMETER_MAX).type(torch.int32).type(torch.float32)

	# optional:
	#    draw random numbers if not
	#    externally supplied
	if rand_rand is None:
		rand_rand = torch.rand((nimg,),device=device)
	if rand_y is None:
		rand_y    = torch.randint(sX,(nimg,),device=device)
	if rand_x is None:
		rand_x    = torch.randint(sX,(nimg,),device=device)

	# logic for negation
	is_negate   = (rand_rand < 0.5)
	negate_sign = is_negate.type(torch.float32) * (-2.0) + 1.0  # value is -1.0 or 1.0

	# parameter values   f float  i int  u unsigned   s signed
	vfu = float_param + bias
	vfs = vfu * negate_sign
	viu = int_param + bias
	vis = viu * negate_sign
	v_trans_x = vfs * sX
	v_trans_y = vfs * sY

	#---
	# One hot encode the ops
	#---
	hops = F.one_hot(ops.type(torch.long), 17)
	
	#------------------------------------------------------
	# Is the operation a 'transform' ?
	#
	# Note: multiplication of one_hot is bitwise_and
	#------------------------------------------------------
	
	is_transform =                   \
		hops[:,TRAUG_IDENTITY] +     \
		hops[:,TRAUG_ROTATE] +       \
		hops[:,TRAUG_SHEAR_X] +      \
		hops[:,TRAUG_SHEAR_Y] +      \
		hops[:,TRAUG_TRANSLATE_X] +  \
		hops[:,TRAUG_TRANSLATE_Y]

	#---
	# If it is a 'transform' what is the transformation matrix ?
	#---
	# transform_identity = [1,0,0,0,1,0]
	transform_identity = torch.zeros((nimg,6),dtype=torch.float32,device=device)
	transform_identity[:,0] = 1
	transform_identity[:,4] = 1

	# clone the transformation matrices for the other operations
	transform_rotate      = torch.clone(transform_identity)
	transform_shear_x     = torch.clone(transform_identity)
	transform_shear_y     = torch.clone(transform_identity)
	transform_translate_x = torch.clone(transform_identity)
	transform_translate_y = torch.clone(transform_identity)

	# create the easy transformation matrices
#	transform_shear_x[:,1]     = vfs
	transform_shear_x[:,1]     = vfs
	transform_shear_y[:,3]     = vfs
	transform_translate_x[:,2] = v_trans_x - 0.5*negate_sign
	transform_translate_y[:,5] = v_trans_y - 0.5*negate_sign

	# now the hard one . . . rotation
	#  firstly, the rotation is around the center of the image
	#  secondly the y-axis is pointing down (goodbye trig.)
	#  this requires some graph paper . . .
	centerX = sX * 0.5
	centerY = sY * 0.5
	theta = -vis * 0.01745329251  # convert degrees to radians (negate for clockwise)
	sinT = torch.sin(theta)
	cosT = torch.cos(theta)
	transform_rotate[:,0] =  cosT # m11
	transform_rotate[:,1] =  sinT # m12
	transform_rotate[:,2] = centerX - centerX*cosT - centerY*sinT      # b1 
	transform_rotate[:,3] = -sinT # m21
	transform_rotate[:,4] =  cosT # m22
	transform_rotate[:,5] = centerY + centerX*sinT - centerY*cosT  # b2 
	
	# here's the cool part, we're going to parallel
	#  merge all of the transform matrices into a single
	#  matrix depending on the image
	hops_TRAUG_IDENTITY    = torch.reshape(hops[:,TRAUG_IDENTITY], (nimg,1)).repeat((1,6))
	hops_TRAUG_ROTATE      = torch.reshape(hops[:,TRAUG_ROTATE], (nimg,1)).repeat((1,6))
	hops_TRAUG_SHEAR_X     = torch.reshape(hops[:,TRAUG_SHEAR_X], (nimg,1)).repeat((1,6))
	hops_TRAUG_SHEAR_Y     = torch.reshape(hops[:,TRAUG_SHEAR_Y], (nimg,1)).repeat((1,6))
	hops_TRAUG_TRANSLATE_X = torch.reshape(hops[:,TRAUG_TRANSLATE_X], (nimg,1)).repeat((1,6))
	hops_TRAUG_TRANSLATE_Y = torch.reshape(hops[:,TRAUG_TRANSLATE_Y], (nimg,1)).repeat((1,6))

	transform = \
		hops_TRAUG_IDENTITY    * transform_identity    +   \
		hops_TRAUG_ROTATE      * transform_rotate      +   \
		hops_TRAUG_SHEAR_X     * transform_shear_x     +   \
		hops_TRAUG_SHEAR_Y     * transform_shear_y     +   \
		hops_TRAUG_TRANSLATE_X * transform_translate_x +   \
		hops_TRAUG_TRANSLATE_Y * transform_translate_y

	result_transform = torch_randaug_batch_affine(imgs, transform, device=device)

	is_transform = torch.reshape(is_transform, (nimg,1,1,1)).repeat((1,chan,sY,sX))

	#------------------------------------------------------
	# Is the operation an 'enhancement' ?
	#------------------------------------------------------
	
	is_enhance =                  \
		hops[:,TRAUG_BRIGHTNESS] +  \
		hops[:,TRAUG_COLOR]      +  \
		hops[:,TRAUG_CONTRAST]   +  \
		hops[:,TRAUG_SHARPNESS]
		

	# define gaussian smoothing for "sharpen"
	global g_torch_randaug_run_ops_padlayer
	global g_torch_randaug_run_ops_smoothlayer
	if g_torch_randaug_run_ops_smoothlayer is None:

		# create the zero-padding layer
		g_torch_randaug_run_ops_padlayer = nn.ZeroPad2d((1,1,1,1))

		# create the gaussian smoothing layer
		#gauss_kernel = [[[ \
		#	[1.0/13.0, 1.0/13.0, 1.0/13.0], \
		#	[1.0/13.0, 5.0/13.0, 1.0/13.0], \
		#	[1.0/13.0, 1.0/13.0, 1.0/13.0] ]]]
		g_torch_randaug_run_ops_smoothlayer = nn.Conv2d(1,1,(3,3),device=device)
		g_torch_randaug_run_ops_smoothlayer.weight.requires_grad = False
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,0,0] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,0,1] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,0,2] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,1,0] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,1,1] = 5.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,1,2] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,2,0] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,2,1] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.weight[0,0,2,2] = 1.0/13.0
		g_torch_randaug_run_ops_smoothlayer.bias.requires_grad = False
		g_torch_randaug_run_ops_smoothlayer.bias[0] = 0.0

	hops_TRAUG_COLOR     = torch.reshape(hops[:,TRAUG_COLOR],     (nimg,1,1,1)).repeat((1,chan,sY,sX))
	hops_TRAUG_CONTRAST  = torch.reshape(hops[:,TRAUG_CONTRAST],  (nimg,1,1,1)).repeat((1,chan,sY,sX))
	hops_TRAUG_SHARPNESS = torch.reshape(hops[:,TRAUG_SHARPNESS], (nimg,1,1,1)).repeat((1,chan,sY,sX))

	# 'grayscale' degenerate image for 'color-enhance'
	gray_imgs = torch.mean(imgs, dim=1)
	gray_imgs = torch.reshape(gray_imgs, (nimg,1,sY,sX))
	gray_imgs = gray_imgs.repeat((1,chan,1,1))
	
	# run convolution but replace boundary with original pixels
	smooth_imgs = torch.clone(imgs)
	smooth_imgs = torch.reshape(smooth_imgs, (nimg*chan,1,sY,sX))
	inner_smooth_imgs = g_torch_randaug_run_ops_smoothlayer(smooth_imgs)
	smooth_imgs[:,:,1:(sY-1),1:(sX-1)] = inner_smooth_imgs
	smooth_imgs = torch.reshape(smooth_imgs, (nimg,chan,sY,sX))

	# 'solid_gray' degenerate image for 'contrast'
	#solid_gray = torch.full((nimg,chan,sY,sX),127.0,dtype=torch.float32,device=device)
	solid_gray = torch.reshape(imgs,(nimg,chan*sY*sX))
	solid_gray = torch.mean(solid_gray,dim = 1)
	solid_gray = torch.reshape(solid_gray,(nimg,1,1,1))
	solid_gray = solid_gray.repeat((1,chan,sY,sX))
	
	

	degenerate = \
		hops_TRAUG_COLOR*gray_imgs + \
		hops_TRAUG_CONTRAST*solid_gray + \
		hops_TRAUG_SHARPNESS*smooth_imgs \
		# hops_TRAUG_BRIGHTNESS*0.0          (not needed because multiply by zero)

	big_vfu = torch.reshape(vfu, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	

	result_enhance = (1.0-big_vfu)*degenerate + big_vfu*imgs
	result_enhance = torch.clamp(result_enhance, 0.0, 255.0)
		
	is_enhance = torch.reshape(is_enhance, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	
	#end_time = time.time()
	#execution_time = end_time - start_time
	#print(f"Execution time for Our_Sharp is : {execution_time:.16f} seconds")
	
	#------------------------
	# Solarize transform
	#------------------------
	
	# is this INVERT, SOLARIZE, or SOLARIZE_ADD transform ?
	hops_TRAUG_INVERT       = hops[:,TRAUG_INVERT]
	hops_TRAUG_SOLARIZE     = hops[:,TRAUG_SOLARIZE]
	hops_TRAUG_SOLARIZE_ADD = hops[:,TRAUG_SOLARIZE_ADD]
	
	#start_time = time.time()
	is_solarize = \
		hops_TRAUG_INVERT   + \
		hops_TRAUG_SOLARIZE + \
		hops_TRAUG_SOLARIZE_ADD

	

	# convert into a single solarize_add 
	#  statment with offset and threshold
	solarize_offset    = hops_TRAUG_SOLARIZE_ADD * vis
	solarize_threshold = \
		hops_TRAUG_INVERT       * (-1.0)       + \
		hops_TRAUG_SOLARIZE     * (256.0-viu)  + \
		hops_TRAUG_SOLARIZE_ADD * 128.0

	# upsample to image dimensions
	solarize_offset    = torch.reshape(solarize_offset,    (nimg,1,1,1)).repeat((1,chan,sY,sX))
	solarize_threshold = torch.reshape(solarize_threshold, (nimg,1,1,1)).repeat((1,chan,sY,sX))
	is_solarize        = torch.reshape(is_solarize,        (nimg,1,1,1)).repeat((1,chan,sY,sX))

	# run solarize
	imgs_add    = imgs + solarize_offset
	imgs_add    = torch.clamp(imgs_add, 0, 255.0)
	imgs_invert = 255.0 - imgs_add
	mask = (imgs_add >= solarize_threshold).type(torch.float32)
	result_solarize = imgs_add*(1.0-mask) + imgs_invert*mask
	
	#end_time = time.time()
	#execution_time = end_time - start_time
	#print(f"Execution time for Our_Solarize is : {execution_time:.16f} seconds")
	
	#------------------------
	# Posterize transform
	#------------------------
	#start_time = time.time()
	is_posterize = torch.reshape(hops[:,TRAUG_POSTERIZE], (nimg,1,1,1)).repeat((1,chan,sY,sX))
	
	result_posterize = imgs.type(torch.uint8)
	bits_keep = viu.type(torch.int32)
	bits_discard = 8-bits_keep
	bits_discard = torch.reshape(bits_discard,(nimg,1,1,1)).repeat((1,chan,sY,sX))

	#input("enter")
	result_posterize = torch.bitwise_right_shift(result_posterize, bits_discard)
	result_posterize = torch.bitwise_left_shift(result_posterize, bits_discard)
	result_posterize = result_posterize.type(torch.float32)


	#------------------------
	# AutoContrast
	#------------------------
	
	nfilters = nimg*chan
	sYX      = sY*sX
	
	is_autocontrast = torch.reshape(hops[:,TRAUG_AUTO_CONTRAST], (nimg,1,1,1)).repeat((1,chan,sY,sX))
	imgs_ac = torch.reshape(imgs, (nfilters,sYX))
	
	hi_val    = torch.amax(imgs_ac, dim=1, keepdim=True)
	lo_val    = torch.amin(imgs_ac, dim=1, keepdim=True)

	scale_val  = 1.0 / (hi_val - lo_val + 0.00001)
	scale_val  = scale_val.repeat((1,sYX))
	lo_val_rep = lo_val.repeat((1,sYX))
	result_autocontrast = 255.0 * (imgs_ac - lo_val_rep) * scale_val
	result_autocontrast = result_autocontrast.reshape((nimg,chan,sY,sX))
	
	#------------------------
	# Histogram equalization
	#------------------------
	start_time = time.time()
	
	is_equalize = torch.reshape(hops[:,TRAUG_EQUALIZE], (nimg,1,1,1)).repeat((1,chan,sY,sX))
	
	# calculate the histogram  (probability density)
	#   result: pdf_u8
	pdf_u8 = imgs.type(torch.uint8)
	pdf_u8 = torch.reshape(pdf_u8, (nfilters,sYX))
	pdf_u8 = F.one_hot(pdf_u8.type(torch.long), 256)
	pdf_u8 = torch.sum(pdf_u8, dim=1)
	
	# parallel sum the histogram  (cumulative distribution)
	#   result: cdf_u8
	pdf_u7 = torch_randaug_reduce_histo(nfilters,256,pdf_u8,device=device)
	pdf_u6 = torch_randaug_reduce_histo(nfilters,128,pdf_u7,device=device)
	pdf_u5 = torch_randaug_reduce_histo(nfilters,64,pdf_u6,device=device)
	pdf_u4 = torch_randaug_reduce_histo(nfilters,32,pdf_u5,device=device)
	pdf_u3 = torch_randaug_reduce_histo(nfilters,16,pdf_u4,device=device)
	pdf_u2 = torch_randaug_reduce_histo(nfilters,8,pdf_u3,device=device)
	pdf_u1 = torch_randaug_reduce_histo(nfilters,4,pdf_u2,device=device)
	cdf_u0 = torch_randaug_reduce_histo(nfilters,2,pdf_u1,device=device)
	cdf_u1 = torch_randaug_calc_cdf(nfilters, 2, pdf_u1, cdf_u0,device=device)
	cdf_u2 = torch_randaug_calc_cdf(nfilters, 4, pdf_u2, cdf_u1,device=device)
	cdf_u3 = torch_randaug_calc_cdf(nfilters, 8, pdf_u3, cdf_u2,device=device)
	cdf_u4 = torch_randaug_calc_cdf(nfilters, 16, pdf_u4, cdf_u3,device=device)
	cdf_u5 = torch_randaug_calc_cdf(nfilters, 32, pdf_u5, cdf_u4,device=device)
	cdf_u6 = torch_randaug_calc_cdf(nfilters, 64, pdf_u6, cdf_u5,device=device)
	cdf_u7 = torch_randaug_calc_cdf(nfilters, 128, pdf_u7, cdf_u6,device=device)
	cdf_u8 = torch_randaug_calc_cdf(nfilters, 256, pdf_u8, cdf_u7,device=device)
	
	#-----------------------------
	#print('bin\tpdf\tcdf')
	
	pdf_u8_np = pdf_u8.detach().cpu().numpy()
	cdf_u8_np = cdf_u8.detach().cpu().numpy()
	
	#for i in range(pdf_u8_np.shape[1]):
		#print(i,'\t', pdf_u8_np[0,i], '\t', cdf_u8_np[0,i])
	
	#-----------------------------
	# find the value x0 of cdf_u8 of the smallest non-empty bin
	
	lo_val = torch.reshape(lo_val,(nfilters,))
	iFilt = torch.arange(nfilters, device=device, requires_grad=False)
	addr = iFilt.to(torch.int32)*256 + lo_val.to(torch.int32)
	flat_pdf_u8 = torch.reshape(pdf_u8,(nfilters*256,))
	x0 = torch.index_select(flat_pdf_u8,0,addr)
	x1 = sYX
	
	
	# solve for m and b for colorscale interpolation
	
	m = 255/(x1-x0+0.000001)
	m = torch.reshape(m, (nfilters,1)).repeat((1,256))
	x0 = torch.reshape(x0, (nfilters,1)).repeat((1,256))
	b = -m*x0

	
	
	# scale the cdf into a "colorscale" 0.0 to 255.0
	#colorscale = cdf_u8 * (255.0/sYX)
	 
	colorscale = m * cdf_u8 + b
	
	
	#colorscale_np = colorscale.detach().cpu().numpy()
	
	#print('bin\tcolorscale')
	#for i in range(colorscale_np.shape[1]):
	#	print(i,'\t', colorscale_np[0,i])
	
	
	colorscale = torch.reshape(colorscale, (nfilters*256,))
	

	# perform the memory lookup
	iFilt = torch.reshape(iFilt, (nfilters,1)).repeat((1,sYX))
	iFilt = torch.reshape(iFilt, (nfilters*sYX,))
	imgs_flat  = torch.reshape(imgs, (nimg*chan*sY*sX,)).type(torch.int32)
	color_addr = iFilt*256 + imgs_flat	
	
	result_equalize = torch.index_select(colorscale,0,color_addr)
	
	# print lo/hi values here
	#print('result_equalize.shape is : ', result_equalize.shape)
	
	#result_equalize_c = torch.reshape(result_equalize,(nfilters,sYX))
	#hi_val = torch.amax(result_equalize_c, dim=1 , keepdim=True) 
	#lo_val = torch.amin(result_equalize_c, dim=1 , keepdim=True)
	
	result_equalize = torch.reshape(result_equalize, (nimg,chan,sY,sX))
	result_equalize = torch.clamp(result_equalize, 0, 255)
	
	end_time = time.time()
	exe_time = end_time - start_time
	print(f"Execution time for Our_Equalize: {exe_time:.16f} seconds")
	#------------------------
	# AutoContrast
	#------------------------
	
	#is_autocontrast = torch.reshape(hops[:,TRAUG_AUTO_CONTRAST], (nimg,1,1,1)).repeat((1,chan,sY,sX))
	#imgs_ac = torch.reshape(imgs, (nfilters,sYX))
	
	#hi_val    = torch.amax(imgs_ac, dim=1, keepdim=True)
	#lo_val    = torch.amin(imgs_ac, dim=1, keepdim=True)

	#scale_val = 1.0 / (hi_val - lo_val + 0.00001)
	#scale_val = scale_val.repeat((1,sYX))
	#lo_val    = lo_val.repeat((1,sYX))
	
	#result_autocontrast = 255.0 * (imgs_ac - lo_val) * scale_val
	#result_autocontrast = result_autocontrast.reshape((nimg,chan,sY,sX))

	#------------------------
	# Cutout
	#------------------------
	
	
	is_cutout = torch.reshape(hops[:,TRAUG_CUTOUT], (nimg,1,1,1)).repeat((1,chan,sY,sX))
	result_cutout = torch_randaug_cutout_abs(imgs, (vfu*min(sY,sX)).type(torch.int32), rand_x, rand_y,device=device)
	
	#------------------------
	# Return Results
	#------------------------
	result = \
		result_transform    * is_transform + \
		result_enhance      * is_enhance + \
		result_solarize     * is_solarize + \
		result_posterize    * is_posterize + \
		result_equalize     * is_equalize + \
		result_autocontrast * is_autocontrast + \
		result_cutout       * is_cutout
	
	    # Transformations to apply

	
	
	return result

#-----------------------------------------------------------------
#
# NOT  The augmentation parameters used by FixMatch
#
#  Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang,
#  H., Raffel, C. A., ... & Li, C. L. (2020).
#  Fixmatch: Simplifying semi-supervised learning with consistency and confidence.
#  Advances in neural information processing systems, 33, 596-608.
#
#-----------------------------------------------------------------
'''
my_augment_pool = (                    \
	(TRAUG_AUTO_CONTRAST, -9999.0, -9999.0),      \
	(TRAUG_BRIGHTNESS, 1.8, 0.1),                 \
	(TRAUG_COLOR, 1.8, 0.1),                     \
	(TRAUG_CONTRAST, 1.8, 0.1),                  \
	(TRAUG_CUTOUT, 0.2, 0),                       \
	(TRAUG_EQUALIZE, -9999.0, -9999.0),          \
	(TRAUG_INVERT, -9999.0, -9999.0),             \
	(TRAUG_POSTERIZE, 4, 4),                      \
	(TRAUG_ROTATE, 30, 0),                        \
	(TRAUG_SHARPNESS, 1.8, 0.1),                 \
	(TRAUG_SHEAR_X, 0.3, 0),                      \
	(TRAUG_SHEAR_Y, 0.3, 0),                      \
	(TRAUG_SOLARIZE, 256, 0),                     \
	(TRAUG_SOLARIZE_ADD, 110, 0),                 \
	(TRAUG_TRANSLATE_X, 0.45, 0),                 \
	(TRAUG_TRANSLATE_Y, 0.45, 0))

'''
my_augment_pool = ((TRAUG_EQUALIZE, -9999.0, -9999.0),)

#-----------------------------------------------------------------
#-----------------------------------------------------------------
# class TorchRandAugment
#
#   Easy to use class to run the
#    torch-accelerated rand-augmentation
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class TorchRandAugment(object):

	#-----------------------------------------------------------------
	#
	#   TorchRandAugment.__init__
	#
	#     aug_pool          default:fixmatch pool
	#         the pool of augmentation choices
	#
	#     n                 default:1
	#         the number of times to apply augmentation per image
	#         assert:  n>=1
	#
	#     m                 default:TRAUG_PARAMETER_MAX(10)
	#         the number of discreet steps for augmentation
	#         assert: (1 <= m <= 10)
	#
	#     final_cutout      default:32*0.5
	#         how large to apply a final cutout to the image
	#
	#         note:  if    final_cutout is None  or   final_cutout<=0
	#               then   no final cutout is performed
	#
	#     clear_chance      default:0.5
	#         what is the chance that any given operation
	#         will be zeroed out by "identity"
	#
	#     store_choices     default:False
	#         do you want to store the random numbers generated ?
	#         this is useful for experimental comparisons with
	#         original randaugment, but typically is not needed
	#
	#     device            default:None
	#         which pytorch device to run on?
	#-----------------------------------------------------------------
	def __init__(self, aug_pool=my_augment_pool, n=1, m=TRAUG_PARAMETER_MAX, final_cutout=32*0.5, clear_chance=0.5, store_choices=False, device=None):
		assert n >= 1
		assert 1 <= m <= 10
		self.n = n
		self.m = m
		self.aug_pool = torch.tensor(aug_pool,requires_grad=False,device=device)
		self.final_cutout = final_cutout
		self.clear_chance = clear_chance
		self.store_choices = store_choices
		self.device = device

	#-----------------------------------------------------------------
	#   TorchRandAugment.__call__
	#
	#     input tensor:
	#          imgs  [nimg,chan,sY,sX]   minibatch of images BEFORE augmentation
	#     output tensor:
	#          imgs  [nimg,chan,sY,sX]   minibatch of images AFTER augmentation
	#-----------------------------------------------------------------
	def __call__(self, imgs):
	
		nimg = imgs.shape[0]
		chan = imgs.shape[1]
		sY   = imgs.shape[2]
		sX   = imgs.shape[3]
		naug = self.aug_pool.shape[0]
		aug_pool_flat = torch.reshape(self.aug_pool, (naug*3,))

		

		if (self.store_choices):
			self.ops       = []
			self.max_v     = []
			self.bias      = []
			self.rand_op   = []
			self.rand_v    = []
			self.rand_rand = []
			self.rand_x    = []
			self.rand_y    = []

		# For every augmentation pass
		for i_n in range(self.n):

		   # start_time = time.time()
		   #for i_op in range(naug):
		    
			# Draw random parameters
			rand_op     = torch.randint(naug,(nimg,),device=self.device)
			rand_v      = torch.randint(self.m,(nimg,),device=self.device)
			rand_rand   = torch.rand((nimg,),device=self.device)
			rand_y      = torch.randint(sX,(nimg,),device=self.device)
			rand_x      = torch.randint(sX,(nimg,),device=self.device)

			# Extract the operation parameters
			ops_idx   = torch.reshape(rand_op, (nimg,1)).repeat((1,3))
			param_idx = torch.arange(3, device=self.device, requires_grad=False).type(torch.int32)
			param_idx = torch.reshape(param_idx, (1,3)).repeat((nimg,1))
			ops_addr  = ops_idx * 3 + param_idx
			
			ops       = traug_index_select(self.aug_pool,ops_addr)
			

			# Do we clear it out ?  (default 50% chance)
			mask_clear  = torch.rand((nimg,),device=self.device)
#			mask_clear  = mask_clear[mask_clear < self.clear_chance].type(torch.int32)
			mask_clear  = (mask_clear < self.clear_chance).type(torch.int32)
			mask_clear  = torch.reshape(mask_clear, (nimg,1)).repeat((1,3))
			
			#ops = ops * mask_clear
			max_v = ops[:,1]
			bias  = ops[:,2]
			ops   = ops[:,0]

			if (self.store_choices):
				self.ops       . append(ops)
				self.max_v     . append(max_v)
				self.bias      . append(bias)
				self.rand_op   . append(rand_op)
				self.rand_v    . append(rand_v)
				self.rand_rand . append(rand_rand)
				self.rand_x    . append(rand_y)
				self.rand_y    . append(rand_x)

			# Run the operation
			start_op_time = time.time()
			imgs = torch_randaug_run_ops(imgs, ops, rand_v, max_v, bias, rand_rand=rand_rand, rand_x=rand_x, rand_y=rand_y, device=self.device)
			end_op_time = time.time()
			op_time =  end_op_time - start_op_time
			
			print(f"Operation execution time: {op_time:.16f} seconds")
			
			

		# Final cutout pass
		if (self.final_cutout is not None and self.final_cutout > 0.0):

			# Where to make the final cutout ?
			rand_y      = torch.randint(sX,(nimg,),device=self.device)
			rand_x      = torch.randint(sX,(nimg,),device=self.device)
			
			# Store the choices ?
			if (self.store_choices):
				self.final_rand_x = rand_x
				self.final_rand_y = rand_y

			# Run the cutout
			imgs = torch_randaug_cutout_abs(imgs, self.final_cutout, rand_x, rand_y, fillval=127, device=self.device)

		# Done running
		return imgs




