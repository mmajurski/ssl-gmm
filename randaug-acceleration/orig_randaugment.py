#
# This is modified to allow us to pass in a separate
#  seed from the torch randaugment
#


# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import time

import torch_randaugment as tra

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
	
	return PIL.ImageOps.autocontrast(img)
	

def Brightness(img, v, max_v, bias=0):
	v = _float_parameter(v, max_v) + bias
	return PIL.ImageEnhance.Brightness(img).enhance(v)
	

def Color(img, v, max_v, bias=0):
	v = _float_parameter(v, max_v) + bias
	return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
	v = _float_parameter(v, max_v) + bias
	return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0, rand_x=None, rand_y=None):
	if v == 0:
		return img
	v = _float_parameter(v, max_v) + bias
	v = int(v * min(img.size))
	return CutoutAbs(img, v, rand_x, rand_y)


def CutoutAbs(img, v, rand_x=None, rand_y=None):

	print("img.size")
	print(img.size)
	w, h = img.size

	if rand_x is None:
		x0 = np.random.uniform(0, w)
	else:
		x0 = rand_x

	if rand_y is None:
		y0 = np.random.uniform(0, h)
	else:
		y0 = rand_y

	x0 = int(max(0, x0 - v / 2.))
	y0 = int(max(0, y0 - v / 2.))
	x1 = int(min(w, x0 + v))
	y1 = int(min(h, y0 + v))
	xy = (x0, y0, x1, y1)
	# gray
	color = (127, 127, 127)
	img = img.copy()
	PIL.ImageDraw.Draw(img).rectangle(xy, color)
	return img


def Equalize(img):
	return PIL.ImageOps.equalize(img)
	

def Identity(img):
	return img


def Invert(img):
	return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
	v = _int_parameter(v, max_v) + int(bias)
	return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0, rand_rand=None):
	v = _int_parameter(v, max_v) + bias
	if rand_rand is None:
		rand_rand = random.random()
	if rand_rand < 0.5:
		v = -v
	return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
	v = _float_parameter(v, max_v) + bias
	return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0, rand_rand=None):
	v = _float_parameter(v, max_v) + bias
	if rand_rand is None:
		rand_rand = random.random()
	if rand_rand < 0.5:
		v = -v
	return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0, rand_rand=None):
	v = _float_parameter(v, max_v) + bias
	if rand_rand is None:
		rand_rand = random.random()
	if rand_rand < 0.5:
		v = -v
	return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
	print('Solarize  v', v, 'max_v', max_v, 'bias', bias)
	v = _int_parameter(v, max_v) + int(bias)
	print('v')
	return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128, rand_rand=None):
	v = _int_parameter(v, max_v) + int(bias)
	if rand_rand is None:
		rand_rand = random.random()
	if rand_rand < 0.5:
		v = -v
	#print('v', v)
	
	img_np = np.array(img).astype(np.int)
	img_np = img_np + v
	img_np = np.clip(img_np, 0, 255)
	img_np = img_np.astype(np.uint8)
	img = Image.fromarray(img_np)
	return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0, rand_rand=None):
	v = _float_parameter(v, max_v) + bias
	if rand_rand is None:
		rand_rand = random.random()
	if rand_rand < 0.5:
		v = -v
	v = int(v * img.size[0])
	return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0, rand_rand=None):
	v = _float_parameter(v, max_v) + bias
	if rand_rand is None:
		rand_rand = random.random()
	if rand_rand < 0.5:
		v = -v
	v = int(v * img.size[1])
	return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
	return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
	return int(v * max_v / PARAMETER_MAX)


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for Original_{func.__name__}: {execution_time:.16f} seconds")
        return result
    return wrapper
    
    
class OrigRandAugmentTester():
	def __init__(self, traug):
		self.traug = traug
	
	def __call__(self, imgs):
		
		# convert tensor to numpy
		imgs = imgs.type(torch.uint8).detach().cpu().numpy()
		
		# extract dimensions
		nimg = imgs.shape[0]
		chan = imgs.shape[1]
		sY   = imgs.shape[2]
		sX   = imgs.shape[3]
		
		# For every augmentation pass
		for i in range(self.traug.n):

			# extract the augmentation parameters
			ops       = self.traug.ops      [i].detach().cpu().numpy()
			max_v     = self.traug.max_v    [i].detach().cpu().numpy()
			bias      = self.traug.bias     [i].detach().cpu().numpy()
			rand_op   = self.traug.rand_op  [i].detach().cpu().numpy()
			rand_v    = self.traug.rand_v   [i].detach().cpu().numpy()
			rand_rand = self.traug.rand_rand[i].detach().cpu().numpy()
			rand_x    = self.traug.rand_y   [i].detach().cpu().numpy()
			rand_y    = self.traug.rand_x   [i].detach().cpu().numpy()
			
			for j in range(nimg):

				# convert from tensor to RGB format
				img_rgb = np.transpose(imgs[j], (1,2,0))
				PIL_img = Image.fromarray(np.uint8(img_rgb)).convert('RGB')
				

				# Run the operation
				
				if ops[j] == tra.TRAUG_IDENTITY:
					PIL_img = Identity(PIL_img)
						
				elif ops[j] == tra.TRAUG_AUTO_CONTRAST:
					PIL_img = AutoContrast(PIL_img)
						
				elif ops[j] == tra.TRAUG_BRIGHTNESS:
					PIL_img = Brightness(PIL_img, rand_v[j], max_v[j], bias[j])
						
				elif ops[j] == tra.TRAUG_COLOR:
					PIL_img = Color(PIL_img, rand_v[j], max_v[j], bias[j])
						
				elif ops[j] == tra.TRAUG_CONTRAST:
					PIL_img = Contrast(PIL_img, rand_v[j], max_v[j], bias[j])
						
				elif ops[j] == tra.TRAUG_EQUALIZE:
					PIL_img = Equalize(PIL_img)
					
						
				elif ops[j] == tra.TRAUG_INVERT:
					PIL_img = Invert(PIL_img)
						
				elif ops[j] == tra.TRAUG_POSTERIZE:
					PIL_img = Posterize(PIL_img, rand_v[j], max_v[j], bias[j])
						
				elif ops[j] == tra.TRAUG_ROTATE:
					PIL_img = Rotate(PIL_img, rand_v[j], max_v[j], bias[j], rand_rand[j])
						
				elif ops[j] == tra.TRAUG_SHARPNESS:
					PIL_img = Sharpness(PIL_img, rand_v[j], max_v[j], bias[j])
						
				elif ops[j] == tra.TRAUG_SHEAR_X:
					PIL_img = ShearX(PIL_img, rand_v[j], max_v[j], bias[j], rand_rand[j])
						
				elif ops[j] == tra.TRAUG_SHEAR_Y:
					PIL_img = ShearY(PIL_img, rand_v[j], max_v[j], bias[j], rand_rand[j])
						
				elif ops[j] == tra.TRAUG_SOLARIZE:
					PIL_img = Solarize(PIL_img, rand_v[j], max_v[j], bias[j])
						
				elif ops[j] == tra.TRAUG_SOLARIZE_ADD:
					PIL_img = SolarizeAdd(PIL_img, rand_v[j], max_v[j], bias[j], 128, rand_rand[j])
						
				elif ops[j] == tra.TRAUG_TRANSLATE_X:
					PIL_img = TranslateX(PIL_img, rand_v[j], max_v[j], bias[j], rand_rand[j])
						
				elif ops[j] == tra.TRAUG_TRANSLATE_Y:
					PIL_img = TranslateY(PIL_img, rand_v[j], max_v[j], bias[j], rand_rand[j])
						
				elif ops[j] == tra.TRAUG_CUTOUT:
					PIL_img = Cutout(PIL_img, rand_v[j], max_v[j], bias[j], rand_x[j], rand_y[j])
					
				# Convert Pillow back to numpy
				img_rgb = np.array(PIL_img.getdata()).reshape(PIL_img.size[0], PIL_img.size[1], 3)
				imgs[j] = np.transpose(img_rgb, (2,0,1))
				
		# Final cutout pass
		if (self.traug.final_cutout is not None and self.traug.final_cutout > 0.0):
		
			# Where to make the final cutout ?
			rand_y = self.traug.final_rand_y.detach().cpu().numpy()
			rand_x = self.traug.final_rand_x.detach().cpu().numpy()

			# For every image
			for j in range(nimg):

				# convert from tensor to RGB format
				img_rgb = np.transpose(imgs[j], (1,2,0))
				PIL_img = Image.fromarray(np.uint8(img_rgb)).convert('RGB')
			
				# Run the cutout
				PIL_img = CutoutAbs(PIL_img, self.traug.final_cutout, rand_x[j], rand_y[j])

				# Convert Pillow back to numpy
				img_rgb = np.array(PIL_img.getdata()).reshape(PIL_img.size[0], PIL_img.size[1], 3)
				imgs[j] = np.transpose(img_rgb, (2,0,1))
		
		
		
		# Convert back into tesor
		imgs = torch.tensor(imgs, device=self.traug.device, requires_grad=False)
		
		# Done running
		return imgs

Equalize = measure_execution_time(Equalize)
'''	
# Apply the decorator to each augmentation function

AutoContrast = measure_execution_time(AutoContrast)
Brightness = measure_execution_time(Brightness)
Color = measure_execution_time(Color)
Contrast = measure_execution_time(Contrast)		
Cutout = measure_execution_time(Cutout)
CutoutAbs = measure_execution_time(CutoutAbs)

Identity = measure_execution_time(Identity)
Invert = measure_execution_time(Invert)
Posterize = measure_execution_time(Posterize)
Rotate = measure_execution_time(Rotate)
Sharpness = measure_execution_time(Sharpness)
ShearX = measure_execution_time(ShearX)
ShearY = measure_execution_time(ShearY)
Solarize = measure_execution_time(Solarize)
SolarizeAdd = measure_execution_time(SolarizeAdd)
TranslateX = measure_execution_time(TranslateX)
TranslateY = measure_execution_time(TranslateY)	
	'''
