# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_pool():
    augs = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return augs




class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val)

        # for op, max_v, bias in ops:
        #     v = np.random.randint(1, self.m)
        #     if random.random() < 0.5:
        #         img = op(img, v=v, max_v=max_v, bias=bias)
        # img = CutoutAbs(img, int(32*0.5))
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)
        return img


# from https://github.com/kekmodel/FixMatch-pytorch
class TransformFixMatch(object):
    WEAK = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])

    STRONG = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            # torchvision.transforms.RandAugment(num_ops=2, magnitude=10)])
            RandAugmentMC(n=2, m=10)])
    def __init__(self, mean, std):
        self.weak = self.WEAK
        self.strong = self.STRONG
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatchSTL(object):
    crop_ratio = 0.875
    img_size = 96

    WEAK = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.RandomCrop(size=img_size,
                                  padding=int(img_size * (1 - crop_ratio)),
                                  padding_mode='reflect')])

    STRONG = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=img_size,
                                  padding=int(img_size * (1 - crop_ratio)),
                                  padding_mode='reflect'),
            RandAugmentMC(n=3, m=5)])
    def __init__(self, mean, std):
        self.weak = self.WEAK
        self.strong = self.STRONG
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

