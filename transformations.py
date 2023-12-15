#!/usr/bin/python

import numpy as np
from sklearn.externals._pilutil import bytescale
import random
import cv2
import random
import torch
import torchvision.transforms as transforms


def create_dense_target(tar: np.ndarray):
    dummy = tar[:,:]
    return dummy

def normalize(inp: np.ndarray, mean: float, std: float):
    inp_out = (inp - mean) / std
    return inp_out

def normalize_01(inp: np.ndarray):
    inp_out = inp/255
    return inp_out

def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])


class MoveAxis:
    def __init__(self, transform_input: bool = True, transform_target: bool = False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp = np.moveaxis(inp, -1, 0)        

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class DenseTarget:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        tar = create_dense_target(tar)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 1)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.fliplr(tar))

        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.moveaxis(inp, 0, -1)
            inp = cv2.flip(inp, 0)
            inp = np.moveaxis(inp, -1, 0)
            tar = np.ndarray.copy(np.flipud(tar))

        rand = random.choice([0, 1])
        if rand == 1:
            inp = np.ndarray.copy(np.rot90(inp, k=1, axes=(1, 2)))
            tar = np.ndarray.copy(np.rot90(tar, k=1, axes=(0, 1)))
        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})
    
class ColorTransformations:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp_tensor = torch.from_numpy(inp)
        tar_tensor = torch.from_numpy(tar)

        color_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])

        inp_tensor = color_transform(inp_tensor)

        inp = inp_tensor.numpy()
        tar = tar_tensor.numpy()

        return inp, tar

class ColorNoise:
    def __init__(self, noise_std=0.05):
        self.noise_std = noise_std

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp_tensor = torch.from_numpy(inp)
        tar_tensor = torch.from_numpy(tar)

        noise = torch.randn_like(inp_tensor) * self.noise_std
        inp_tensor += noise

        inp_tensor = torch.clamp(inp_tensor, 0, 1)

        inp = inp_tensor.numpy()
        tar = tar_tensor.numpy()

        return inp, tar

class RandomCrop_John:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):

        crop_width = 1900
        crop_height = 1900

        max_x = inp.shape[1] - crop_width
        max_y = inp.shape[2] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        inp = np.moveaxis(inp, 0, -1)
        inp = inp[x: x + crop_width, y: y + crop_height,:]
        inp = cv2.resize(inp, (512,512), interpolation = cv2.INTER_NEAREST)
        inp = np.moveaxis(inp, -1, 0)

        tar = tar[x: x + crop_width, y: y + crop_height]
        tar = cv2.resize(tar, (512,512), interpolation = cv2.INTER_NEAREST)

        return inp, tar

class RandomCrop_USA:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        crop_width = 512
        crop_height =512

        max_x = inp.shape[1] - crop_width
        max_y = inp.shape[2] - crop_height

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        inp = np.moveaxis(inp, 0, -1)
        inp = inp[x: x + crop_width, y: y + crop_height,:]
        inp = np.moveaxis(inp, -1, 0)

        tar = tar[x: x + crop_width, y: y + crop_height]

        return inp, tar


class Resize_Sample:
    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        inp = np.moveaxis(inp, 0, -1)
        inp = cv2.resize(inp, (256,256), interpolation = cv2.INTER_NEAREST)
        inp = np.moveaxis(inp, -1, 0)
        tar = cv2.resize(tar, (256,256), interpolation = cv2.INTER_NEAREST)

        return inp, tar

class Normalize01:
    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01(inp)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize:
    def __init__(self,
                 mean: float,
                 std: float,
                 transform_input=True,
                 transform_target=False
                 ): 

        self.transform_input = transform_input
        self.transform_target = transform_target
        self.mean = mean
        self.std = std

    def __call__(self, inp, tar):
        inp = normalize(inp)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})
