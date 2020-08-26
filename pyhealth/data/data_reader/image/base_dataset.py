# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch


class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

def get_params(params, size):
    
    new_h = new_w = params['load_size']
    x = random.randint(0, np.maximum(0, new_w - params['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - params['crop_size']))
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
    
def get_transform(params, trans_params=None, grayscale=False, method=Image.BICUBIC, run_type = 'train'):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if run_type == 'train':
        osize = [params['load_size'], params['load_size']]
        transform_list.append(transforms.Resize(osize, method))

        if trans_params is None:
            transform_list.append(transforms.RandomCrop(params['crop_size']))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, trans_params['crop_pos'], params['crop_size'])))

        transform_list.append(transforms.Lambda(lambda img: __flip(img, trans_params['flip'])))

        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
        osize = [params['load_size'], params['load_size']]
        transform_list.append(transforms.Resize(osize, method))
        transform_list.append(transforms.Lambda(lambda img: __crop(img, ((params['load_size']-params['crop_size'])/2,\
                                                                         (params['load_size']-params['crop_size'])/2), \
                                                                           params['crop_size'])))
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
