# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause


import torch
import numpy as np
from PIL import Image
from .base_dataset import get_params, get_transform, BaseDataset

class DatasetReader(BaseDataset):

    def __init__(self, data, params, run_type = 'train', data_type = 'distribute'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, params)
        self.data_type = data_type
        self.feat_info = data['x']
        self.label_list = data['y']
        self.run_type = run_type
        self.params = params
        
    def __getitem__(self, index):
        if self.data_type == 'distribute':
            xray_path = self.feat_info[index]
            xray = Image.open(xray_path).convert('L')
        else:
            xray = self.feat_info[index]
        label = self.label_list[index]
        transform_params = get_params(self.params, xray.size)
        xray_transform = get_transform(self.params, transform_params, grayscale=True, run_type = self.run_type)
        xray = xray_transform(xray)
        xray = torch.cat((xray, xray, xray), 0)
        return {'X': xray, 'Y': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.feat_info)

