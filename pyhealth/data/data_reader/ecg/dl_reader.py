# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os.path
from .base_dataset import BaseDataset
import pandas as pd
import numpy as np
import torch
import pickle

class DatasetReader(BaseDataset):

    def __init__(self, data, data_type = 'aggregation'):
        BaseDataset.__init__(self, data) 
        self.data_type = data_type
        if self.data_type == 'aggregation':
            self.feat_info = data['x']
            self.label = data['y']
        else:
            raise 'distribut reader coming...'

    def __getitem__(self, index):
        if self.data_type == 'aggregation':
            sdata = np.array(self.feat_info[index]).reshape(1, -1)
            label = self.label[index]
            return {'X': sdata, 'Y': np.array(label)}

    def __len__(self):
        return len(self.feat_info)


