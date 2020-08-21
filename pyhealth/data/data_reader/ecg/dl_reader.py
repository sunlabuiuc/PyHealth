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

    def __init__(self, data):
        BaseDataset.__init__(self, data)     
        self.series = data['x']
        self.label = data['y']
        
    def __getitem__(self, index):
        sdata = np.array(self.series[index]).reshape(1, -1)
        label = self.label[index]
        return {'X': sdata, 'Y': np.array(label)}

    def __len__(self):
        return len(self.series)

