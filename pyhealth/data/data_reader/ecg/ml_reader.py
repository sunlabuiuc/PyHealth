# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os.path
from .base_dataset import BaseDataset
import pandas as pd
import numpy as np
import torch
import pickle


class DatasetReader:

    def __init__(self, data, task_type = None): 
        self.series = data['x']
        self.label = data['y']
        self.task_type = task_type
        
    def get_data(self):
        if self.task_type is None:
            raise Exception('fill in correct task-type xxx from [\'binaryclass\', \'multiclass\', \'multilabel\', \'regression\']')
        target_x = np.array(self.series)
        if self.task_type == 'multilabel':
            label_y = np.array(self.label)
        else:
            labels = []
            if self.task_type == 'multiclass':
                for rowlabel in self.label:
                    labels.append(np.argmax(np.array(rowlabel)))
                labels = np.array(labels)
            else:
                labels = np.array(self.label)
            label_y = labels.reshape(-1, 1)
        return {'X': target_x, 'Y': label_y}
