
import os.path
from .base_dataset import BaseDataset
import pandas as pd
import numpy as np
import torch
import pickle

def time_series_get(fpath):
    data = pd.read_csv(fpath, sep=',')
    return data.values

class DatasetReader:

    def __init__(self, data, sub_group = 2, task_type = None): 
        self.series_paths = data['x']
        self.label_list = data['y']
        self.seq_len = data['l']
        self.sub_group = sub_group
        self.task_type = task_type
        
    def get_data(self):
        if self.task_type is None:
            raise Exception('fill in correct task-type xxx from [\'binaryclass\', \'multiclass\', \'multilabel\', \'regression\']')
        target_x = []
        for index in range(len(self.series_paths)):
            xpath = self.series_paths[index]
            s_data = time_series_get(xpath)[:, 1:]
            sub_group_len = int(self.seq_len[index]/self.sub_group)
            cur_target_x = []
            for i in range(self.sub_group):
                cur_x = s_data[i * sub_group_len: (i + 1) * sub_group_len, : ]
                cur_target_x.append(np.mean(cur_x, 0))
            target_x.append(np.concatenate(cur_target_x, 0))
        target_x = np.array(target_x)
        if self.task_type == 'multilabel':
            label_y = np.array(self.label_list)
        else:
            labels = []
            if self.task_type == 'multilabel':
                if self.task_type == 'multiclass':
                    for rowlabel in self.label_list:
                        labels.append(np.argmax(np.array(rowlabel)))
                labels = np.array(labels)
            else:
                labels = np.array(self.label_list)
            label_y = labels.reshape(-1, 1)
        return {'X': target_x, 'Y': label_y}
