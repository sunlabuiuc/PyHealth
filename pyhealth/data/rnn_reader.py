
import os.path
from .base_dataset import BaseDataset
import pandas as pd
import numpy as np
import torch
import pickle

def time_series_get(fpath):
    data = pd.read_csv(fpath, sep=',')
    return data.values

class DatasetReader(BaseDataset):

    def __init__(self, data):
        BaseDataset.__init__(self, data)     
        self.series_paths = data['x']
        self.label_list = data['y']
#         self.seq_len = 200
        self.seq_len = max(data['l']) if max(data['l'])<200 else 200
        
    def __getitem__(self, index):
        xpath = self.series_paths[index]
        s_data = time_series_get(xpath)
        data = s_data[:self.seq_len, 1: ]
        l, w = np.shape(data)
        time = s_data[:self.seq_len, 0]
        time[1:] = time[1:] - time[:-1]
        time[0] = 0.
        x_time = np.zeros(self.seq_len)
        x_time[:l] = time
        x_series = np.zeros([self.seq_len, w])
        x_series[:l, :] = data
        x_mask = np.zeros(self.seq_len)
        x_mask[:l] = 1.
        x_mask_cur = np.zeros(self.seq_len)
        x_mask_cur[l-1] = 1.
        label = self.label_list[index]
        return {'X': np.array(x_series), 'M': np.array(x_mask), 'cur_M': np.array(x_mask_cur), 'Y': np.array(label), 'T':np.array(x_time)}

    def __len__(self):
        return len(self.series_paths)

