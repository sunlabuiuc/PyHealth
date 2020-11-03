# -*- coding: utf-8 -*-

# License: BSD 2 clause

import os.path
import pandas as pd
import numpy as np
import torch
import pickle
import pywt

class DatasetReader:

    def __init__(self, data, n_level = 4):
        self.series = data['x']
        self.label = data['y']
        self.n_cases, self.n_feat = np.shape(np.array(data['x']))
        self.n_level = n_level
        self.n_visits = n_level + 2
        
    def __getitem__(self, index):
        sdata = np.array(self.series[index]).reshape(1, -1)
        featd = []
        featd.append(sdata[0])
        for _ in range(self.n_level):
            l, h = pywt.dwt(sdata, 'db9')
            cur_f = np.zeros(self.n_feat)
            cur_f[:len(h[0])] = h[0]
            featd.append(cur_f)
            sdata = l
        cur_f = np.zeros(self.n_feat)
        cur_f[:len(sdata[0])] = sdata[0]
        featd.append(cur_f)
        featd = np.stack(featd, axis = 0)
        label = self.label[index]
        return {'X': featd, 'Y': np.array(label)}

    def __len__(self):
        return len(self.series)


if __name__ == '__main__':
    data = {'x':np.random.randint(10,size=(20, 16)), 'y': np.zeros(20)}
    datareader = DatasetReader(data)
    _loader = torch.utils.data.DataLoader(datareader,
                                          batch_size=5,
                                          drop_last = True,
                                          shuffle=True)
    for idx, x in enumerate(_loader):
        print (x['X'].shape, x['Y'].shape)
