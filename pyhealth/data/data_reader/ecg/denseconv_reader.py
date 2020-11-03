# -*- coding: utf-8 -*-

# License: BSD 2 clause

import os.path
import pandas as pd
import numpy as np
import torch
import pickle
from scipy.fftpack import fft,ifft
import numpy as np

class DatasetReader:

    def __init__(self, data):
        self.series = data['x']
        self.label = data['y']
        if len(np.shape(np.array(data['x']))) == 2:
            self.n_cases, self.n_feat = np.shape(np.array(data['x']))
            self.n_channel = 1
        elif len(np.shape(np.array(data['x']))) == 3:
            self.n_cases, self.n_channel, self.n_feat = np.shape(np.array(data['x']))
        else:
            raise 'currently cannot support the data format'
        
    def __getitem__(self, index):
        if self.n_channel == 1:
            featd = fft(np.array(self.series[index]))
        else:
            featd = []
            for idx in range(self.n_channel):
                featd.append(fft(np.array(self.series[index][idx])))
            featd = np.array(featd)
        featd = np.reshape(featd, [self.n_channel, -1])    
        label = self.label[index]
        return {'X': featd, 'Y': np.array(label)}

    def __len__(self):
        return len(self.series)


if __name__ == '__main__':
    data = {'x':np.random.randint(10,size=(20, 3, 16)), 'y': np.zeros(20)}
    datareader = DatasetReader(data)
    _loader = torch.utils.data.DataLoader(datareader,
                                          batch_size=5,
                                          drop_last = True,
                                          shuffle=True)
    for idx, x in enumerate(_loader):
        print (x['X'].shape, x['Y'].shape)

