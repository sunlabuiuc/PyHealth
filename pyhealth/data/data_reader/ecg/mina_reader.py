# -*- coding: utf-8 -*-

# License: BSD 2 clause

import os.path
import pandas as pd
import numpy as np
import torch
import pickle
import numpy as np

import scipy.io
from scipy.signal import butter, lfilter, periodogram

def filter_channel(x):
    
    signal_freq = 300
    
    ### candidate channels for ECG
    P_wave = (0.67,5)
    QRS_complex = (10,50)
    T_wave = (1,7)
    muscle = (5,50)
    resp = (0.12,0.5)
    ECG_preprocessed = (0.5, 50)
    wander = (0.001, 0.5)
    noise = 50
    
    ### use low (wander), middle (ECG_preprocessed) and high (noise) for example
    bandpass_list = [wander, ECG_preprocessed]
    highpass_list = [noise]
    
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    ### out including original x
    out_list = [x]
    
    for bandpass in bandpass_list:
        low = bandpass[0] / nyquist_freq
        high = bandpass[1] / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, x)
        out_list.append(y)
        
    for highpass in highpass_list:
        high = highpass / nyquist_freq
        b, a = butter(filter_order, high, btype="high")
        y = lfilter(b, a, x)
        out_list.append(y)
        
    out = np.array(out_list)
    
    return out

def compute_beat(X):
    out = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.concatenate([[0], np.abs(np.diff(X[i,j,:]))])
    return out

def compute_rhythm(X, n_split):
    cnt_split = int(X.shape[2]/n_split)
    out = np.zeros((X.shape[0], X.shape[1], cnt_split))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            tmp_ts = X[i,j,:]
            tmp_ts_cut = np.split(tmp_ts, X.shape[2]/n_split)
            for k in range(cnt_split):
                out[i, j, k] = np.std(tmp_ts_cut[k])
    return out

def compute_freq(X):
    out = np.zeros((X.shape[0], X.shape[1], 1))
    fs = 300
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            _, Pxx_den = periodogram(X[i,j,:], fs)
            out[i, j, 0] = np.sum(Pxx_den)
    return out

  
def make_knowledge_physionet(X_data, n_split=50):
    # compute knowledge
    K_train_beat = compute_beat(X_data)
    K_train_rhythm = compute_rhythm(X_data, n_split)
    K_train_freq = compute_freq(X_data)
    return X_data, K_train_beat, K_train_rhythm, K_train_freq


class DatasetReader:

    def __init__(self, data, n_split=50):
        self.label = data['y']
        self.n_split = n_split
        if len(np.shape(np.array(data['x']))) == 2:
            self.series = data['x']
            self.n_cases, self.n_feat = np.shape(np.array(data['x']))
            self.n_channel = 4
        elif len(np.shape(np.array(data['x']))) == 3:
            self.n_cases, n_channel, self.n_feat = np.shape(np.array(data['x']))
            if n_channel != 1:
                raise 'currently cannot support the data format'   
            self.n_channel = 4
            self.series = data['x'][:, 0, :]
        else:
            raise 'currently cannot support the data format'
        
    def __getitem__(self, index):
        feat_x = filter_channel(self.series[index])
        X_data, K_train_beat, K_train_rhythm, K_train_freq = make_knowledge_physionet(np.array([feat_x]), n_split = self.n_split)
        label = self.label[index]
        return {'X_data': X_data[0], 
                'K_train_beat': K_train_beat[0],
                'K_train_rhythm': K_train_rhythm[0],
                'K_train_freq': K_train_freq[0],
                'Y': np.array(label)}

    def __len__(self):
        return len(self.series)


if __name__ == '__main__':
    data = {'x':np.random.randint(10,size=(20, 1, 300)), 'y': np.zeros(20)}
    datareader = DatasetReader(data)
    _loader = torch.utils.data.DataLoader(datareader,
                                          batch_size=5,
                                          drop_last = True,
                                          shuffle=True)
    for idx, x in enumerate(_loader):
        print (x['X_data'].shape, x['K_train_beat'].shape, x['K_train_rhythm'].shape, x['K_train_freq'].shape, x['Y'].shape)


