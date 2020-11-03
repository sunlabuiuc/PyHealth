# -*- coding: utf-8 -*-

# License: BSD 2 clause

import os.path
import pandas as pd
import numpy as np
import torch
import pickle
import numpy as np

class DatasetReader:

    def __init__(self, 
                 data, 
                 n_window_size = 50, 
                 slide_thres = 10):
        self.label = np.array([np.argmax(cy) for cy in data['y']])
        self.labels = data['y']
        self.slide_thres = slide_thres
        self.n_window_size = n_window_size
        if len(np.shape(np.array(data['x']))) == 2:
            self.n_cases, self.n_feat = np.shape(np.array(data['x']))
            self.series = np.array(data['x']).reshape((self.n_cases, 1, self.n_feat))
            self.n_channel = 1
        elif len(np.shape(np.array(data['x']))) == 3:
            self.n_cases, n_channel, self.n_feat = np.shape(np.array(data['x']))
            self.n_channel = n_channel
            self.series = data['x']
        else:
            raise 'currently cannot support the data format'
        unique,count=np.unique(self.label,return_counts=True)
        self.max_n_visits = int(self.n_feat/int(slide_thres * min(count)/max(count))) + 1
        self.dict = {unique[idx]: int(slide_thres * count[idx]/max(count)) for idx in range(len(unique))}
        self.pad_len = self.max_n_visits * max(self.n_window_size, slide_thres)

    def __getitem__(self, index):
        label = self.label[index]
        new_feat = np.zeros((self.n_channel, self.pad_len))
        new_feat[:, :np.shape(self.series[index])[-1]] = self.series[index]
        reog_feat = []
        cur_slide_size = self.dict[label]
        mask_cur = mask = np.zeros(self.max_n_visits)
        flag = -1
        for idx in range(self.max_n_visits):
            if idx*cur_slide_size<np.shape(self.series[index])[-1] and (idx+1)*cur_slide_size>=np.shape(self.series[index])[-1]:
                flag = idx
            reog_feat.append(new_feat[:, idx*cur_slide_size: idx*cur_slide_size+self.n_window_size])
        reog_feat = np.array(reog_feat)
        mask_cur[idx] = 1
        mask[:idx+1] = 1
        return {'X': reog_feat,
                'M': mask,
                'cur_M': mask_cur,
                'Y': np.array(self.labels[index])}

    def __len__(self):
        return len(self.series)


if __name__ == '__main__':
    data = {'x':np.random.randint(10,size=(20, 2, 300)), 'y': np.random.randint(0,4,20)}
    datareader = DatasetReader(data)
    _loader = torch.utils.data.DataLoader(datareader,
                                          batch_size=5,
                                          drop_last = True,
                                          shuffle=True)
    for idx, x in enumerate(_loader):
        print (x['X'].shape, x['M'].shape, x['cur_M'].shape, x['Y'].shape)


