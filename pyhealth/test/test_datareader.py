# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
import numpy as np

import torch

import zipfile  

import shutil

from pyhealth.data.simulation_data import generate_simulation_sequence_data
from pyhealth.data.simulation_data import generate_simulation_image_data
from pyhealth.data.simulation_data import generate_simulation_ecg_data

from pyhealth.data.data_reader.sequence import dl_reader as seq_dl_reader
from pyhealth.data.data_reader.sequence import ml_reader as seq_ml_reader
from pyhealth.data.data_reader.image import dl_reader as image_dl_reader
from pyhealth.data.data_reader.ecg import dl_reader as ecg_dl_reader
from pyhealth.data.data_reader.ecg import ml_reader as ecg_ml_reader

class TestDatareader(unittest.TestCase):

    def test_seq_ml_reader(self):
        test_n_sample = 10
        test_batch_size = 2
        test_n_feat = 30
        test_sub_group = 3
        data = generate_simulation_sequence_data(n_sample = test_n_sample, 
                                                 n_feat = test_n_feat,
                                                 task = 'binaryclass')()
        seq_ds = seq_ml_reader.DatasetReader(data, 
                                             sub_group = test_sub_group, 
                                             data_type = 'aggregation', 
                                             task_type = 'binaryclass').get_data()
        assert np.shape(seq_ds['X'])[0] == test_n_sample
        assert np.shape(seq_ds['X'])[1] == test_n_feat * test_sub_group
        assert np.shape(seq_ds['Y'])[0] == test_n_sample
        assert np.shape(seq_ds['Y'])[1] == 1

        test_n_sample = 10
        test_batch_size = 2
        test_n_feat = 30
        test_sub_group = 3
        test_n_class = 3
        data = generate_simulation_sequence_data(n_sample = test_n_sample, 
                                                 n_feat = test_n_feat,
                                                 task = 'multiclass',
                                                 n_class = test_n_class)()
        seq_ds = seq_ml_reader.DatasetReader(data, 
                                             sub_group = test_sub_group, 
                                             data_type = 'aggregation', 
                                             task_type = 'multiclass').get_data()
        assert np.shape(seq_ds['X'])[0] == test_n_sample
        assert np.shape(seq_ds['X'])[1] == test_n_feat * test_sub_group
        assert np.shape(seq_ds['Y'])[0] == test_n_sample
        assert np.shape(seq_ds['Y'])[1] == 1

        test_n_sample = 10
        test_batch_size = 2
        test_n_feat = 30
        test_sub_group = 3
        test_n_class = 3
        data = generate_simulation_sequence_data(n_sample = test_n_sample, 
                                                 n_feat = test_n_feat,
                                                 task = 'multilabel',
                                                 n_class = test_n_class)()
        seq_ds = seq_ml_reader.DatasetReader(data, 
                                             sub_group = test_sub_group, 
                                             data_type = 'aggregation', 
                                             task_type = 'multilabel').get_data()
        assert np.shape(seq_ds['X'])[0] == test_n_sample
        assert np.shape(seq_ds['X'])[1] == test_n_feat * test_sub_group
        assert np.shape(seq_ds['Y'])[0] == test_n_sample
        assert np.shape(seq_ds['Y'])[1] == test_n_class

    def test_seq_dl_reader(self):
        test_batch_size = 2
        test_n_feat = 30
        data = generate_simulation_sequence_data(n_sample = 10, 
                                                 n_feat = test_n_feat,
                                                 task = 'binaryclass',
                                                 n_class = 2)()
        seq_ds = seq_dl_reader.DatasetReader(data, data_type = 'aggregation')
        seq_loader = torch.utils.data.DataLoader(seq_ds, batch_size=test_batch_size)
        for batch_idx, databatch in enumerate(seq_loader):
            assert databatch['X'].size()[0] == test_batch_size
            assert databatch['X'].size()[2] == test_n_feat
            assert databatch['M'].size()[0] == test_batch_size
            assert databatch['cur_M'].size()[0] == test_batch_size
            assert databatch['Y'].size()[0] == test_batch_size
            assert len(databatch['Y'].size()) == 1
            assert databatch['T'].size()[0] == test_batch_size
            assert databatch['X'].size()[1] == databatch['M'].size()[1]
            assert databatch['X'].size()[1] == databatch['cur_M'].size()[1]
            assert databatch['X'].size()[1] == databatch['T'].size()[1]

        test_n_class = 3
        data = generate_simulation_sequence_data(n_sample = 10, 
                                                 n_feat = test_n_feat,
                                                 task = 'multiclass',
                                                 n_class = test_n_class)()
        seq_ds = seq_dl_reader.DatasetReader(data, data_type = 'aggregation')
        seq_loader = torch.utils.data.DataLoader(seq_ds, batch_size=test_batch_size)
        for batch_idx, databatch in enumerate(seq_loader):
            assert databatch['X'].size()[0] == test_batch_size
            assert databatch['X'].size()[2] == test_n_feat
            assert databatch['M'].size()[0] == test_batch_size
            assert databatch['cur_M'].size()[0] == test_batch_size
            assert databatch['Y'].size()[0] == test_batch_size
            assert databatch['Y'].size()[1] == test_n_class
            assert databatch['T'].size()[0] == test_batch_size
            assert databatch['X'].size()[1] == databatch['M'].size()[1]
            assert databatch['X'].size()[1] == databatch['cur_M'].size()[1]
            assert databatch['X'].size()[1] == databatch['T'].size()[1]

        test_n_class = 3
        data = generate_simulation_sequence_data(n_sample = 10, 
                                                 n_feat = test_n_feat,
                                                 task = 'multilabel',
                                                 n_class = test_n_class)()
        seq_ds = seq_dl_reader.DatasetReader(data, data_type = 'aggregation')
        seq_loader = torch.utils.data.DataLoader(seq_ds, batch_size=test_batch_size)
        for batch_idx, databatch in enumerate(seq_loader):
            assert databatch['X'].size()[0] == test_batch_size
            assert databatch['X'].size()[2] == test_n_feat
            assert databatch['M'].size()[0] == test_batch_size
            assert databatch['cur_M'].size()[0] == test_batch_size
            assert databatch['Y'].size()[0] == test_batch_size
            assert databatch['Y'].size()[1] == test_n_class
            assert databatch['T'].size()[0] == test_batch_size
            assert databatch['X'].size()[1] == databatch['M'].size()[1]
            assert databatch['X'].size()[1] == databatch['cur_M'].size()[1]
            assert databatch['X'].size()[1] == databatch['T'].size()[1]

        test_batch_size = 2
        test_n_feat = 30
        data = generate_simulation_sequence_data(n_sample = 10, 
                                                 n_feat = test_n_feat,
                                                 task = 'regression',
                                                 n_class = 2)()
        seq_ds = seq_dl_reader.DatasetReader(data, data_type = 'aggregation')
        seq_loader = torch.utils.data.DataLoader(seq_ds, batch_size=test_batch_size)
        for batch_idx, databatch in enumerate(seq_loader):
            assert databatch['X'].size()[0] == test_batch_size
            assert databatch['X'].size()[2] == test_n_feat
            assert databatch['M'].size()[0] == test_batch_size
            assert databatch['cur_M'].size()[0] == test_batch_size
            assert databatch['Y'].size()[0] == test_batch_size
            assert len(databatch['Y'].size()) == 1
            assert databatch['T'].size()[0] == test_batch_size
            assert databatch['X'].size()[1] == databatch['M'].size()[1]
            assert databatch['X'].size()[1] == databatch['cur_M'].size()[1]
            assert databatch['X'].size()[1] == databatch['T'].size()[1]
