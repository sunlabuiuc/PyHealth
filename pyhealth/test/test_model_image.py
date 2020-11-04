import unittest
import numpy as np
import torch

import os
import shutil

from pyhealth.data.expdata_generator import imagedata as expdata_generator
from pyhealth.models.image.typicalcnn import TypicalCNN
from pyhealth.models.image.basicnn import BasicCNN

import sys 
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile


class TestImageModel(unittest.TestCase):
    
    expdata_id = 'test.image.model'
    
    def test_01(self):
        if os.path.exists('./experiments_data') is False:
            os.mkdir('./experiments_data')
        if os.path.exists('./datasets/image') is False:
            z = zipfile.ZipFile("./datasets/image.zip", "r")
            seq_x = []
            label_y = []
            for filename in z.namelist( ):
                z.extract(filename,'./datasets')
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.get_exp_data(sel_task='diagnose', data_root='./datasets/image')

    def test_02_basiccnn_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.basicnn.cpu'
        clf = BasicCNN(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=6)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_basiccnn_gpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.basicnn.gpu'
        clf = BasicCNN(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=6)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_typicalcnn_gpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.typicalcnn'
        clf = TypicalCNN(expmodel_id=expmodel_id, 
                         cnn_name = 'resnet18',
                         n_batchsize=20, 
                         use_gpu=False,
                         n_epoch=6)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_03_delete(self):
        shutil.rmtree(os.path.join('./experiments_data', self.expdata_id))
        shutil.rmtree(os.path.join('./experiments_records', 'test.basicnn.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.basicnn.gpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.typicalcnn'))

