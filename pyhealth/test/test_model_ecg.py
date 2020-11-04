import unittest
import numpy as np
import torch

import os
import shutil

from pyhealth.data.expdata_generator import ecgdata as expdata_generator
from pyhealth.models.ecg.conv1d import Conv1D
from pyhealth.models.ecg.dblstm_ws import DBLSTM_WS
from pyhealth.models.ecg.deepres1d import DeepRES1D
from pyhealth.models.ecg.denseconv import DenseConv
from pyhealth.models.ecg.mina import MINA
from pyhealth.models.ecg.rcrnet import RCRNet
from pyhealth.models.ecg.sdaelstm import SDAELSTM
from pyhealth.models.ecg.rf import RandomForest
from pyhealth.models.ecg.xgboost import XGBoost

import sys 
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile

class TestImageModel(unittest.TestCase):
    
    expdata_id = 'test.ecg.model'
    
    def test_01(self):
        if os.path.exists('./experiments_data') is False:
            os.mkdir('./experiments_data')
        if os.path.exists('./datasets/image') is False:
            z = zipfile.ZipFile("./datasets/ecg.zip", "r")
            seq_x = []
            label_y = []
            for filename in z.namelist( ):
                z.extract(filename,'./datasets')
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.get_exp_data(sel_task='diagnose', data_root='./datasets/ecg')

    def test_02_conv1d_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.conv1d.cpu'
        clf = Conv1D(expmodel_id=expmodel_id, 
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

    def test_02_conv1d_gpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.conv1d.gpu'
        clf = Conv1D(expmodel_id=expmodel_id, 
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

    def test_02_rf(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.randomforest'
        clf = RandomForest(expmodel_id=expmodel_id)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_dblstm_ws_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.dblstm_ws.cpu'
        clf = DBLSTM_WS(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=3)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_deepres1d_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.deepres1d.cpu'
        clf = DeepRES1D(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=3)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_denseconv_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.denseconv.cpu'
        clf = DenseConv(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=3)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_mina_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.mina.cpu'
        clf = MINA(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=3)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_rcrnet_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.rcrnet.cpu'
        clf = RCRNet(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=3)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_sdaelstm_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.sdaelstm.cpu'
        clf = SDAELSTM(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=3)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_xgboost(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.xgboost'
        clf = XGBoost(expmodel_id=expmodel_id)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_03_delete(self):
        shutil.rmtree(os.path.join('./experiments_data', self.expdata_id))
        shutil.rmtree(os.path.join('./experiments_records', 'test.conv1d.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.conv1d.gpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.randomforest'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.xgboost'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.sdaelstm.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.rcrnet.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.mina.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.denseconv.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.deepres1d.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.dblstm_ws.cpu'))









