import unittest
import numpy as np
import torch

import os
import shutil

from pyhealth.models.sequence.dipole import Dipole
from pyhealth.models.sequence.lstm import LSTM
from pyhealth.models.sequence.gru import GRU
from pyhealth.models.sequence.embedgru import EmbedGRU
from pyhealth.models.sequence.retain import Retain
from pyhealth.models.sequence.raim import RAIM
from pyhealth.models.sequence.tlstm import tLSTM
from pyhealth.models.sequence.stagenet import StageNet
from pyhealth.models.sequence.xgboost import XGBoost
from pyhealth.models.sequence.rf import RandomForest

from pyhealth.data.expdata_generator import sequencedata as expdata_generator
from pyhealth.evaluation.evaluator import func

import sys 
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile

class TestSequentialModel(unittest.TestCase):
    
    expdata_id = 'test.sequence.model'
    
    def test_01(self):
        if os.path.exists('./experiments_data') is False:
            os.mkdir('./experiments_data')
        if os.path.exists('./datasets/mimic') is False:
            z = zipfile.ZipFile("./datasets/mimic.zip", "r")
            seq_x = []
            label_y = []
            for filename in z.namelist( ):
                z.extract(filename,'./datasets')
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.get_exp_data(sel_task='mortality', data_root='./datasets/mimic')

    def test_02_lstm_cpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.lstm.gpu'
        clf = LSTM(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=False,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()


    def test_02_lstm_gpu(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.lstm.cpu'
        clf = LSTM(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_gru(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.gru'
        clf = GRU(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_embedgru(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.embedgru'
        clf = EmbedGRU(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_dipole(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.dipole'
        clf = Dipole(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_retain(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.retain'
        clf = Retain(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_raim(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.raim'
        clf = RAIM(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_tlstm(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.tlstm'
        clf = tLSTM(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
        clf.fit(cur_dataset.train, cur_dataset.valid)
        clf.load_model()
        clf.inference(cur_dataset.test)
        pred_results = clf.get_results()
        assert np.shape(pred_results['hat_y']) == np.shape(pred_results['y'])
        assert True not in np.isnan(pred_results['hat_y']).tolist()
        assert True not in np.isnan(pred_results['hat_y']*0).tolist()

    def test_02_stagenet(self):
        cur_dataset = expdata_generator(self.expdata_id)
        cur_dataset.load_exp_data()
        expmodel_id = 'test.stagenet'
        clf = StageNet(expmodel_id=expmodel_id, 
                   n_batchsize=20, 
                   use_gpu=True,
                   n_epoch=10)
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

    def test_02_rm(self):
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

    def test_03_delete(self):
        shutil.rmtree(os.path.join('./experiments_data', self.expdata_id))
        shutil.rmtree(os.path.join('./experiments_records', 'test.lstm.cpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.lstm.gpu'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.gru'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.embedgru'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.dipole'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.retain'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.raim'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.tlstm'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.stagenet'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.xgboost'))
        shutil.rmtree(os.path.join('./experiments_records', 'test.randomforest'))
