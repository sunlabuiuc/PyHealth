# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import shutil

import numpy as np

import zipfile  

from pyhealth.data.expdata_generator import sequencedata
from pyhealth.data.expdata_generator import imagedata
from pyhealth.data.expdata_generator import ecgdata

class TestGenerator(unittest.TestCase):

    def test_mimic_generator(self):
        expdata_id = 'xxx.id.mimic'
        if os.path.exists('./experiments_data') is False:
            os.mkdir('./experiments_data')
        if os.path.exists('./datasets/mimic'):
              test_seq = sequencedata(expdata_id)
              test_seq.get_exp_data(sel_task='mortality',data_root = './datasets/mimic')
              test_seq.load_exp_data()
              shutil.rmtree(os.path.join('./experiments_data', expdata_id))
        else:
            z = zipfile.ZipFile("./datasets/mimic.zip", "r")
            seq_x = []
            label_y = []
            for filename in z.namelist( ):
                z.extract(filename,'./datasets')
            test_seq = sequencedata(expdata_id)
            test_seq.get_exp_data(sel_task='mortality',data_root = './datasets/mimic')
            test_seq.load_exp_data()
            shutil.rmtree(os.path.join('./experiments_data', expdata_id))
            shutil.rmtree('./datasets/mimic')
        shutil.rmtree('./experiments_data')

    def test_image_generator(self):
        expdata_id = 'xxx.id.image'
        if os.path.exists('./experiments_data') is False:
            os.mkdir('./experiments_data')
        if os.path.exists('./datasets/image'):
              test_image = imagedata(expdata_id)
              test_image.get_exp_data(sel_task='diagnose',data_root = './datasets/image')
              test_image.load_exp_data()
              shutil.rmtree(os.path.join('./experiments_data', expdata_id))
        else:
            z = zipfile.ZipFile("./datasets/image.zip", "r")
            seq_x = []
            label_y = []
            for filename in z.namelist( ):
                z.extract(filename,'./datasets')
            test_image = imagedata(expdata_id)
            test_image.get_exp_data(sel_task='diagnose',data_root = './datasets/image')
            test_image.load_exp_data()
            shutil.rmtree(os.path.join('./experiments_data', expdata_id))
            shutil.rmtree('./datasets/image')
        shutil.rmtree('./experiments_data')

    def test_ecg_generator(self):
        expdata_id = 'xxx.id.image'
        if os.path.exists('./experiments_data') is False:
            os.mkdir('./experiments_data')
        if os.path.exists('./datasets/ecg'):
              test_ecg = ecgdata(expdata_id)
              test_ecg.get_exp_data(sel_task='diagnose',data_root = './datasets/ecg')
              test_ecg.load_exp_data()
              shutil.rmtree(os.path.join('./experiments_data', expdata_id))
        else:
            z = zipfile.ZipFile("./datasets/ecg.zip", "r")
            seq_x = []
            label_y = []
            for filename in z.namelist( ):
                z.extract(filename,'./datasets')
            test_ecg = ecgdata(expdata_id)
            test_ecg.get_exp_data(sel_task='diagnose',data_root = './datasets/ecg')
            test_ecg.load_exp_data()
            shutil.rmtree(os.path.join('./experiments_data', expdata_id))
            shutil.rmtree('./datasets/ecg')
        shutil.rmtree('./experiments_data')
