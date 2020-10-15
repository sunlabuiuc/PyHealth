# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
''
import unittest

import numpy as np

from pyhealth.utils.check import check_expdata_dir,  \
                                 check_model_dir,  \
                                 label_check
from pyhealth.data.simulation_data import generate_label

class TestCheck(unittest.TestCase):

    def test_check_expdata_dir(self):
        test_data_id = 'data_01'
        check_expdata_dir(test_data_id)
        assert os.path.exists(os.path.join('./experiments_data',test_data_id))
        os.rmdir(os.path.join('./experiments_data',test_data_id))
        os.rmdir('./experiments_data')

    def test_check_model_dir(self):
        test_model_id = 'model_01'
        check_model_dir(test_model_id)
        assert os.path.exists(os.path.join('./experiments_records',test_model_id))
        assert os.path.exists(os.path.join('./experiments_records',test_model_id, 'checkouts'))
        assert os.path.exists(os.path.join('./experiments_records',test_model_id, 'results'))
        os.rmdir(os.path.join('./experiments_records',test_model_id, 'checkouts'))
        os.rmdir(os.path.join('./experiments_records',test_model_id, 'results'))
        os.rmdir(os.path.join('./experiments_records',test_model_id))
        os.rmdir('./experiments_records')

    def test_label_check(self):
        y = generate_label(100, task = 'binaryclass', n_class = 2).reshape(-1, 1)
        tasktype = label_check(y, None)
        assert tasktype == 'binaryclass'

        y = generate_label(100, task = 'multiclass', n_class = 3)
        tasktype = label_check(y, None)
        assert tasktype == 'multiclass'

        y = generate_label(100, task = 'multilabel', n_class = 3)
        tasktype = label_check(y, None)
        assert tasktype == 'multilabel'
