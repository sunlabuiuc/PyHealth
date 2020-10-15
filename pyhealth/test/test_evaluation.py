# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

import numpy as np

from pyhealth.evaluation.binaryclass import evaluator as binary_evalua
from pyhealth.evaluation.multiclass import evaluator as multiclass_evalua
from pyhealth.evaluation.multilabel import evaluator as multilabel_evalua
from pyhealth.evaluation.evaluator import func as evaluator
from pyhealth.data.simulation_data import generate_label

class TestEvaluation(unittest.TestCase):

    def _simulate_binary_data(self, N_case):
        y = generate_label(N_case, task = 'binaryclass').reshape(-1, 1)
        hat_y = np.random.rand(N_case).reshape(-1, 1)
        return hat_y, y

    def _simulate_multiclass_data(self, N_case, N_class):
        y = generate_label(N_case, task = 'multiclass', n_class = N_class)
        hat_y = np.random.rand(N_case, N_class)
        return hat_y, y

    def _simulate_multilabel_data(self, N_case, N_class):
        y = generate_label(N_case, task = 'multilabel', n_class = N_class)
        hat_y = np.random.rand(N_case, N_class)
        return hat_y, y

    def test_binary(self):
        hat_y, y = self._simulate_binary_data(100)
        assert np.shape(hat_y) == np.shape(y)
        assert int(np.sum(np.sum(y, 1))) > 0 and int(np.sum(np.sum(y, 1))) < 100
        results = binary_evalua(hat_y, y)

    def test_multiclass(self):
        hat_y, y = self._simulate_multiclass_data(100, 3)
        assert np.shape(hat_y) == np.shape(y)
        assert int(np.sum(np.sum(y, 1))/100) == 1
        results = multiclass_evalua(hat_y, y)

    def test_multilabel(self):
        hat_y, y = self._simulate_multilabel_data(100, 3)
        assert np.shape(hat_y) == np.shape(y)
        assert np.sum(np.sum(y, 1))/100. > 1.
        results = multilabel_evalua(hat_y, y)

    def test_evaluator(self):
        hat_y, y = self._simulate_binary_data(100)
        unified_r = evaluator(hat_y, y)
        be_r = binary_evalua(hat_y, y)
        for key in unified_r.keys():
            assert unified_r[key] == be_r[key]

        hat_y, y = self._simulate_multiclass_data(100, 3)
        unified_r = evaluator(hat_y, y)
        me_r = multiclass_evalua(hat_y, y)
        for key in unified_r.keys():
            assert unified_r[key] == me_r[key]

        hat_y, y = self._simulate_multilabel_data(100, 3)
        unified_r = evaluator(hat_y, y)
        me_r = multilabel_evalua(hat_y, y)
        for key in unified_r.keys():
            assert unified_r[key] == me_r[key]
