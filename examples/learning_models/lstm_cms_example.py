# -*- coding: utf-8 -*-
"""Example of using LSTM on CMS phenotyping prediction
"""
# License: BSD 2 clause


# environment setting
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
os.chdir(root_dir)

sys.path.append(root_dir)

from pyhealth.data.expdata_generator import cms as cms_expdata_generator
from pyhealth.models.lstm import LSTM
from pyhealth import evaluation

if __name__ == "__main__":
    # override here to specify where the data locates
    # root_dir = ''
    # root_dir = os.path.abspath(os.path.join(__file__, "../../.."))

    expdata_id = '2020.0802.data.phenotyping.test.v4'

    # set up the datasets
    cur_dataset = cms_expdata_generator(expdata_id, root_dir=root_dir)
    cur_dataset.get_exp_data()
    cur_dataset.load_exp_data()
    # cur_dataset.show_data()

    # initialize the model for training
    expmodel_id = '2020.0802.model.phenotyping.test.v1'
    clf = LSTM(expmodel_id=expmodel_id, n_epoch=100)
    clf.fit(cur_dataset.train, cur_dataset.valid)

    # load the best model for inference
    clf.load_model()
    clf.inference(cur_dataset.test)
    pred_results = clf.get_results()
    print(pred_results)

    # evaluate the model
    evaluator = evaluation.__dict__['phenotyping']
    r = evaluator(pred_results['hat_y'], pred_results['y'])
    print(r)
