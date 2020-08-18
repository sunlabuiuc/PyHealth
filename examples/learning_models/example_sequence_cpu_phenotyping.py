# -*- coding: utf-8 -*-
"""Example of using GRU on MIMIC demo mortality prediction
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

### May choose any of these models
from pyhealth.models.sequence.dipole import Dipole
# from pyhealth.models.sequence.lstm import LSTM as model
# from pyhealth.models.sequence.gru import GRU as GRU
# from pyhealth.models.sequence.embedgru import EmbedGRU as model
# from pyhealth.models.sequence.retain import Retain as model
# from pyhealth.models.sequence.raim import RAIM as model
# from pyhealth.models.sequence.tlstm import tLSTM as model
# from pyhealth.models.sequence.xgboost import XGBoost as model
# from pyhealth.models.sequence.rf import RandomForest as model

from pyhealth.data.expdata_generator import sequencedata as expdata_generator
from pyhealth.evaluation.evaluator import func

if __name__ == "__main__":
    # override here to specify where the data locates
    # root_dir = ''
    # root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    data_dir = os.path.join(root_dir, 'datasets', 'mimic')

    expdata_id = '2020.0811.data.phenotyping.test.v2'

    # set up the datasets
    cur_dataset = expdata_generator(expdata_id, root_dir=root_dir)
    cur_dataset.get_exp_data(sel_task='phenotyping', data_root=data_dir)
    cur_dataset.load_exp_data()
    # cur_dataset.show_data()

    # initialize the model for training
    expmodel_id = '2020.0811.model.phenotyping.test.v2'
    clf = Dipole(expmodel_id=expmodel_id, n_batchsize=20, use_gpu=False,
                 n_epoch=100)
    clf.fit(cur_dataset.train, cur_dataset.valid)

    # load the best model for inference
    clf.load_model()
    clf.inference(cur_dataset.test)
    pred_results = clf.get_results()
    print(pred_results)

    # evaluate the model
    r = func(pred_results['hat_y'], pred_results['y'])
    print(r)
