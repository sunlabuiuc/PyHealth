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

from pyhealth.data.expdata_generator import mimic as mimic_expdata_generator
from pyhealth.models.gru import GRU
from pyhealth import evaluation

if __name__ == "__main__":
    # override here to specify where the data locates
    # root_dir = ''
    # root_dir = os.path.abspath(os.path.join(__file__, "../../.."))

    expdata_id = '2020.0802.data.mortality.test.v1'

    # set up the datasets
    cur_dataset = mimic_expdata_generator(expdata_id, root_dir=root_dir)
    cur_dataset.get_exp_data(sel_task='mortality')
    cur_dataset.load_exp_data()
    # cur_dataset.show_data()

    # initialize the model for training
    expmodel_id = '2020.0802.data.mortality.test.v1'
    clf = GRU(expmodel_id=expmodel_id, task='mortality', n_epoch=100)
    clf.fit(cur_dataset.train, cur_dataset.valid)

    # load the best model for inference
    clf.load_model()
    clf.inference(cur_dataset.test)
    pred_results = clf.get_results()
    print(pred_results)

    # evaluate the model
    evaluator = evaluation.__dict__['mortality']
    r = evaluator(pred_results['hat_y'], pred_results['y'])
    print(r)
