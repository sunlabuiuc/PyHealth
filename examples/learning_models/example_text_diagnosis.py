# -*- coding: utf-8 -*-
"""Example of clinical note tasks
"""
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

# environment setting
import os
import sys
from pathlib import Path

# this should be learning_models
curr_dir = os.getcwd()

# this should be pyhealth, which is two level up from learning_models library
root_dir = Path(curr_dir).parents[1]
os.chdir(root_dir)


from sklearn.ensemble import RandomForestClassifier

from pyhealth.data.expdata_generator import textdata as expdata_generator
from pyhealth.evaluation.evaluator import func

#from pyhealth.models.text.cnn import CNN as model
#from pyhealth.models.text.gru import GRU as model
#from pyhealth.models.text.dr_caml import DRCAML as model
#from pyhealth.models.text.jointlaat import JointLAAT as model
#from pyhealth.models.text.multirescnn import MultiResCNN as model
from pyhealth.models.text.dcan import DCAN as model
#from pyhealth.models.text.rf import RandomForest as model
#from pyhealth.models.text.xgboost import XGBoost as model

if __name__ == "__main__":
    # override here to specify where the data locates
    #    root_dir = '.'
    # root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    data_dir = os.path.join(root_dir, 'datasets', 'text')

    expdata_id = '2021.0105.text'

    # set up the datasets
    cur_dataset = expdata_generator(expdata_id, root_dir=root_dir)
    cur_dataset.get_exp_data(sel_task='diagnose', data_root=data_dir)
    cur_dataset.load_exp_data()
    # cur_dataset.show_data()

    # initialize the model for training
    expmodel_id = '2021.0103.cnn.text.diagnose.'
    clf = model(expmodel_id=expmodel_id, use_gpu=True)
    clf.fit(cur_dataset.train, cur_dataset.valid)
    
    # load the best model for inference
    clf.load_model()
    clf.inference(cur_dataset.valid)
    results = clf.get_results()
    # evaluate the model
    r = func(results['hat_y'], results['y'])
    print(r)
