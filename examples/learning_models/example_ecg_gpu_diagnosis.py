# -*- coding: utf-8 -*-
"""Example of image tasks with CPU
"""
# License: BSD 2 clause

# environment setting
import os
import sys
from pathlib import Path

# this should be learning_models
curr_dir = os.getcwd()

# this should be pyhealth, which is two level up from learning_models library
root_dir = Path(curr_dir).parents[1]
os.chdir(root_dir)

sys.path.append(root_dir)

from pyhealth.data.expdata_generator import ecgdata as expdata_generator
from pyhealth.models.ecg.conv1d import Conv1D as model
#from pyhealth.models.ecg.dblstm_ws import DBLSTM_WS as model
#from pyhealth.models.ecg.denseconv import DenseConv as model
#from pyhealth.models.ecg.deepres1d import DeepRES1D as model
#from pyhealth.models.ecg.sdaelstm import SDAELSTM as model
#from pyhealth.models.ecg.mina import MINA as model
#from pyhealth.models.ecg.rcrnet import RCRNet as model
#from pyhealth.models.ecg.rf import RandomForest as model
#from pyhealth.models.ecg.xgboost import XGBoost as model
from pyhealth.evaluation.evaluator import func

data_dir = os.path.join(root_dir, 'datasets', 'ecg')

expdata_id = '2020.1104.data.diagnose.ecg'

# set up the datasets
cur_dataset = expdata_generator(expdata_id, root_dir=root_dir)
cur_dataset.get_exp_data(sel_task='diagnose', data_root=data_dir)
cur_dataset.load_exp_data()
cur_dataset.show_data()

# initialize the model for training
expmodel_id = '2020.1104.ecg.diagnose.'
clf = model(expmodel_id=expmodel_id, n_epoch=10, use_gpu=True)
clf.fit(cur_dataset.train, cur_dataset.valid)

# load the best model for inference
clf.load_model()
clf.inference(cur_dataset.test)
results = clf.get_results()
print(results)

# evaluate the model
r = func(results['hat_y'], results['y'])
print(r)

