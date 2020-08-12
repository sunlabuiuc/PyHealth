from __future__ import division
from __future__ import print_function

import os
root_dir = os.getcwd().split('\examples\learning_models')[0]
print (root_dir)
os.chdir(root_dir)
import sys
sys.path.append(root_dir)

from pyhealth.data.expdata_generator import sequencedata as expdata_generator
expdata_id = '2020.0810.data.mortality.cms'

# phenotyping
cur_dataset = expdata_generator(expdata_id)
cur_dataset.get_exp_data(sel_task = 'mortality', data_root = r'./datasets/cms')
cur_dataset.load_exp_data()
cur_dataset.show_data()

from pyhealth.models.sequence.dipole import Dipole as model
#from pyhealth.models.sequence.lstm import LSTM as model
#from pyhealth.models.sequence.gru import GRU as model
#from pyhealth.models.sequence.embedgru import EmbedGRU as model
#from pyhealth.models.sequence.retain import Retain as model
#from pyhealth.models.sequence.raim import RAIM as model
#from pyhealth.models.sequence.tlstm import tLSTM as model
#from pyhealth.models.sequence.xgboost import XGBoost as model
#from pyhealth.models.sequence.rf import RandomForest as model

expmodel_id = '2020.0810.rf.test.phenotyping.cpu'
clf = model(expmodel_id = expmodel_id)
clf.fit(cur_dataset.train)

clf.load_model()
clf.inference(cur_dataset.test)
results = clf.get_results()
print (results)

from pyhealth.evaluation.evaluator import func 
r = func(results['hat_y'], results['y'])
print (r)