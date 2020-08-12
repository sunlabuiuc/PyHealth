from __future__ import division
from __future__ import print_function

import os
root_dir = os.getcwd().split('\examples\learning_models')[0]
os.chdir(root_dir)
import sys
sys.path.append(root_dir)

from pyhealth.data.expdata_generator import imagedata as expdata_generator
expdata_id = '2020.0810.image'
cur_dataset = expdata_generator(expdata_id)
#cur_dataset.get_exp_data(sel_task = 'diagnose', data_root = r'./datasets/image')
cur_dataset.load_exp_data()
cur_dataset.show_data()

from pyhealth.models.image.typicalcnn import TypicalCNN as model

expmodel_id = '2020.0810.cnn.test.diagnose.'
clf = model(expmodel_id = expmodel_id)

#clf.fit(cur_dataset.train, cur_dataset.valid)

clf.load_model()
clf.inference(cur_dataset.test)
results = clf.get_results()
print (results)

from pyhealth.evaluation.evaluator import func 
r = func(results['hat_y'], results['y'])
print (r)