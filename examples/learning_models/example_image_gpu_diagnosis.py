# -*- coding: utf-8 -*-
"""Example of image diagnosis with GPU on CNN
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

from pyhealth.data.expdata_generator import imagedata as expdata_generator
from pyhealth.models.image.typicalcnn import TypicalCNN as model
from pyhealth.evaluation.evaluator import func

if __name__ == "__main__":
    # override here to specify where the data locates
    # root_dir = ''
    # root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    data_dir = os.path.join(root_dir, 'datasets', 'image')

    expdata_id = '2020.0810.image'

    # set up the datasets
    cur_dataset = expdata_generator(expdata_id, root_dir=root_dir)
    cur_dataset.get_exp_data(sel_task='diagnose', data_root=data_dir)
    cur_dataset.load_exp_data()
    # cur_dataset.show_data()

    # initialize the model for training
    expmodel_id = '2020.0810.cnn.image.diagnose.'
    clf = model(expmodel_id=expmodel_id, n_epoch=100, use_gpu=True,
                gpu_ids='0,1')
    clf.fit(cur_dataset.train, cur_dataset.valid)

    # load the best model for inference
    clf.load_model()
    clf.inference(cur_dataset.test)
    results = clf.get_results()
    print(results)

    # evaluate the model
    r = func(results['hat_y'], results['y'])
    print(r)
