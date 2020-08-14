# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os
import numpy as np

def check_expdata_dir(expdata_id):
    """
    Check whether the exp data folder  exist,
        If not, will create the folder

    Parameters

    ----------
    root_dir :  str,
        root dir of current project

    expdata_id : str, optional (default='init.test') 
        name of current experiment data

    """

    r_root = os.path.join('./', 'experiments_data')
    if os.path.exists(r_root) is False:
        os.mkdir(r_root)
    exp_root = os.path.join(r_root, expdata_id)
    if os.path.exists(exp_root) is False:
        os.mkdir(exp_root)

def check_model_dir(expmodel_id):
    """
    Check whether the checkouts/results folders of current experiment(exp_id) exist,
        If not, will create both folders

    Parameters

    ----------
    root_dir :  str,
        root dir of current project

    expmodel_id : str, optional (default='init.test') 
        name of current experiment

    """

    r_root = os.path.join('./', 'experiments_records')
    if os.path.exists(r_root) is False:
        os.mkdir(r_root)
    exp_root = os.path.join(r_root, expmodel_id)
    if os.path.exists(exp_root) is False:
        os.mkdir(exp_root)
    checkout_dir = os.path.join(exp_root, 'checkouts')
    result_dir = os.path.join(exp_root, 'results')
    if os.path.exists(checkout_dir) is False:
        os.mkdir(checkout_dir)
    if os.path.exists(result_dir) is False:
        os.mkdir(result_dir)

def label_check(y, hat_y = None, assign_task_type = None):
    
    def check_task_type(y, hat_y = None):
        if hat_y is not None:
            try:
                hat_y = np.array(hat_y).astype(float)
                y = np.array(y).astype(float)
            except:
                raise Exception('not support current data type of hat_y, y')
            _shape_hat_y, _shape_y = np.shape(hat_y), np.shape(y)
            if _shape_hat_y != _shape_y:
                raise Exception('the data shape is not inconformity between y and hey_y')

        label_n_check = set([])
        label_item_set = set([])
        label_row_set = set([])
        for each_y_path in y:
            label_n_check.add(len(np.array(each_y_path)))
            label_item_set.update(np.array(each_y_path).astype(int).tolist())
            label_row_set.add(sum(np.array(each_y_path).astype(int)))

        if len(label_n_check) != 1:
            raise Exception('label_n is inconformity in data')

        if len(label_item_set) <= 1:
            raise Exception('value space size <=1 is unvalid')
        elif len(label_item_set) == 2:
            if 0 in label_item_set and 1 in label_item_set:
                if list(label_n_check)[0] == 1:
                    task_type = 'binaryclass'
                else:
                    if max(label_row_set) == 1:
                        task_type = 'multiclass'
                    else:
                        task_type = 'multilabel'
            else:
                raise Exception('odd value exist in label value space')
        else:
            if list(label_n_check)[0] == 1:
                task_type = 'regression'
            else:
                raise Exception('odd value exist in label value space')
        return task_type

    pre_task_type = check_task_type(y, hat_y)
    if assign_task_type != None:
        if assign_task_type in ['binaryclass', 'multilabel', 'multiclass', 'regression']:
            if assign_task_type == pre_task_type:
                task_type = pre_task_type
            else:
                raise Exception('current data not support the filled task-type {0}, task-type {1} is suggested'\
                                .format(assign_task_type, pre_task_type))                
        else:
            raise Exception('fill in correct task-type [\'binaryclass\', \'multilabel\', \'multiclass\', \'regression\'], \
                                or Without fill in Anyvalue')
    else:
        task_type = pre_task_type
        
    return task_type