import numpy as np
import pickle
import os

from .binaryclass import evaluator as binary_eval
from .multilabel import evaluator as multilabel_eval

def check_evalu_type(hat_y, y):
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
                evalu_type = 'binaryclass'
            else:
                if max(label_row_set) == 1:
                    evalu_type = 'multiclass'
                else:
                    evalu_type = 'multilabel'
        else:
            raise Exception('odd value exist in label value space')
    else:
        if list(label_n_check)[0] == 1:
            evalu_type = 'regression'
        else:
            raise Exception('odd value exist in label value space')
    return evalu_type

evalu_func_mapping_dict = {
    'binaryclass': binary_eval, 
    'multilabel': multilabel_eval, 
    'multiclass': None, 
    'regression': None
}

def func(hat_y, y, evalu_type = None):
    pre_evalu_type = check_evalu_type(hat_y, y)
    if evalu_type != None:
        if evalu_type is ['binary', 'multilabel', 'multiclass', 'regression']:
            if evalu_type == pre_evalu_type:
                evalu_type = pre_evalu_type
            else:
                raise Exception('current data not support the filled evaluation-type {0}, evaluation-type {1} is suggested'\
                                .format(evalu_type, pre_evalu_type))                
        else:
            raise Exception('fill in correct evaluation type [\'binary\', \'multilabel\', \'multiclass\', \'regression\'], \
                                or Without fill in Anyvalue')
    else:
        evalu_type = pre_evalu_type
    print ('current data evaluate using {0} evaluation-type'.format(evalu_type))
    evalu_func = evalu_func_mapping_dict[evalu_type]
    return evalu_func(hat_y, y)

if __name__ == '__main__':
    y = np.array([0.,1.])
    hat_y = np.array([[0.3],[0.8]])
    z = func(hat_y, y)
    print (z)
    y = np.array([[0., 1., 0.],[1., 0., 1.]])
    hat_y = np.array([[0.3, 0.7, 0.1],[0.1, 0.2, 0.8]])
    z = func(hat_y, y)
    print (z)
