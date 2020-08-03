__all__ = ['mortality']

import numpy as np
import pickle
import os

from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

def get_predict_results(hat_y, y):
    _hat_y = np.array([1. if hat_y[i][0]>0.5 else 0. for i in range(len(hat_y))])
    _y = y.squeeze()
    values = {}
    values['recall'] = np.sum(_hat_y * _y)/ np.sum(_y) if np.sum(_y)>0 else 0.
    values['precision'] = np.sum(_hat_y * _y)/ np.sum(_hat_y)  if np.sum(_hat_y)>0 else 0.
    values['f1_score'] = 2 * values['precision'] * values['recall'] / (values['precision'] + values['recall']) if (values['precision'] + values['recall'])>0 else 0.
    return values

def get_avg_results(hat_y, y):
    values = {}
    values['l1'] = np.sum(abs(hat_y-y))
    values['avg_precision_micro'] = average_precision_score(y, hat_y, average = 'micro')
    values['avg_precision_macro'] = average_precision_score(y, hat_y, average = 'macro')
    values['roc_auc_score_micro'] = roc_auc_score(y, hat_y, average = 'micro')
    values['roc_auc_score_macro'] = roc_auc_score(y, hat_y, average = 'macro')
    return values

def mortality(hat_y, y):
    values = get_avg_results(hat_y, y)
    values_2 = get_predict_results(hat_y, y)
    values.update(values_2)
    return values

if __name__ == '__main__':
    y = np.array([0.,1.])
    hat_y = np.array([[0.3],[0.8]])
    cm = callMetric()
    run_type = 'train'
    z = cm.measure(hat_y, y)
    print (z)