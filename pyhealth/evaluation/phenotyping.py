__all__ = ['phenotyping']

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

def get_top_k_results(hat_y, y, k=1):
    format_hat_y = np.zeros(np.shape(hat_y))
    for i in range(len(hat_y)):
        i_s = np.argsort(hat_y[i, :])[-k:]
        for j in i_s:
            format_hat_y[i, j] = 1.

    values = {}
    values['hamming_loss'+'@'+str(k)] = hamming_loss(y, format_hat_y)
#    values['jaccard_score_micro'+'@'+str(k)] = jaccard_score(y, format_hat_y, average = 'micro')
#    values['jaccard_score_macro'+'@'+str(k)] = jaccard_score(y, format_hat_y, average = 'macro')
    values['recall'+'@'+str(k)] = np.mean(np.sum(format_hat_y * y, axis=1)/ np.sum(y, axis=1))
    values['precision'+'@'+str(k)] = np.mean(np.sum(format_hat_y * y, axis=1)/ np.sum(format_hat_y, axis=1))
    return values

def get_avg_results(hat_y, y):
    values = {}
    values['avg_precision_micro'] = average_precision_score(y, hat_y, average = 'micro')
#    values['avg_precision_macro'] = average_precision_score(y, hat_y, average = 'macro')
    values['roc_auc_score_micro'] = roc_auc_score(y, hat_y, average = 'micro')
#    values['roc_auc_score_macro'] = roc_auc_score(y, hat_y, average = 'macro')
    values['coverage_error'] = coverage_error(y, hat_y)
    values['label_ranking_average_precision_score'] = label_ranking_average_precision_score(y, hat_y)
    values['label_ranking_loss'] = label_ranking_loss(y, hat_y)
    return values

def phenotyping(hat_y, y):
    values = get_avg_results(hat_y, y)
    values_2 = get_top_k_results(hat_y, y, k=1)
    values_3 = get_top_k_results(hat_y, y, k=3)
    values.update(values_2)
    values.update(values_3)
    return values

if __name__ == '__main__':
    y = np.array([[0., 1., 0.],[1., 0., 1.]])
    hat_y = np.array([[0.3, 0.7, 0.1],[0.1, 0.2, 0.8]])
    z = phenotyping(hat_y, y)
    print (z)
