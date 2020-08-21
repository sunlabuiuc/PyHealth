# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import NLLLoss
from torch.nn import BCELoss

from torch.nn import Sigmoid
from torch.nn import LogSigmoid
from torch.nn import Softmax
from torch.nn import LogSoftmax

class md_NNLoss(nn.Module):
        
    def __init__(self,
                 reduction = 'sum'):
        super(md_NNLoss, self).__init__()
        self.reduction = reduction

    def forward(self, neg_log_hat_y, y):
        if self.reduction == 'sum':
            return -1 * torch.sum(neg_log_hat_y*y)
        elif self.reduction == 'mean':
            return -1 * torch.mean(neg_log_hat_y*y)
        

loss_dict = {
    'multilabel': {
        'L1LossSigmoid': {'activate': Sigmoid, 'lossfunc': L1Loss},
        'L1LossSoftmax': {'activate': Softmax, 'lossfunc': L1Loss},
        'MSELossSigmoid': {'activate': Sigmoid, 'lossfunc': MSELoss},
        'MSELossSoftmax': {'activate': Softmax, 'lossfunc': MSELoss},
        'CELossSigmoid': {'activate': LogSigmoid, 'lossfunc': BCELoss},
        'CELossSoftmax': {'activate': LogSoftmax, 'lossfunc': BCELoss}
    },
    'multiclass' : {
        'L1LossSoftmax': {'activate': LogSoftmax, 'lossfunc': L1Loss},
        'CELossSoftmax': {'activate': LogSoftmax, 'lossfunc': md_NNLoss},
    },
    'binaryclass': {
        'L1LossSigmoid': {'activate': Sigmoid, 'lossfunc': L1Loss},
        'MSELossSigmoid': {'activate': Sigmoid, 'lossfunc': MSELoss},
        'BCELossSigmoid': {'activate': Sigmoid, 'lossfunc': BCELoss}
    }
}

class callLoss(nn.Module):
        
    def __init__(self,
                 task = 'multilabel',
                 loss_name = 'L1LossSigmoid',
                 aggregate = 'sum'):
        super(callLoss, self).__init__()
        self.loss_fn = loss_dict[task][loss_name]['lossfunc'](reduction = aggregate)
        if 'Softmax' in loss_name:
            self.activate_fn = loss_dict[task][loss_name]['activate'](dim=-1)
        else:
            self.activate_fn = loss_dict[task][loss_name]['activate']()

    def forward(self, data):
        """
        
        Parameters
        
        ----------
        data = {
                  'hat_y': shape (batchsize, n_label)
                  
                  'y': shape (batchsize, n_label)
                  
                  'mask': [optional] shape (batchsize, n_timestep)
                      when target_repl is True
 
                  'all_hat_y': [optional] shape (batchsize, n_timestep, n_label) 
                      when target_repl is True
               }
  
  
        """
        
        hat_y, y = data['hat_y'], data['y']
        n_sample, n_label = y.size()
        y = y.view(-1, n_label)
        hat_y = hat_y.view(-1, n_label)
        hat_y = self.activate_fn(hat_y)
        single_loss = self.loss_fn(hat_y, y)
        loss = single_loss
        return loss
