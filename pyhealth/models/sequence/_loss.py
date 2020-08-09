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

loss_dict = {
    'multilabel': {
        'L1LossSigmoid': {'activate': Sigmoid, 'lossfunc': L1Loss},
        'L1LossSoftmax': {'activate': Softmax, 'lossfunc': L1Loss},
        'MSELossSigmoid': {'activate': Sigmoid, 'lossfunc': MSELoss},
        'MSELossSoftmax': {'activate': Softmax, 'lossfunc': MSELoss},
        'CELossSigmoid': {'activate': LogSigmoid, 'lossfunc': BCELoss},
        'CELossSoftmax': {'activate': LogSoftmax, 'lossfunc': BCELoss}
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
                 target_repl = False,
                 target_repl_coef = 0,
                 aggregate = 'sum'):
        super(callLoss, self).__init__()
        self.loss_fn = loss_dict[task][loss_name]['lossfunc'](reduction = aggregate)
        if 'Softmax' in loss_name:
            self.activate_fn = loss_dict[task][loss_name]['activate'](dim=-1)
        else:
            self.activate_fn = loss_dict[task][loss_name]['activate']()
        self.target_repl = target_repl 
        self.target_repl_coef = target_repl_coef

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
        if self.target_repl:
            all_hat_y, mask = data['all_hat_y'], data['mask']
            _, n_timestep = mask.size()
            mask_flag = mask.unsqueeze(-1)
            all_hat_y = self.activate_fn(all_hat_y) * mask_flag
            all_hat_y = all_hat_y.view(-1, n_label)
            all_y = y.unsqueeze(1).repeat(1,n_timestep,1) * mask_flag
            all_y = all_y.view(-1, n_label)
            all_loss = self.loss_fn(all_hat_y, all_y)
            loss = (1-self.target_repl_coef) * single_loss + self.target_repl_coef * all_loss
        else:
            loss = single_loss
        return loss
