# -*- coding: utf-8 -*-

"""StageNet model. Adapted and modified from

https://github.com/v1xerunt/StageNet

"""
import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import pickle
import warnings
from ._loss import callLoss
from ._dlbase import BaseController

warnings.filterwarnings('ignore')

class callPredictor(nn.Module):

    def __init__(self, 
                 input_dim = None, 
                 hidden_dim = 384, 
                 conv_size = 10, 
                 levels = 3, 
                 dropconnect = 0.3, 
                 dropout = 0.3, 
                 dropres = 0.3,
                 label_size = None,
                 device = None
                 ):

        super(callPredictor, self).__init__()

        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = label_size
        self.levels = levels
        self.chunk_size = hidden_dim // levels
        self.device = device

        self.kernel = nn.Linear(int(input_dim+1), int(hidden_dim*4+levels*2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(int(hidden_dim+1), int(hidden_dim*4+levels*2))
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)
        self.nn_output = nn.Linear(int(self.conv_dim), int(self.output_dim))

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

    def cumax(self, x, mode='l2r'):

        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval):

        x_in = inputs
        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')
        f_master_gate = f_master_gate.unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels*2], 'r2l')
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels*2:]
        x_out = x_out.reshape(-1, self.levels*4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels*2])
        o_gate = torch.sigmoid(x_out[:, self.levels*2:self.levels*3])
        c_in = torch.tanh(x_out[:, self.levels*3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (i_master_gate - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, input_data):
        
        """
        
        Parameters
        
        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                      'cur_M': shape (batchsize, n_timestep)
                      'T': shape (batchsize, n_timestep)
                     }
        
        Return
        
        ----------
        
        all_output, shape (batchsize, n_timestep, n_labels)
            
            predict output of each time step
            
        cur_output, shape (batchsize, n_labels)
        
            predict output of last time step

        
        """
        X = input_data['X']
        M = input_data['M']
        cur_M = input_data['cur_M']
        T = input_data['T']

        batch_size, time_step, feature_dim = X.size()
        c_out = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        h_out = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size, batch_size, self.hidden_dim).to(self.device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(self.device)
        h = []
        origin_h = []
        distance = []

        for t in range(time_step):
            out, c_out, h_out = self.step(X[:, t, :], c_out, h_out, T[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim+self.levels], -1)
            cur_distance_in = torch.mean(out[..., self.hidden_dim+self.levels:], -1)
            origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)
            distance.append(cur_distance)
            #Re-weighted convolution operation
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)
            #Re-calibrate Progression patterns
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme)
            local_theme = torch.sigmoid(local_theme)
            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)  

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)
        output = self.nn_output(rnn_outputs)
        output = output.contiguous().view(batch_size, time_step, self.output_dim)
        all_output = torch.sigmoid(output)
        cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output

class StageNet(BaseController):
    
    """
    
    StageNet: Stage-Aware Neural Networks for Health Risk Prediction.
    
    """
    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 hidden_size = 384, 
                 conv_size = 10, 
                 levels = 3, 
                 dropconnect = 0.3, 
                 dropout = 0.3, 
                 dropres = 0.3,
                 batch_first = True,
                 loss_name = 'L1LossSigmoid',
                 target_repl = False,
                 target_repl_coef = 0.,
                 aggregate = 'sum',
                 optimizer_name = 'adam',
                 use_gpu = False,
                 gpu_ids = '0'
                 ):
        """
        Applies an Attention-based Bidirectional Recurrent Neural Networks for an healthcare data sequence


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
            
        n_epoch : int, optional (default = 100)
            number of epochs with the initial learning rate
            
        n_batchsize : int, optional (default = 5)
            batch size for model training
   
        learn_ratio : float, optional (default = 1e-4)
            initial learning rate for adam
  
        weight_decay : float, optional (default = 1e-4)
            weight decay (L2 penalty)
  
        n_epoch_saved : int, optional (default = 1)
            frequency of saving checkpoints at the end of epochs
                   
        hidden_size : int, optional (default = 384)

        conv_size : int, optional (default = 10) 

        levels : int, optional (default = 3)

        dropconnect : int, optional (default = 0.3) 

        dropout : int, optional (default = 0.3) 

        dropres : int, optional (default = 0.3)

        batch_first : bool, optional (default = False)
            If True, then the input and output tensors are provided as (batch, seq, feature). 
             
        loss_name : str, optional (default='SigmoidCELoss') 
            Name or objective function.

        use_gpu : bool, optional (default=False) 
            If yes, use GPU recources; else use CPU recources 

				gpu_ids : str, optional (default='') 
										If yes, assign concrete used gpu ids such as '0,2,6'; else use '0' 

        """
 
        super(StageNet, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.hidden_size = hidden_size 
        self.conv_size = conv_size
        self.levels = levels
        self.dropconnect = dropconnect 
        self.dropout = dropout 
        self.dropres = dropres
        self.batch_first = batch_first
        self.loss_name = loss_name
        self.target_repl = target_repl
        self.target_repl_coef = target_repl_coef
        self.aggregate = aggregate
        self.optimizer_name = optimizer_name
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self._args_check()
        
    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        
        _config = {
             'input_dim' : self.input_size, 
             'hidden_dim' : self.hidden_size, 
             'conv_size' : self.conv_size, 
             'levels' : self.levels, 
             'dropconnect' : self.dropconnect, 
             'dropout' : self.dropout, 
             'dropres' : self.dropres,
             'label_size' : self.label_size,
             'device' : self.device
            }

        self.predictor = callPredictor(**_config).to(self.device)
        if self.dataparallal:
            self.predictor= torch.nn.DataParallel(self.predictor)
        self._save_predictor_config({key: value for key, value in _config.items() if key != 'device'})
        self.criterion = callLoss(task = self.task_type,
                                  loss_name = self.loss_name,
                                  target_repl = self.target_repl,
                                  target_repl_coef = self.target_repl_coef,
                                  aggregate = self.aggregate)
        self.optimizer = self._get_optimizer(self.optimizer_name)

    def fit(self, train_data, valid_data, assign_task_type = None):
        
        """
        Parameters

        ----------

        train_data : {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                      }

            The input train samples dict.
 
        valid_data : {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                      }

            The input valid samples dict.

        assign_task_type: str (default = None)
            predifine task type to model mapping <feature, label>
            current support ['binary','multiclass','multilabel','regression']

        Returns

        -------

        self : object

            Fitted estimator.

        """
        self.task_type = assign_task_type
        self._data_check([train_data, valid_data])
        self._build_model()
        train_reader = self._get_reader(train_data, 'train')
        valid_reader = self._get_reader(valid_data, 'valid')
        self._fit_model(train_reader, valid_reader)
  
    def load_model(self, loaded_epoch = ''):
        """
        Parameters

        ----------

        loaded_epoch : str, loaded model name 
        
            we save the model by <epoch_count>.epoch, latest.epoch, best.epoch

        Returns

        -------

        self : object

            loaded estimator.

        """

        predictor_config = self._load_predictor_config()
        predictor_config['device'] = self.device
        self.predictor = callPredictor(**predictor_config).to(self.device)
        self._load_model(loaded_epoch)
 

    def _args_check(self):
        """
        
        Check args whether valid/not and give tips
 
        
        """
        assert isinstance(self.n_batchsize,int) and self.n_batchsize>0, \
            'fill in correct n_batchsize (int, >0)'
        assert isinstance(self.n_epoch,int) and self.n_epoch>0, \
            'fill in correct n_epoch (int, >0)'
        assert isinstance(self.learn_ratio,float) and self.learn_ratio>0., \
            'fill in correct learn_ratio (float, >0.)'
        assert isinstance(self.weight_decay,float) and self.weight_decay>=0., \
            'fill in correct weight_decay (float, >=0.)'
        assert isinstance(self.n_epoch_saved,int) and self.n_epoch_saved>0 and self.n_epoch_saved < self.n_epoch, \
            'fill in correct n_epoch (int, >0 and <{0}).format(self.n_epoch)'
        assert isinstance(self.hidden_size,int) and self.hidden_size>0, \
            'fill in correct hidden_size (int, 8)'
        assert isinstance(self.conv_size,int) and self.conv_size>0, \
            'fill in correct conv_size (int, 10)'
        assert isinstance(self.levels,int) and self.levels>0, \
            'fill in correct levels (int, 10)'
        assert isinstance(self.dropconnect,float) and self.dropconnect>=0. and self.dropconnect<1., \
            'fill in correct dropconnect (float, >=0 and <1.)'
        assert isinstance(self.dropout,float) and self.dropout>=0. and self.dropout<1., \
            'fill in correct dropout (float, >=0 and <1.)'
        assert isinstance(self.dropres,float) and self.dropres>=0. and self.dropres<1., \
            'fill in correct dropres (float, >=0 and <1.)'
        assert isinstance(self.batch_first,bool), \
            'fill in correct batch_first (bool)'
        assert isinstance(self.target_repl,bool), \
            'fill in correct target_repl (bool)'
        assert isinstance(self.target_repl_coef,float) and self.target_repl_coef>=0. and self.target_repl_coef<=1., \
            'fill in correct target_repl_coef (float, >=0 and <=1.)'
        assert isinstance(self.aggregate,str) and self.aggregate in ['sum','avg'], \
            'fill in correct aggregate (str, [\'sum\',\'avg\'])'
        assert isinstance(self.optimizer_name,str) and self.optimizer_name in ['adam'], \
            'fill in correct optimizer_name (str, [\'adam\'])'
        assert isinstance(self.use_gpu,bool), \
            'fill in correct use_gpu (bool)'
        assert isinstance(self.loss_name,str), \
            'fill in correct optimizer_name (str)'
        assert isinstance(self.gpu_ids,str), \
            'fill in correct use_gpu (str, \'0,2,7\')'
        self.device = self._get_device()
