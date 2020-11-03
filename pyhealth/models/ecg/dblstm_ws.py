# -*- coding: utf-8 -*-

# License: BSD 2 clause

import os
import torch
import torch.nn as nn
import pickle
import warnings
import torchvision.models as models
from torch.nn import LSTM
from ._loss import callLoss
from ._dlbase import BaseControler
from pyhealth.data.data_reader.ecg import dblstm_ws_reader

warnings.filterwarnings('ignore')

class callPredictor(nn.Module):

    def __init__(self, 
                 n_level,
                 input_size,
                 hidden_sizes = [64, 32],
                 fc_size = [128],
                 label_size = 4
                ):
        super(callPredictor, self).__init__()
        self.rnn_layers = nn.ModuleList([])
        in_f = input_size
        for unit_f in hidden_sizes:
            self.rnn_layers.append(LSTM(in_f, unit_f, bias = True, bidirectional = True))
            in_f = unit_f * 2
        self.fc_layers = nn.ModuleList([])
        fc_in_d = in_f * (n_level+2)
        for fc_out_d in fc_size:
            self.fc_layers.append(nn.Linear(fc_in_d, fc_out_d))
            fc_in_d = fc_out_d
        self.label_size = label_size
        self.output_size = fc_in_d
        self.output_func = nn.Linear(self.output_size, self.label_size)

    def forward(self, x):
        for rnn_layer in self.rnn_layers:
            x, _ = rnn_layer(x)
        x = torch.flatten(x, start_dim=1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.output_func(x)
        return x

class DBLSTM_WS(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_level = 4,
                 n_epoch = 100,
                 n_batchsize = 5,
                 hidden_sizes = [64, 32],
                 fc_size = [128],
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 loss_name = 'L1LossSoftmax',
                 aggregate = 'sum',
                 optimizer_name = 'adam',
                 use_gpu = False,
                 gpu_ids = '0'
                 ):

        """
 
        DBLSTM-WS: A deep bidirectional LSTM network based on wavelet sequences for 
            classifying electrocardiogram (ECG) signals. 


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment

        n_level: int, optional (default = 4)
            number of levels for level-filter
 
        n_epoch : int, optional (default = 100)
            number of epochs with the initial learning rate
            
        n_batchsize : int, optional (default = 5)
            batch size for model training
        
        hidden_sizes : list, optional (default = [64, 32])
            define number of hidden unit of each bi-lstm layer 

        fc_size : list, optional (default = [128])
            define number of fc layer, and output feature dim of each fc layer 

        learn_ratio : float, optional (default = 1e-4)
            initial learning rate for adam
  
        weight_decay : float, optional (default = 1e-4)
            weight decay (L2 penalty)
  
        n_epoch_saved : int, optional (default = 1)
            frequency of saving checkpoints at the end of epochs

        loss_name : str, optional (default='SigmoidCELoss') 
            Name or objective function.

        use_gpu : bool, optional (default=False) 
            If yes, use GPU recources; else use CPU recources 

            gpu_ids : str, optional (default='') 
                    If yes, assign concrete used gpu ids such as '0,2,6'; else use '0' 

        """
 
        super(DBLSTM_WS, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.n_level = n_level
        self.hidden_sizes = hidden_sizes
        self.fc_size = fc_size
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.loss_name = loss_name
        self.aggregate = aggregate
        self.optimizer_name = optimizer_name
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self._args_check()
 
    def _build_model(self):

        """
        
        Build the crucial components for model training 
 
        
        """
        if self.is_loadmodel is False:        
            _config = {
                 'n_level': self.n_level,
                 'input_size': self.input_size,
                 'hidden_sizes': self.hidden_sizes,
                 'fc_size': self.fc_size,
                 'label_size': self.label_size
                }
            self.predictor = callPredictor(**_config).to(self.device)
            self._save_predictor_config(_config)
            
        if self.dataparallal:
            self.predictor= torch.nn.DataParallel(self.predictor)
        self.criterion = callLoss(task = self.task_type,
                                  loss_name = self.loss_name,
                                  aggregate = self.aggregate)
        self.optimizer = self._get_optimizer(self.optimizer_name)

    def _get_reader(self, data, dtype = 'train'):
        """
        Parameters

        ----------

        data : {
                  'x':list[episode_file_path], 
                  'y':list[label], 
                  'l':list[seq_len], 
                  'feat_n': n of feature space, 
                  'label_n': n of label space
               }

            The input samples dict.
 
        dtype: str, (default='train')
        
            dtype in ['train','valid','test'], different type imapct whether use shuffle for data
 
        Return
        
        ----------
        
        data_loader : dataloader of input data dict
        
            Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

            refer to torch.utils.data.dataloader
        
        """
        _dataset = dblstm_ws_reader.DatasetReader(data)            
        self.input_size = _dataset.n_feat
        _loader = torch.utils.data.DataLoader(_dataset,
                                              batch_size=self.n_batchsize,
                                              drop_last = True,
                                              shuffle=True if dtype == 'train' else False)
        return _loader


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
        train_reader = self._get_reader(train_data, 'train')
        valid_reader = self._get_reader(valid_data, 'valid')
        self._build_model()
        self._fit_model(train_reader, valid_reader)
  
    def load_model(self, 
                   loaded_epoch = '',
                   config_file_path = '',
                   model_file_path = ''):

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

        predictor_config = self._load_predictor_config(config_file_path)
        self.predictor = callPredictor(**predictor_config).to(self.device)
        self._load_model(loaded_epoch, model_file_path)
 

    def _args_check(self):

        """
        
        Check args whether valid/not and give tips
 
        
        """
        assert isinstance(self.n_batchsize,int) and self.n_batchsize>0, \
            'fill in correct n_batchsize (int, >0)'
        assert isinstance(self.n_level,int) and self.n_level>0, \
            'fill in correct n_level (int, >0)' 
        assert isinstance(self.n_epoch,int) and self.n_epoch>0, \
            'fill in correct n_epoch (int, >0)'
        assert isinstance(self.learn_ratio,float) and self.learn_ratio>0., \
            'fill in correct learn_ratio (float, >0.)'
        assert isinstance(self.weight_decay,float) and self.weight_decay>=0., \
            'fill in correct weight_decay (float, >=0.)'
        assert isinstance(self.n_epoch_saved,int) and self.n_epoch_saved>0 and self.n_epoch_saved < self.n_epoch, \
            'fill in correct n_epoch (int, >0 and <{0}).format(self.n_epoch)'
        assert isinstance(self.aggregate,str) and self.aggregate in ['sum','avg'], \
            'fill in correct aggregate (str, [\'sum\',\'avg\'])'
        assert isinstance(self.optimizer_name,str) and self.optimizer_name in ['adam'], \
            'fill in correct optimizer_name (str, [\'adam\'])'
        assert isinstance(self.use_gpu,bool), \
            'fill in correct use_gpu (bool)'
        assert isinstance(self.loss_name,str), \
            'fill in correct optimizer_name (str)'
        assert isinstance(self.hidden_sizes,list), \
            'fill in correct hidden_sizes (list, [64, 32])'
        assert isinstance(self.fc_size,list), \
            'fill in correct fc_size (list, [128])'
        self.device = self._get_device()
