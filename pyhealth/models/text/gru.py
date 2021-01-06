# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os
import torch
import torch.nn as nn
import pickle
import warnings
import torchvision.models as models
from pyhealth.models.text._loss import callLoss
from pyhealth.models.text._dlbase import BaseControler
from pyhealth.data.data_reader.text import flatten_dl_reader as dl_reader
from pyhealth.data.expdata_generator import textdata as expdata_generator

warnings.filterwarnings('ignore')

class callPredictor(nn.Module):
    def __init__(self, 
                 input_size = None,
                 layer_hidden_sizes = [10,20,15],
                 num_layers = 3,
                 bias = True,
                 dropout = 0.5,
                 bidirectional = True,
                 batch_first = True,
                 label_size = 1):
        super(callPredictor, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size' 
        self.num_layers = num_layers
        self.rnn_models = nn.ModuleList([])
        if bidirectional:
            layer_input_sizes = [input_size] + [2 * chs for chs in layer_hidden_sizes]
        else:
            layer_input_sizes = [input_size] + layer_hidden_sizes
        for i in range(num_layers):
            self.rnn_models.append(nn.GRU(input_size = layer_input_sizes[i],
                                     hidden_size = layer_hidden_sizes[i],
                                     num_layers = num_layers,
                                     bias = bias,
                                     dropout = dropout,
                                     bidirectional = bidirectional,
                                     batch_first = batch_first))
        self.label_size = label_size
        self.output_size = layer_input_sizes[-1]
        self.output_func = nn.Linear(self.output_size, self.label_size)
            
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
        _data = X
        for temp_rnn_model in self.rnn_models:
            _data, _ = temp_rnn_model(_data)
        outputs = _data
        all_output = outputs * M.unsqueeze(-1)
        n_batchsize, n_timestep, n_featdim = all_output.shape
        all_output = self.output_func(outputs.reshape(n_batchsize*n_timestep, n_featdim)).reshape(n_batchsize, n_timestep, self.label_size)
        cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output

class GRU(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 layer_hidden_sizes = [10,20,15],
                 bias = True,
                 dropout = 0.5,
                 bidirectional = True,
                 batch_first = True,
                 loss_name = 'L1LossSigmoid',
                 target_repl = False,
                 target_repl_coef = 0.,
                 aggregate = 'sum',
                 optimizer_name = 'adam',
                 use_gpu = False,
                 gpu_ids = '0',
                 embed_type = 'BioCharBERT'
                 ):
        """
        Applies a multi-layer Gated recurrent unit (GRU) RNN to an healthcare text data.


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
            
        layer_hidden_sizes : list, optional (default = [10,20,15])
            The number of features of the hidden state h of each layer
            
        bias : bool, optional (default = True)
            If False, then the layer does not use bias weights b_ih and b_hh. 
            
        dropout : float, optional (default = 0.5)
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, 
            with dropout probability equal to dropout. 

        bidirectional : bool, optional (default = True)
            If True, becomes a bidirectional GRU. 
            
        batch_first : bool, optional (default = False)
            If True, then the input and output tensors are provided as (batch, seq, feature). 
             
        loss_name : str, optional (default='SigmoidCELoss') 
            Name or objective function.

        use_gpu : bool, optional (default=False) 
            If yes, use GPU recources; else use CPU recources 

				gpu_ids : str, optional (default='') 
										If yes, assign concrete used gpu ids such as '0,2,6'; else use '0' 


        """
 
        super(GRU, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.layer_hidden_sizes = layer_hidden_sizes
        self.num_layers = len(layer_hidden_sizes)
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.loss_name = loss_name
        self.target_repl = target_repl
        self.target_repl_coef = target_repl_coef
        self.aggregate = aggregate
        self.optimizer_name = optimizer_name
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.embed_type = embed_type
        self._args_check()
 
    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        if self.is_loadmodel is False:  
            _config = {
                'input_size': 768,
                'layer_hidden_sizes': self.layer_hidden_sizes,
                'num_layers': self.num_layers,
                'bias': self.bias,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'batch_first': self.batch_first,
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
        _dataset = dl_reader.DatasetReader(data, device = self.device, embed_type = self.embed_type)            
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
        self._build_model()
        train_reader = self._get_reader(train_data, 'train')
        valid_reader = self._get_reader(valid_data, 'valid')
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
        assert isinstance(self.n_epoch,int) and self.n_epoch>0, \
            'fill in correct n_epoch (int, >0)'
        assert isinstance(self.learn_ratio,float) and self.learn_ratio>0., \
            'fill in correct learn_ratio (float, >0.)'
        assert isinstance(self.weight_decay,float) and self.weight_decay>=0., \
            'fill in correct weight_decay (float, >=0.)'
        assert isinstance(self.n_epoch_saved,int) and self.n_epoch_saved>0 and self.n_epoch_saved < self.n_epoch, \
            'fill in correct n_epoch (int, >0 and <{0}).format(self.n_epoch)'
        assert isinstance(self.layer_hidden_sizes,list) and len(self.layer_hidden_sizes)>0, \
            'fill in correct layer_hidden_sizes (list, such as [10,20,15])'
        assert isinstance(self.num_layers,int) and self.num_layers>0, \
            'fill in correct num_layers (int, >0)'
        assert isinstance(self.bias,bool), \
            'fill in correct bias (bool)'
        assert isinstance(self.dropout,float) and self.dropout>0. and self.dropout<1., \
            'fill in correct learn_ratio (float, >0 and <1.)'
        assert isinstance(self.bidirectional,bool), \
            'fill in correct bidirectional (bool)'
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


#expdata_id = '2021.0102.text'
#
#cur_dataset = expdata_generator(expdata_id, root_dir='./')
#cur_dataset.load_exp_data()
#
#_dataset = dl_reader.DatasetReader(cur_dataset.valid)
#
##print (len(_dataset.label_list))
#
#model = GRU()
#dtype = 'train'
#_loader = torch.utils.data.DataLoader(_dataset,
#                                      batch_size=3,
#                                      drop_last=True,
#                                      shuffle=True if dtype == 'train' else False)
#model.fit(cur_dataset.valid, cur_dataset.valid)
#for batch_idx, databatch in enumerate(_loader):
#    for key, value in databatch.items():
#        print (key, value.shape)
#    y, cur_y = model.predictor({key: value.float() for key, value in databatch.items()})
#    print (y.shape, cur_y.shape)
#    break


