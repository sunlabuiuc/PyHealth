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

class conv_layer(nn.Module):

    def __init__(self, 
                 input_channel = None,
                 output_channel = None,
                 bias = False
                ):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn = nn.BatchNorm1d(output_channel)
        self.activate = nn.ReLU(inplace=True)
#        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
		
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
#        x = self.maxpool(x)
        return x

class callPredictor(nn.Module):

    def __init__(self, 
                 input_channel = 1,
                 conv_channel = [8, 8],
                 fc_size = [64, 16],
                 label_size = 1
                ):
        super(callPredictor, self).__init__()
        
        self.conv_layers = nn.ModuleList([])
        in_c = input_channel
        for out_c in conv_channel:
            self.conv_layers.append(conv_layer(in_c, out_c))
            in_c = out_c
        self.maxpool = nn.AdaptiveMaxPool1d((1))
        self.fc_layers = nn.ModuleList([])
        fc_in_d = in_c
        for fc_out_d in fc_size:
            self.fc_layers.append(nn.Linear(fc_in_d, fc_out_d))
            fc_in_d = fc_out_d
        self.label_size = label_size
        self.output_size = fc_in_d
        self.output_func = nn.Linear(self.output_size, self.label_size)

    def forward(self, input_data):
        x = input_data['X']
        x = x.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.output_func(x)
        return None, x


class CNN(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 conv_channel = [8, 8],
                 fc_size = [64, 16],
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 loss_name = 'L1LossSoftmax',
                 aggregate = 'sum',
                 optimizer_name = 'adam',
                 use_gpu = False,
                 gpu_ids = '0',
                 embed_type = 'BioCharBERT'
                 ):
        """
        Several typical & popular CNN networks for ECG prediction 


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
       
        n_epoch : int, optional (default = 100)
            number of epochs with the initial learning rate
            
        n_batchsize : int, optional (default = 5)
            batch size for model training
        
        conv_channel : list, optional (default = [8, 8, 6])
            define number of conv layer, and output channel number of each conv layer 
						
        fc_size : list, optional (default = [64, 16])
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
 
        super(CNN, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.conv_channel = conv_channel
        self.fc_size = fc_size
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.loss_name = loss_name
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
                 'input_channel': 768,
                 'conv_channel': self.conv_channel,
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
        assert isinstance(self.aggregate,str) and self.aggregate in ['sum','avg'], \
            'fill in correct aggregate (str, [\'sum\',\'avg\'])'
        assert isinstance(self.optimizer_name,str) and self.optimizer_name in ['adam'], \
            'fill in correct optimizer_name (str, [\'adam\'])'
        assert isinstance(self.use_gpu,bool), \
            'fill in correct use_gpu (bool)'
        assert isinstance(self.loss_name,str), \
            'fill in correct optimizer_name (str)'
        assert isinstance(self.conv_channel,list), \
            'fill in correct conv_channel (list, [8, 8, 6])'
        assert isinstance(self.fc_size,list), \
            'fill in correct fc_size (list, [64, 16])'
        self.device = self._get_device()

#expdata_id = '2021.0102.text'
#
#cur_dataset = expdata_generator(expdata_id, root_dir='./')
#cur_dataset.load_exp_data()
#
##_dataset = dl_reader.DatasetReader(cur_dataset.valid)
#
##print (len(_dataset.label_list))
#
#model = CNN()
#dtype = 'train'
##_loader = torch.utils.data.DataLoader(_dataset,
##                                      batch_size=3,
##                                      drop_last=True,
##                                      shuffle=True if dtype == 'train' else False)
#model.fit(cur_dataset.valid, cur_dataset.valid)
#for batch_idx, databatch in enumerate(_loader):
#    print (databatch['X'].shape)
#    print (databatch['M'].shape)
#    print (databatch['Y'].shape)
#    y = model.predictor(databatch['X'].float())
#    print (y.shape)
#    break
#
