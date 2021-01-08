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
from torch.nn.init import xavier_uniform_ as xavier_uniform
warnings.filterwarnings('ignore')

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

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
		
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class callPredictor(nn.Module):

    def __init__(self, 
                 input_channel = 1,
                 conv_channel = [8, 8],
                 fc_size = [64, 16],
                 filer_list = [3,5,9],
                 label_size = 1
                ):
        super(callPredictor, self).__init__()
        
        self.label_size = label_size
        self.output_size = conv_channel[-1]
        self.output_func = nn.Linear(len(filer_list) * self.output_size, self.label_size)
        self.weights_f = nn.Linear(len(filer_list) * self.output_size, 1)
        self.conv_set = nn.ModuleList()

        for filter_size in filer_list:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(input_channel, input_channel, kernel_size=filter_size, padding=int(filter_size / 2))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)
            input_size = input_channel
            for idx, output_size in enumerate(conv_channel):
                tmp = ResidualBlock(input_size, output_size, filter_size, 1, True, True)
                one_channel.add_module('resconv-{}'.format(idx), tmp)
                input_size = output_size
            self.conv_set.add_module('channel-{}'.format(filter_size), one_channel)
        
    def forward(self, input_data):
        conv_result = []
        n_case, n_visit, n_feat = input_data['X'].shape
        x = input_data['X'].permute(0,2,1)
        for conv in self.conv_set:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:

                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        weights = self.weights_f(x.reshape(n_case * n_visit, -1)).reshape(n_case, n_visit)
        weights = torch.exp(weights) * input_data['M']
        weights = weights/torch.sum(weights, -1).reshape(n_case, 1)
        ensemble_f = torch.sum(torch.unsqueeze(weights, -1) * x, 1)
        hat_y = self.output_func(ensemble_f)
        return None, hat_y


class MultiResCNN(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 conv_channel = [8, 8],
                 fc_size = [64, 16],
                 filer_list = [3,5,9],
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
        
        filer_list : list, optional (default = [3,5,9])
            
            
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
 
        super(MultiResCNN, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.conv_channel = conv_channel
        self.fc_size = fc_size
        self.filer_list = filer_list
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
                 'filer_list': self.filer_list,
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
        assert isinstance(self.filer_list,list), \
            'fill in correct filer_list (list, [3,5,9])'
        self.device = self._get_device()



#expdata_id = '2021.0102.text'
#
#cur_dataset = expdata_generator(expdata_id, root_dir='./')
#cur_dataset.load_exp_data()
#
#_dataset = dl_reader.DatasetReader(cur_dataset.valid, 'BioCharBERT')
#
##print (len(_dataset.label_list))
#
#model = MultiResCNN()
#dtype = 'train'
#_loader = torch.utils.data.DataLoader(_dataset,
#                                      batch_size=3,
#                                      drop_last=True,
#                                      shuffle=True if dtype == 'train' else False)
#model.fit(cur_dataset.valid, cur_dataset.valid)
#for batch_idx, databatch in enumerate(_loader):
#    print (databatch['X'].shape)
#    print (databatch['M'].shape)
#    print (databatch['Y'].shape)
#    _, y = model.predictor({key: value.float() for key, value in databatch.items()})
#    print (y.shape)
#    break


