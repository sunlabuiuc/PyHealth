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
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm
import torch.nn.functional as F
warnings.filterwarnings('ignore')


class OutputLayer(nn.Module):

    def __init__(self, input_size, label_size):
        super(OutputLayer, self).__init__()
        self.U = nn.Linear(input_size, label_size)
        self.final = nn.Linear(input_size, label_size)
        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.final.weight)

    def forward(self, x):
        att = self.U.weight.matmul(x.transpose(1, 2))
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x)
        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return logits

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.conv1.weight)
        xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            xavier_uniform_(self.downsample.weight)
            
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class callPredictor(nn.Module):

    def __init__(self, 
                 input_channel = 1,
                 nhid = 8,
                 n_level = 8,
                 kernel_size = 2,
                 hidden_size = 8,
                 label_size = 1
                ):
        super(callPredictor, self).__init__()
        
        num_chans = [nhid] * n_level
        self.tcn = TemporalConvNet(input_channel, num_chans, kernel_size, True)
        self.lin = nn.Linear(num_chans[-1], hidden_size)
        self.output_layer = OutputLayer(hidden_size, label_size)
        xavier_uniform_(self.lin.weight)

    def forward(self, input_data):
        conv_result = []
        n_case, n_visit, n_feat = input_data['X'].shape
        x = input_data['X']
        hid_seq = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # [bs, seq_len, nhid]
        hid_seq = F.relu(self.lin(hid_seq))
        logits = self.output_layer(hid_seq)
        return None, logits


class DCAN(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 nhid = 8,
                 n_level = 8,
                 kernel_size = 2,
                 hidden_size = 8,                 
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

        DCAN: Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
       
        n_epoch : int, optional (default = 100)
            number of epochs with the initial learning rate
            
        n_batchsize : int, optional (default = 5)
            batch size for model training
        
        nhid: int, optional (default = 5)
            
            
        n_level: int, optional (default = 5)
            

        kernel_size: int, optional (default = 2)
            
            
        hidden_size: int, optional (default = 8)
            
            
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
 
        super(DCAN, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.nhid = nhid
        self.n_level = n_level
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
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
                 'nhid': self.nhid,
                 'n_level': self.n_level,
                 'kernel_size': self.kernel_size,
                 'hidden_size': self.hidden_size,
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
            'fill in correct loss_name (int)'
        assert isinstance(self.nhid,int), \
            'fill in correct nhid (int)'
        assert isinstance(self.n_level,int), \
            'fill in correct n_level (int)'
        assert isinstance(self.kernel_size,int), \
            'fill in correct kernel_size (int)'
        assert isinstance(self.hidden_size,int), \
            'fill in correct hidden_size (int)'

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
#model = DCAN()
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

