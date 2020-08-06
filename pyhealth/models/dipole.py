import os
import torch
import torch.nn as nn
import pickle
import warnings
from ..utils.loss import callLoss
from .base import BaseControler

warnings.filterwarnings('ignore')

class LocationAttention(nn.Module):

    def __init__(self, hidden_size):
        super(LocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_value_ori_func = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>         
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>       
        reshape_feat = input_data.reshape(n_batch*n_seq, hidden_size)
        # shape of attention_value_ori: <n_batch*n_seq, 1>       
        attention_value_ori = torch.exp(self.attention_value_ori_func(reshape_feat))
        # shape of attention_value_format: <n_batch, 1, n_seq>       
        attention_value_format = attention_value_ori.reshape(n_batch, n_seq).unsqueeze(1)        
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0 
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 1).permute(1, 0).unsqueeze(0)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value/accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output

class GeneralAttention(nn.Module):

    def __init__(self, hidden_size):
        super(GeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.correlated_value_ori_func = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>         
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>       
        reshape_feat = input_data.reshape(n_batch*n_seq, hidden_size)
        # shape of correlated_value_ori: <n_batch, n_seq, hidden_size>       
        correlated_value_ori = self.correlated_value_ori_func(reshape_feat).reshape(n_batch, n_seq, hidden_size)
        # shape of _extend_correlated_value_ori: <n_batch, n_seq, 1, hidden_size>   
        _extend_correlated_value_ori = correlated_value_ori.unsqueeze(-2)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>   
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _extend_input_data: <n_batch, n_seq, n_seq, hidden_size> 
        _correlat_value = _extend_correlated_value_ori * _extend_input_data
        # shape of attention_value_format: <n_batch, n_seq, n_seq>       
        attention_value_format = torch.exp(torch.sum(_correlat_value, dim = -1))
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0 
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 1).permute(1, 0).unsqueeze(0)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value/accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output

class ConcatenationAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim = 16):
        super(ConcatenationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim        
        self.attention_map_func = nn.Linear(2 * self.hidden_size, self.attention_dim)
        self.activate_func = nn.Tanh()
        self.correlated_value_ori_func = nn.Linear(self.attention_dim, 1)
        
    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>         
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of _extend_input_data: <n_batch, n_seq, 1, hidden_size>       
        _extend_input_data_f = input_data.unsqueeze(-2)
        # shape of _repeat_extend_correlated_value_ori: <n_batch, n_seq, n_seq, hidden_size>   
        _repeat_extend_input_data_f = _extend_input_data_f.repeat(1,1,n_seq,1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>   
        _extend_input_data_b = input_data.unsqueeze(1)
        # shape of _repeat_extend_input_data: <n_batch, n_seq, n_seq, hidden_size>   
        _repeat_extend_input_data_b = _extend_input_data_b.repeat(1,n_seq,1,1)
        # shape of _concate_value: <n_batch, n_seq, n_seq, 2 * hidden_size>           
        _concate_value = torch.cat([_repeat_extend_input_data_f, _repeat_extend_input_data_b], dim = -1)        
        # shape of _correlat_value: <n_batch, n_seq, n_seq> 
        _correlat_value = self.activate_func(self.attention_map_func(_concate_value.reshape(-1, 2 * hidden_size)))
        _correlat_value = self.correlated_value_ori_func(_correlat_value).reshape(n_batch, n_seq, n_seq)
        # shape of attention_value_format: <n_batch, n_seq, n_seq>       
        attention_value_format = torch.exp(_correlat_value)
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0 
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 1).permute(1, 0).unsqueeze(0)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value/accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output

class callPredictor(nn.Module):
    def __init__(self, 
                 input_size = None,
                 embed_size = 16,
                 hidden_size = 8,
                 output_size = 10,
                 bias = True,
                 dropout = 0.5,
                 batch_first = True,
                 label_size = 1,
                 attention_type = 'location_based',
                 attention_dim = 8):
        super(callPredictor, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size' 
 
        self.input_size = input_size        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_size = label_size

        self.embed_func = nn.Linear(self.input_size, self.embed_size)
        self.rnn_model = nn.GRU(input_size = embed_size,
                                 hidden_size = hidden_size,
                                 bias = bias,
                                 dropout = dropout,
                                 bidirectional = True,
                                 batch_first = batch_first)
        if attention_type == 'location_based':
            self.attention_func = LocationAttention(2*hidden_size)
        elif attention_type == 'general':
            self.attention_func = GeneralAttention(2*hidden_size)
        elif attention_type == 'concatenation_based':
            self.attention_func = ConcatenationAttention(2*hidden_size, attention_dim = attention_dim)
        else:
            raise Exception('fill in correct attention_type, [location_based, general, concatenation_based]')
        self.output_func = nn.Linear(4 * hidden_size, self.output_size) 
        self.output_activate = nn.Tanh()
        self.predict_func = nn.Linear(self.output_size, self.label_size)
            
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
        batchsize, n_timestep, n_orifeatdim = X.shape
        _ori_X = X.view(-1, n_orifeatdim)
        _embed_X = self.embed_func(_ori_X)
        _embed_X = _embed_X.reshape(batchsize, n_timestep, self.embed_size)
        _embed_F, _ = self.rnn_model(_embed_X)
        _embed_F_w = self.attention_func(_embed_F)
        _mix_F = torch.cat([_embed_F, _embed_F_w], dim = -1)
        _mix_F_reshape = _mix_F.view(-1, 4 * self.hidden_size)
        outputs = self.output_activate(self.output_func(_mix_F_reshape)).reshape(batchsize, n_timestep, self.output_size)
        n_batchsize, n_timestep, output_size = outputs.shape
        all_output = self.predict_func(outputs.reshape(n_batchsize*n_timestep, output_size)).\
                         reshape(n_batchsize, n_timestep, self.label_size) * M.unsqueeze(-1)
        cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output

class Dipole(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 attention_type = 'location_based',
                 attention_dim = 8,
                 embed_size = 16,
                 hidden_size = 8,
                 output_size = 8,
                 bias = True,
                 dropout = 0.5,
                 batch_first = True,
                 loss_name = 'L1LossSigmoid',
                 target_repl = False,
                 target_repl_coef = 0.,
                 aggregate = 'sum',
                 optimizer_name = 'adam',
                 use_gpu = False
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
        
        attention_type : str, optional (default = 'location_based')
            Apply attention mechnism to derive a context vector that captures relevant information to 
            help predict target.
            Current support attention methods in [location_based, general, concatenation_based] proposed in KDD2017
            'location_based'      ---> Location-based Attention. Alocation-based attention function is to 
                                       calculate the weights solely from hidden state 
            'general'             ---> General Attention. An easy way to capture the relationship between two hidden states
            'concatenation_based' ---> Concatenation-based Attention. Via concatenating two hidden states, then use multi-layer          
                                       perceptron(MLP) to calculatethe contextvector
        
        attention_dim : int, optional (default = 8)
            It is the latent dimensionality used for attention weight computing just for for concatenation_based attention mechnism 
            
        embed_size: int, optional (default = 16)
            The number of the embeded features of original input
            
        hidden_size : int, optional (default = 8)
            The number of features of the hidden state h
 
        output_size : int, optional (default = 8)
            The number of mix features
            
        bias : bool, optional (default = True)
            If False, then the layer does not use bias weights b_ih and b_hh. 
            
        dropout : float, optional (default = 0.5)
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, 
            with dropout probability equal to dropout. 

        batch_first : bool, optional (default = False)
            If True, then the input and output tensors are provided as (batch, seq, feature). 
             
        loss_name : str, optional (default='SigmoidCELoss') 
            Name or objective function.

        use_gpu : bool, optional (default=False) 
            If yes, use GPU recources; else use CPU recources 

        """
 
        super(Dipole, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias
        self.dropout = dropout
        self.batch_first = batch_first
        self.loss_name = loss_name
        self.target_repl = target_repl
        self.target_repl_coef = target_repl_coef
        self.aggregate = aggregate
        self.optimizer_name = optimizer_name
        self.use_gpu = use_gpu
        self._args_check()
        
    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        
        _config = {
            'input_size': self.input_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'bias': self.bias,
            'dropout': self.dropout,
            'batch_first': self.batch_first,
            'label_size': self.label_size,
            'attention_type': self.attention_type,
            'attention_dim': self.attention_dim
            }
        self.predictor = callPredictor(**_config).to(self.device)
        self.predictor= torch.nn.DataParallel(self.predictor)
        self._save_predictor_config(_config)
        self.criterion = callLoss(task = self.task,
                                  loss_name = self.loss_name,
                                  target_repl = self.target_repl,
                                  target_repl_coef = self.target_repl_coef,
                                  aggregate = self.aggregate)
        self.optimizer = self._get_optimizer(self.optimizer_name)

    def fit(self, train_data, valid_data):
        
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


        Returns

        -------

        self : object

            Fitted estimator.

        """
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
        assert isinstance(self.attention_type,str) and self.attention_type in ['location_based', 'general', 'concatenation_based'], \
            'fill in correct attention_type (str, [\'location_based\', \'general\', \'concatenation_based\'])'
        assert isinstance(self.attention_dim,int) and self.attention_dim>0, \
            'fill in correct attention_dim (int, >0)'        
        assert isinstance(self.embed_size,int) and self.embed_size>0, \
            'fill in correct embed_size (int, >0)'
        assert isinstance(self.hidden_size,int) and self.hidden_size>0, \
            'fill in correct hidden_size (int, 8)'
        assert isinstance(self.output_size,int) and self.output_size>0, \
            'fill in correct output_size (int, 8)'
        assert isinstance(self.bias,bool), \
            'fill in correct bias (bool)'
        assert isinstance(self.dropout,float) and self.dropout>0. and self.dropout<1., \
            'fill in correct learn_ratio (float, >0 and <1.)'
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
        self.device = self._get_device()
