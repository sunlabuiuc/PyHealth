import os
import torch
import torch.nn as nn
import pickle
import warnings
import tqdm
from tqdm._tqdm import trange
from ..utils.loss import callLoss
from .base import BaseControler

warnings.filterwarnings('ignore')

class RetainAttention(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(RetainAttention, self).__init__()
        self.embed_size = embed_size        
        self.hidden_size = hidden_size
        self.attention_beta = nn.Linear(hidden_size, embed_size)
        self.activate_beta = nn.Tanh()
        self.attention_alpha = nn.Linear(hidden_size, 1)
        self.activate_alpha = nn.Softmax(dim = -1)
        
    def forward(self, data_alpha, data_beta, data_embed, data_mask):
        # shape of data_alpha: <n_batch, n_seq, hidden_size>         
        # shape of data_beta : <n_batch, n_seq, hidden_size>         
        # shape of data_embed: <n_batch, n_seq, embed_size>   
        # shape of data_mask: <n_batch, n_seq>   
        
        # generate beta weights
        n_batch, n_seq, hidden_size = data_beta.shape
        # shape of beta_weights: <n_batch, n_seq, embed_size>   
        beta_weights = self.activate_beta(self.attention_beta(data_beta.reshape(-1, hidden_size))).reshape(n_batch, n_seq, self.embed_size)

        # generate alpha weights
        n_batch, n_seq, hidden_size = data_alpha.shape
        # shape of _ori_correlate_value: <n_batch, 1, n_seq>   
        _correlate_value = self.attention_alpha(data_alpha.reshape(-1, hidden_size)).reshape(n_batch, n_seq).unsqueeze(1)
        # shape of attention_value_format: <n_batch, 1, n_seq>   
        attention_value_format = torch.exp(_correlate_value)
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 1  1  1 ] 
        #    [ 0  1  1 ]
        #    [ 0  0  1 ]]]
        ensemble_flag = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 0).unsqueeze(0)
        # shape of _format_mask: <n_batch, 1, n_seq>   
        _format_mask = data_mask.unsqueeze(1)
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        ensemble_flag_format = ensemble_flag * _format_mask
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        alpha_weights = each_attention_value/accumulate_attention_value
        
        # shape of _visit_beta_weights: <n_batch, 1, n_seq, embed_size>
        _visit_beta_weights = beta_weights.unsqueeze(1)
        # shape of _visit_alpha_weights: <n_batch, n_seq, n_seq, 1>
        _visit_alpha_weights = alpha_weights.unsqueeze(-1)
        # shape of _visit_data_embed: <n_batch, 1, n_seq, embed_size>           
        _visit_data_embed = data_embed.unsqueeze(1)
        
        # shape of mix_weights: <n_batch, n_seq, n_seq, embed_size>
        mix_weights = _visit_beta_weights * _visit_alpha_weights
        # shape of weighted_output: <n_batch, n_seq, embed_size>        
        weighted_output = torch.sum(mix_weights * _visit_data_embed, dim = -2)
        
        return weighted_output

class callPredictor(nn.Module):
    def __init__(self, 
                 input_size = None,
                 embed_size = 16,
                 hidden_size = 8,
                 bias = True,
                 dropout = 0.5,
                 batch_first = True,
                 label_size = 1):
        super(callPredictor, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size' 
 
        self.input_size = input_size        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.embed_func = nn.Linear(self.input_size, self.embed_size)
        self.rnn_model_alpha = nn.GRU(input_size = embed_size,
                                      hidden_size = hidden_size,
                                      bias = bias,
                                      dropout = dropout,
                                      bidirectional = False,
                                      batch_first = batch_first)
        self.rnn_model_beta = nn.GRU(input_size = embed_size,
                                     hidden_size = hidden_size,
                                     bias = bias,
                                     dropout = dropout,
                                     bidirectional = False,
                                     batch_first = batch_first)
 
        self.attention_func = RetainAttention(self.embed_size, self.hidden_size)
        self.predict_func = nn.Linear(self.embed_size, self.label_size)
            
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
        n_batchsize, n_timestep, n_orifeatdim = X.shape
        _ori_X = X.view(-1, n_orifeatdim)
        _embed_X = self.embed_func(_ori_X)        
        _embed_X = _embed_X.reshape(n_batchsize, n_timestep, self.embed_size)        
        _embed_alpha, _ = self.rnn_model_alpha(_embed_X)
        _embed_beta, _ = self.rnn_model_beta(_embed_X)
        weight_outputs = self.attention_func(_embed_alpha, _embed_beta, _embed_X, M)
        
        weight_outputs_reshape = weight_outputs.view(-1, self.embed_size)
        all_output = self.predict_func(weight_outputs_reshape).\
                         reshape(n_batchsize, n_timestep, self.label_size) * M.unsqueeze(-1)
        cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output

class Retain(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 embed_size = 16,
                 hidden_size = 8,
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
        
        embed_size: int, optional (default = 16)
            The number of the embeded features of original input
            
        hidden_size : int, optional (default = 8)
            The number of features of the hidden state h
 
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
 
        super(Retain, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.batch_first = batch_first
        self.loss_name = loss_name
        self.target_repl = target_repl
        self.target_repl_coef = target_repl_coef
        self.aggregate = aggregate
        self.optimizer_name = optimizer_name
        self.use_gpu = use_gpu
        self._set_reverse()
        self._args_check()
        
    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        
        _config = {
            'input_size': self.input_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'bias': self.bias,
            'dropout': self.dropout,
            'batch_first': self.batch_first,
            'label_size': self.label_size
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
        assert isinstance(self.embed_size,int) and self.embed_size>0, \
            'fill in correct embed_size (int, >0)'
        assert isinstance(self.hidden_size,int) and self.hidden_size>0, \
            'fill in correct hidden_size (int, 8)'
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
