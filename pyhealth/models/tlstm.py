import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import pickle
import warnings
import tqdm
from tqdm._tqdm import trange
from ..utils.loss import callLoss
from .base import BaseControler

warnings.filterwarnings('ignore')

class tLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(tLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.c2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.activate_func_c = nn.Tanh()
        self.activate_func_h = nn.Sigmoid()
        self.activate_func_t = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, data_x, data_t, h_t_1, c_t_1):   
#         print (data_x.shape)
#         print (data_t.shape)
#         print (h_t_1.shape)
#         print (c_t_1.shape)

        # shape of data_x    : <n_batch, input_size> 
        # shape of data_t    : <n_batch, 1> 
        # shape of data_h_t_1: <n_batch, hidden_size> 
        # shape of data_s_t_1: <n_batch, hidden_size> 
        
        # shape of gate_set  : <n_batch, 4 * hidden_size> 
        gate_set = self.x2h(data_x)  + self.h2h(h_t_1)
        # shape of f_t       : <n_batch, hidden_size> 
        # shape of i_t       : <n_batch, hidden_size> 
        # shape of o_t       : <n_batch, hidden_size> 
        # shape of c_hat     : <n_batch, hidden_size> 
        _f_t, _i_t, _o_t, _c_hat = gate_set.chunk(4, -1)
        f_t = self.activate_func_h(_f_t)
        i_t = self.activate_func_h(_i_t)
        o_t = self.activate_func_h(_o_t)
        c_cur = self.activate_func_c(_c_hat)
        
        # shape of c_s_t_1   : <n_batch, hidden_size> 
        c_s_t_1 = self.activate_func_c(self.c2h(c_t_1))
        c_s_t_1_hat = c_s_t_1 * self.activate_func_t(data_t)
        c_T_t_1 = c_t_1 - c_s_t_1
        c_star_t_1 = c_T_t_1 + c_s_t_1_hat
        
        # shape of c_t       : <n_batch, hidden_size> 
        # shape of h_t       : <n_batch, hidden_size> 
        c_t = f_t * c_star_t_1 + i_t * c_cur
        h_t = o_t * self.activate_func_c(c_t)
        return (h_t, c_t)

class callPredictor(nn.Module):
    
    def __init__(self, 
                 input_size = None,
                 hidden_size = 16,
                 output_size = 8,
                 batch_first = True,
                 dropout = 0.5,
                 label_size = 1):
        super(callPredictor, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size' 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.output_size = output_size
        self.predict_func = nn.Linear(self.output_size, self.label_size)
        self.rnn_unit = tLSTMCell(input_size, hidden_size) 

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
        batchsize, n_timestep, _ = X.shape
        h0 = Variable(torch.zeros(batchsize, self.hidden_size))
        c0 = Variable(torch.zeros(batchsize, self.hidden_size))
        outputs = []      
        h_t, c_t = h0, c0
        for t in range(n_timestep):
            h_t, c_t = self.rnn_unit(X[:,t,:], T[:, t].reshape(-1, 1), h_t, c_t) 
            outputs.append(h_t)
        outputs = torch.stack(outputs, dim=1)
        n_batchsize, n_timestep, n_featdim = outputs.shape
        all_output = self.predict_func(outputs.reshape(n_batchsize*n_timestep, n_featdim)).\
                        reshape(n_batchsize, n_timestep, self.label_size) * M.unsqueeze(-1)
        cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output

class tLSTM(BaseControler):
    
    """
    
    Time-Aware LSTM (T-LSTM), A kind of time-aware RNN neural network;
        Used to handle irregular time intervals in longitudinal patient records.
    
    """
    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_batchsize = 5,
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
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
                   
        hidden_size : int, optional (default = 8)
            The number of features of the hidden state h

        output_size: int, optional (default = 8)
            The number of the embeded features of rnn output

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
 
        super(tLSTM, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
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
        self._set_reverse()
        self._args_check()
        
    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        
        _config = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
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
        assert isinstance(self.hidden_size,int) and self.hidden_size>0, \
            'fill in correct hidden_size (int, 8)'
        assert isinstance(self.output_size,int) and self.output_size>0, \
            'fill in correct output_size (int, >0)'
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
