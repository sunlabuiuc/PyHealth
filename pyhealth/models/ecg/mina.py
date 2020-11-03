# -*- coding: utf-8 -*-

# License: BSD 2 clause

import os
import torch
import torch.nn as nn
import pickle
import warnings
import torchvision.models as models
from ._loss import callLoss
from ._dlbase import BaseControler
from pyhealth.data.data_reader.ecg import mina_reader
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

warnings.filterwarnings('ignore')

class BaseNet(nn.Module):
    def __init__(self, n_dim, n_split):
        super(BaseNet, self).__init__()
        
        self.n_dim = n_dim
        self.n_split = n_split
        self.n_seg = int(n_dim/n_split)

        ### Input: (batch size, number of channels, length of signal sequence)
        self.conv_out_channels = 64
        self.conv_kernel_size = 32
        self.conv_stride = 2
        self.conv = nn.Conv1d(in_channels=1, 
                              out_channels=self.conv_out_channels, 
                              kernel_size=self.conv_kernel_size, 
                              stride=self.conv_stride)
        self.conv_k = nn.Conv1d(in_channels=1, 
                                out_channels=1, 
                                kernel_size=self.conv_kernel_size, 
                                stride=self.conv_stride)
        self.att_cnn_dim = 8
        self.W_att_cnn = nn.Parameter(torch.randn(self.conv_out_channels+1, self.att_cnn_dim))
        self.v_att_cnn = nn.Parameter(torch.randn(self.att_cnn_dim, 1))
        
        ### Input: (batch size, length of signal sequence, input_size)
        self.rnn_hidden_size = 32
        self.lstm = nn.LSTM(input_size=(self.conv_out_channels), 
                            hidden_size=self.rnn_hidden_size, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.att_rnn_dim = 8
        self.W_att_rnn = nn.Parameter(torch.randn(2*self.rnn_hidden_size+1, self.att_rnn_dim))
        self.v_att_rnn = nn.Parameter(torch.randn(self.att_rnn_dim, 1))
        
        ### fc
        self.do = nn.Dropout(p=0.5)
        self.out_size = 8
        self.fc = nn.Linear(2*self.rnn_hidden_size, self.out_size)
    
    def forward(self, x, k_beat, k_rhythm):
        
        self.batch_size = x.size()[0]

        ############################################
        ### reshape
        ############################################
        # print('orignial x:', x.size())
        x = x.reshape(-1, self.n_split)
        x = x.unsqueeze(1)
        
        k_beat = k_beat.reshape(-1, self.n_split)
        k_beat = k_beat.unsqueeze(1)
        
        ############################################
        ### conv
        ############################################
        x = F.relu(self.conv(x))
        
        k_beat = F.relu(self.conv_k(k_beat))
        
        ############################################
        ### attention conv
        ############################################
        x = x.permute(0, 2, 1)
        k_beat = k_beat.permute(0, 2, 1)
        tmp_x = torch.cat((x, k_beat), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_cnn)
        e = torch.matmul(torch.tanh(e), self.v_att_cnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        
        ############################################
        ### reshape for rnn
        ############################################
        x = x.view(self.batch_size, self.n_seg, -1)
    
        ############################################
        ### rnn        
        ############################################
        
        k_rhythm = k_rhythm.unsqueeze(-1)
        o, (ht, ct) = self.lstm(x)
        tmp_o = torch.cat((o, k_rhythm), dim=-1)
        e = torch.matmul(tmp_o, self.W_att_rnn)
        e = torch.matmul(torch.tanh(e), self.v_att_rnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        beta = torch.div(n1, n2)
        x = torch.sum(torch.mul(beta, o), 1)
        
        ############################################
        ### fc
        ############################################
        x = F.relu(self.fc(x))
        x = self.do(x)
        
        return x, alpha, beta        

class callPredictor(nn.Module):
    def __init__(self, 
                 n_dim, 
                 n_channel = 4, 
                 n_split = 50,
                 label_size=4
                 ):
        super(callPredictor, self).__init__()
        
        self.n_channel = n_channel
        self.n_dim = n_dim
        self.n_split = n_split
        self.label_size = label_size
        
        self.base_net_0 = BaseNet(self.n_dim, self.n_split)
        self.base_net_1 = BaseNet(self.n_dim, self.n_split)
        self.base_net_2 = BaseNet(self.n_dim, self.n_split)
        self.base_net_3 = BaseNet(self.n_dim, self.n_split)
            
        ### attention
        self.out_size = 8
        self.att_channel_dim = 2
        self.W_att_channel = nn.Parameter(torch.randn(self.out_size+1, self.att_channel_dim))
        self.v_att_channel = nn.Parameter(torch.randn(self.att_channel_dim, 1))
        
        ### fc
        self.fc = nn.Linear(self.out_size, self.label_size)
        
    def forward(self, x_0, x_1, x_2, x_3, 
                k_beat_0, k_beat_1, k_beat_2, k_beat_3, 
                k_rhythm_0, k_rhythm_1, k_rhythm_2, k_rhythm_3, 
                k_freq):

        x_0, alpha_0, beta_0 = self.base_net_0(x_0, k_beat_0, k_rhythm_0)
        x_1, alpha_1, beta_1 = self.base_net_1(x_1, k_beat_1, k_rhythm_1)
        x_2, alpha_2, beta_2 = self.base_net_2(x_2, k_beat_2, k_rhythm_2)
        x_3, alpha_3, beta_3 = self.base_net_3(x_3, k_beat_3, k_rhythm_3)
        
        x = torch.stack([x_0, x_1, x_2, x_3], 1)

        # ############################################
        # ### attention on channel
        # ############################################
#        k_freq = k_freq.permute(1, 0, 2)

        tmp_x = torch.cat((x, k_freq), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_channel)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        gama = torch.div(n1, n2)
        x = torch.sum(torch.mul(gama, x), 1)
        
        ############################################
        ### fc
        ############################################
        x = F.softmax(self.fc(x), 1)
        
        ############################################
        ### return 
        ############################################
        
        att_dic = {"alpha_0":alpha_0, "beta_0":beta_0, 
                  "alpha_1":alpha_1, "beta_1":beta_1, 
                  "alpha_2":alpha_2, "beta_2":beta_2, 
                  "alpha_3":alpha_3, "beta_3":beta_3, 
                  "gama":gama}
        
        return x

class MINA(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_epoch = 100,
                 n_split = 50,
                 n_batchsize = 5,
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
        
        MultIlevel kNowledge-guided Attention networks (MINA) that predict heart diseases from ECG signals with in- tuitive explanation aligned with medical knowledge.

        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
       
        n_epoch : int, optional (default = 100)
            number of epochs with the initial learning rate

        n_split : int, optional (default = 50)
            number of splited sequence

        n_batchsize : int, optional (default = 5)
            batch size for model training
        
        embed_channel : int, optional (default = 4)
            define number of embeded channel 
						
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
 
        super(MINA, self).__init__(expmodel_id)
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.learn_ratio = learn_ratio
        self.n_split = n_split
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
                 'n_dim': self.n_feat, 
                 'n_channel': self.n_channel, 
                 'n_split': self.n_split,
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
        _dataset = mina_reader.DatasetReader(data, n_split=self.n_split)            
        self.n_channel = _dataset.n_channel
        self.n_feat = _dataset.n_feat
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

    def _train_model(self, train_loader):
        
        """
        Parameters

        ----------

        train_loader : dataloader of train data
        
            Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

            refer to torch.utils.data.dataloader

        """

        loss_v = []
        self.predictor.train()
        for batch_idx, databatch in enumerate(train_loader):
            X_data = Variable(databatch['X_data']).float().to(self.device)
            K_train_beat = Variable(databatch['K_train_beat']).float().to(self.device)
            K_train_rhythm = Variable(databatch['K_train_rhythm']).float().to(self.device)
            K_train_freq = Variable(databatch['K_train_freq']).float().to(self.device)
            targets = Variable(databatch['Y']).float().to(self.device)
            outputs = self.predictor(X_data[:,0,:], X_data[:,1,:], X_data[:,2,:], X_data[:,3,:],
                                    K_train_beat[:, 0, :], K_train_beat[:, 1, :], K_train_beat[:, 2, :], K_train_beat[:, 3, :],
                                    K_train_rhythm[:, 0, :], K_train_rhythm[:, 1, :], K_train_rhythm[:, 2, :], K_train_rhythm[:, 3, :],
                                    K_train_freq)
            loss = self.criterion({'hat_y': outputs, 'y': targets})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_v.append(loss.cpu().data.numpy())
        self.acc['train'].append(np.mean(np.array(loss_v)))

    def _valid_model(self, valid_loader):
        """
        Parameters

        ----------

        valid_loader : dataloader of valid data
        
            Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

            refer to torch.utils.data.dataloader

        """

        loss_v = []
        for batch_idx, databatch in enumerate(valid_loader):
            X_data = Variable(databatch['X_data']).float().to(self.device)
            K_train_beat = Variable(databatch['K_train_beat']).float().to(self.device)
            K_train_rhythm = Variable(databatch['K_train_rhythm']).float().to(self.device)
            K_train_freq = Variable(databatch['K_train_freq']).float().to(self.device)
            targets = Variable(databatch['Y']).float().to(self.device)
            outputs = self.predictor(X_data[:,0,:], X_data[:,1,:], X_data[:,2,:], X_data[:,3,:],
                                    K_train_beat[:, 0, :], K_train_beat[:, 1, :], K_train_beat[:, 2, :], K_train_beat[:, 3, :],
                                    K_train_rhythm[:, 0, :], K_train_rhythm[:, 1, :], K_train_rhythm[:, 2, :], K_train_rhythm[:, 3, :],
                                    K_train_freq)
            loss = self.criterion({'hat_y': outputs, 'y': targets})
            loss_v.append(loss.cpu().data.numpy())
        self.acc['valid'].append(np.mean(np.array(loss_v)))

    def _test_model(self, test_loader):
        """
        Parameters

        ----------

        test_loader : dataloader of test data
        
            Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

            refer to torch.utils.data.dataloader

        """

        # switch to train mode
        self.predictor.eval()
        pre_v = []
        prob_v = []
        real_v = []
        for batch_idx, databatch in enumerate(test_loader):
            X_data = Variable(databatch['X_data']).float().to(self.device)
            K_train_beat = Variable(databatch['K_train_beat']).float().to(self.device)
            K_train_rhythm = Variable(databatch['K_train_rhythm']).float().to(self.device)
            K_train_freq = Variable(databatch['K_train_freq']).float().to(self.device)
            outputs = self.predictor(X_data[:,0,:], X_data[:,1,:], X_data[:,2,:], X_data[:,3,:],
                                    K_train_beat[:, 0, :], K_train_beat[:, 1, :], K_train_beat[:, 2, :], K_train_beat[:, 3, :],
                                    K_train_rhythm[:, 0, :], K_train_rhythm[:, 1, :], K_train_rhythm[:, 2, :], K_train_rhythm[:, 3, :],
                                    K_train_freq)

            if self.task_type in ['multiclass']:
                prob_h = F.softmax(outputs, dim = -1)
            else:
                prob_h = F.sigmoid(outputs)
            pre_v.append(outputs.cpu().detach().numpy())
            prob_v.append(prob_h.cpu().detach().numpy())
            real_v.append(targets.cpu().detach().numpy())
        pickle.dump(np.concatenate(pre_v, 0), open(os.path.join(self.result_dir, 'hat_ori_y.'+self._loaded_epoch),'wb'))
        pickle.dump(np.concatenate(prob_v, 0), open(os.path.join(self.result_dir, 'hat_y.'+self._loaded_epoch),'wb'))
        pickle.dump(np.concatenate(real_v, 0), open(os.path.join(self.result_dir, 'y.'+self._loaded_epoch),'wb'))

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
        assert isinstance(self.n_split,int) and self.n_split>0, \
            'fill in correct n_split (int, >0, default = 50)'
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
        self.device = self._get_device()

