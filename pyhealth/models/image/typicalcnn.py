import os
import torch
import torch.nn as nn
import pickle
import warnings
import torchvision.models as models
from ._loss import callLoss
from ._dlbase import BaseControler

warnings.filterwarnings('ignore')

class TypicalCNN(BaseControler):

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 cnn_name = 'resnet18',
                 pretrained = False,
                 n_epoch = 100,
                 n_batchsize = 5,
                 load_size = 255,
                 crop_size = 224,
                 learn_ratio = 1e-4,
                 weight_decay = 1e-4,
                 n_epoch_saved = 1,
                 bias = True,
                 dropout = 0.5,
                 batch_first = True,
                 loss_name = 'L1LossSoftmax',
                 aggregate = 'sum',
                 optimizer_name = 'adam',
                 use_gpu = False,
                 gpu_ids = '0'
                 ):
        """
        Several typical & popular CNN networks for medical image prediction 


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
       
        cnn_name : str, optional (default = 'resnet18')
            name of typical/popular CNN networks
        
        pretrained : bool, optional (default = False)
            used for pre-trained model load, True -> load pretrained model; False -> not load
            
        n_epoch : int, optional (default = 100)
            number of epochs with the initial learning rate
            
        n_batchsize : int, optional (default = 5)
            batch size for model training
        
        load_size : int, optional (default = 255)
            scale images to this size

        crop_size : int, optional (default = 224)
            crop load_sized image into to this size
            
        learn_ratio : float, optional (default = 1e-4)
            initial learning rate for adam
  
        weight_decay : float, optional (default = 1e-4)
            weight decay (L2 penalty)
  
        n_epoch_saved : int, optional (default = 1)
            frequency of saving checkpoints at the end of epochs
            
        bias : bool, optional (default = True)
            If False, then the layer does not use bias weights b_ih and b_hh. 
            
        dropout : float, optional (default = 0.5)
            If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
            with dropout probability equal to dropout. 

        batch_first : bool, optional (default = False)
            If True, then the input and output tensors are provided as (batch, seq, feature). 
             
        loss_name : str, optional (default='SigmoidCELoss') 
            Name or objective function.

        use_gpu : bool, optional (default=False) 
            If yes, use GPU recources; else use CPU recources 

				gpu_ids : str, optional (default='') 
										If yes, assign concrete used gpu ids such as '0,2,6'; else use '0' 

        """
 
        super(TypicalCNN, self).__init__(expmodel_id)
        self.cnn_name = cnn_name
        self.pretrained = pretrained
        self.n_batchsize = n_batchsize
        self.n_epoch = n_epoch
        self.load_size = load_size
        self.crop_size = crop_size
        self.learn_ratio = learn_ratio
        self.weight_decay = weight_decay
        self.n_epoch_saved = n_epoch_saved
        self.bias = bias
        self.dropout = dropout
        self.batch_first = batch_first
        self.loss_name = loss_name
        self.aggregate = aggregate
        self.optimizer_name = optimizer_name
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self._args_check()
 
    def _get_predictor(self):
        # create model
        if self.pretrained:
            print("=> using pre-trained model '{}'".format(self.cnn_name))
            predictor = models.__dict__[self.cnn_name](pretrained=True)
        else:
            print("=> creating model '{}'".format(self.cnn_name))
            predictor = models.__dict__[self.cnn_name](pretrained=False)
        # modify model-output 
        if self.cnn_name == 'resnet18':
            predictor.fc = torch.nn.Linear(512, self.label_size, bias=True)
        elif self.cnn_name == 'resnet50':
            predictor.fc = torch.nn.Linear(2048, self.label_size, bias=True)
        elif self.cnn_name == 'resnet101':
            predictor.fc = torch.nn.Linear(2048, self.label_size, bias=True)
        elif self.cnn_name == 'resnet152':
            predictor.fc = torch.nn.Linear(2048, self.label_size, bias=True)
        elif self.cnn_name == 'densenet121':
            predictor.classifier = torch.nn.Linear(1024, self.label_size, bias=True)
        elif self.cnn_name == 'densenet161':
            predictor.classifier = torch.nn.Linear(2208, self.label_size, bias=True)
        print('    Total params: %.2fM' % (sum(p.numel() for p in predictor.parameters())/1000000.0))
        return predictor
    
    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        _config = {'label_size': self.label_size}
        self._save_predictor_config(_config)
        predictor = self._get_predictor()
        self.predictor = predictor.to(self.device)
        if self.dataparallal:
            self.predictor= torch.nn.DataParallel(self.predictor)
        self.criterion = callLoss(task = self.task_type,
                                  loss_name = self.loss_name,
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

        _config = self._load_predictor_config()
        self.label_size = _config['label_size']
        predictor = self._get_predictor()
        self.predictor = predictor.to(self.device)
        self._load_model(loaded_epoch)
 

    def _args_check(self):
        """
        
        Check args whether valid/not and give tips
 
        
        """
        assert isinstance(self.cnn_name,str) and self.cnn_name in ['resnet18'], \
            'fill in correct cnn_name (str, [\'resnet18\'])'
        assert isinstance(self.pretrained,bool), \
            'fill in correct pretrained (bool)'
        assert isinstance(self.n_batchsize,int) and self.n_batchsize>0, \
            'fill in correct n_batchsize (int, >0)'
        assert isinstance(self.n_epoch,int) and self.n_epoch>0, \
            'fill in correct n_epoch (int, >0)'
        assert isinstance(self.load_size,int) and self.load_size>0, \
            'fill in correct load_size (int, >0)'
        assert isinstance(self.crop_size,int) and self.crop_size>0 and self.crop_size<self.load_size, \
            'fill in correct crop_size (int, >0, <{0})'.format(self.load_size)
        assert isinstance(self.learn_ratio,float) and self.learn_ratio>0., \
            'fill in correct learn_ratio (float, >0.)'
        assert isinstance(self.weight_decay,float) and self.weight_decay>=0., \
            'fill in correct weight_decay (float, >=0.)'
        assert isinstance(self.n_epoch_saved,int) and self.n_epoch_saved>0 and self.n_epoch_saved < self.n_epoch, \
            'fill in correct n_epoch (int, >0 and <{0}).format(self.n_epoch)'
        assert isinstance(self.bias,bool), \
            'fill in correct bias (bool)'
        assert isinstance(self.dropout,float) and self.dropout>0. and self.dropout<1., \
            'fill in correct learn_ratio (float, >0 and <1.)'
        assert isinstance(self.aggregate,str) and self.aggregate in ['sum','avg'], \
            'fill in correct aggregate (str, [\'sum\',\'avg\'])'
        assert isinstance(self.optimizer_name,str) and self.optimizer_name in ['adam'], \
            'fill in correct optimizer_name (str, [\'adam\'])'
        assert isinstance(self.use_gpu,bool), \
            'fill in correct use_gpu (bool)'
        assert isinstance(self.loss_name,str), \
            'fill in correct optimizer_name (str)'
        self.device = self._get_device()
