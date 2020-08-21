# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import warnings
import json
import abc
import six
import tqdm
from tqdm._tqdm import trange
from ._loss import loss_dict
from pyhealth.utils.check import *
from pyhealth.data.data_reader.ecg import dl_reader

@six.add_metaclass(abc.ABCMeta)
class BaseControler(object):
    """
    Abstract class for all healthcare predict algorithms.
    
    """
    
    @abc.abstractmethod
    def __init__(self, expmodel_id):
        """

        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
            
        """
        check_model_dir(expmodel_id =  expmodel_id)
        self.checkout_dir = os.path.join('./experiments_records', expmodel_id, 'checkouts')
        self.result_dir = os.path.join('./experiments_records', expmodel_id, 'results')
        self.acc = {'train':[],'valid':[]}
        self.reverse = False
        # make saving directory if needed
        if not os.path.isdir(self.checkout_dir):
            os.makedirs(self.checkout_dir)
            
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        self.task_type = None
        self.predictor = None
        self.criterion = None
        self.optimizer = None
        self.dataparallal = False
        self.is_loadmodel = False

    @abc.abstractmethod
    def _build_model(self):
        pass

    def _get_device(self):
        if self.use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                if ',' in self.gpu_ids:
                    ids = self.gpu_ids.split(',')
                    if len(ids) > 1:
                        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
                        self.dataparallal = True
                        print ('use {0} GPUs recource'.format(len(ids)))
                else:
                    print ('use GPU recource')
            else:
                device = torch.device("cpu")
                print ('not find effcient GPU, use CPU recource')
        else:
            device = torch.device("cpu")
            print ('use CPU recource')
        return device

    def _data_check(self, datalist):

        """
        
        Target to 1) check train_data/valid_data valid, if not give tips about data problem
                  2) check loss function valid, if not recommend proper loss func
        
        Parameters

        ----------

        datalist = [data1 = {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'label_n': n of label space
                    },
                    data2 = {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'label_n': n of label space
                    }, ...
                    ]
        Returns

        -------

        self : object


        """
        
        label_n_check = set([])
        task_type_check = set([])
        for each_data in datalist:
            label_n_check.add(np.shape(np.array(each_data['y']))[1])
            task_type_check.add(label_check(each_data['y'], hat_y = None, assign_task_type = self.task_type))
            
        if len(task_type_check) != 1:
            raise Exception('task_type is inconformity in data')
        
        pre_task_type = list(task_type_check)[0]
        if self.task_type == None:
            self.task_type = pre_task_type
        elif self.task_type == pre_task_type:
            pass
        else:
            raise Exception('predifine task-type {0}, but data support task-type {1}'.format(self.task_type, pre_task_type))
        self.label_size = list(label_n_check)[0]
        self.loss_name = self._get_lossname(self.loss_name)
        print ('current task can beed seen as {0}'.format(self.task_type))
        
    def _get_lossname(self, loss_name):
        if self.task_type == 'multilabel':
            if loss_name == None or loss_name == '':
                _loss_name = 'CELossSigmoid'
            elif loss_name in loss_dict['multilabel'].keys():
                _loss_name = loss_name
            else:
                raise Exception('input correct lossfun name')
        elif self.task_type == 'multiclass':
            if loss_name == None or loss_name == '':
                _loss_name = 'L1LossSoftmax'
            elif loss_name in loss_dict['multiclass'].keys():
                _loss_name = loss_name
            else:
                raise Exception('input correct lossfun name')
        elif self.task_type == 'binaryclass':
            if loss_name == None or loss_name == '':
                _loss_name = 'BCELossSigmoid'
            elif loss_name in loss_dict['binaryclass'].keys():
                _loss_name = loss_name
            else:
                raise Exception('input correct lossfun name')
        return _loss_name

    def _get_optimizer(self, optimizer_name):
        if optimizer_name == 'adam':
             return optim.Adam(self.predictor.parameters(),
                               lr=self.learn_ratio,
                               weight_decay=self.weight_decay)
 
    def _set_reverse(self):
        self.reverse = True
 
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
        _dataset = dl_reader.DatasetReader(data)            
 
        _loader = torch.utils.data.DataLoader(_dataset,
                                              batch_size=self.n_batchsize,
                                              drop_last = True,
                                              shuffle=True if dtype == 'train' else False)
        return _loader

    def _save_predictor_config(self, predictor_config):
        temp_path = os.path.join(self.checkout_dir, "predictor_config.json")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(predictor_config, indent=4))

    def _load_predictor_config(self, config_file_path = ''):
        if config_file_path == '':
            temp_path = os.path.join(self.checkout_dir, 'predictor_config.json')
            assert os.path.exists(temp_path), 'cannot find predictor_config.json, please it in dir {0}'.format(self.checkout_dir)
        else:
            temp_path = config_file_path
            assert os.path.exists(temp_path), 'cannot find predictor_config file from path: {0}'.format(config_file_path)
        print ('load predictor config file from {0}'.format(temp_path))
        with open(temp_path, 'r') as f:
            predictor_config = json.load(f)
        return predictor_config

    def _save_checkpoint(self, state, epoch_id, is_best, filename='checkpoint.pth.tar'):
        if epoch_id<0:
            filepath = os.path.join(self.checkout_dir, 'latest.'+filename)
        elif is_best:
            filepath = os.path.join(self.checkout_dir, 'best.'+filename)
        else:
            filepath = os.path.join(self.checkout_dir, str(epoch_id)+'.'+filename)
        torch.save(state, filepath)

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
            inputs = databatch['X']
            targets = databatch['Y']
            inputs = Variable(inputs).float().to(self.device)
            targets = Variable(targets).float().to(self.device)
            outputs = self.predictor(inputs)
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
            inputs = databatch['X']
            targets = databatch['Y']
            inputs = Variable(inputs).float().to(self.device)
            targets = Variable(targets).float().to(self.device)
            outputs = self.predictor(inputs)
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
            inputs = databatch['X']
            targets = databatch['Y']
            inputs = Variable(inputs).float().to(self.device)
            targets = Variable(targets).float().to(self.device)
            outputs = self.predictor(inputs)
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

    def _fit_model(self, train_reader, valid_reader):
        best_score = 1e5
        tqdm_trange = trange(0, self.n_epoch)
        for epoch in tqdm_trange:
            self._train_model(train_reader)
            self._valid_model(valid_reader)
            train_loss = self.acc['train'][-1]
            valid_loss = self.acc['valid'][-1]
            train_loss_str = '{:.3f}'.format(self.acc['train'][-1])
            valid_loss_str = '{:.3f}'.format(self.acc['valid'][-1])
            tqdm_trange.set_description(f'tr=>epoch={epoch} Valid Loss: {valid_loss_str}, Train Loss: {train_loss_str}')
            unit = {'epoch': epoch,
                    'state_dict': self.predictor.state_dict(),
                    'score': valid_loss,
                    'best_score': best_score,
                    'optimizer' : self.optimizer.state_dict()}
            if valid_loss<best_score:
                best_score = valid_loss
                unit['best_score'] = valid_loss
                self._save_checkpoint(unit, epoch, is_best = True)
            if epoch%self.n_epoch_saved == 0:
                self._save_checkpoint(unit, epoch, is_best = False)
            self._save_checkpoint(unit, -1, is_best = False)

    def _load_model(self, 
                    loaded_epoch = '',  
                    model_file_path = ''):
        if model_file_path != '':
            load_checkpoint_path = model_file_path
            print ('load model from {0}'.format(model_file_path))
        else:
            if loaded_epoch != '':
                self._loaded_epoch = loaded_epoch
            else:
                self._loaded_epoch = 'best'
            load_checkpoint_path = os.path.join(self.checkout_dir, self._loaded_epoch + '.checkpoint.pth.tar')
            print ('load '+self._loaded_epoch+'-th epoch model')
        if os.path.exists(load_checkpoint_path):
            try:
                checkpoint = torch.load(load_checkpoint_path)
            except:
                checkpoint = torch.load(load_checkpoint_path, map_location = 'cpu')
            try:
                self.predictor.load_state_dict({key: value for key, value in checkpoint['state_dict'].items()})
            except:
                self.predictor.load_state_dict({key[7:]: value for key, value in checkpoint['state_dict'].items()})							  
        else:
            print ('no exist '+self._loaded_epoch+'-th epoch model, please dbcheck in dir {0}'.format(self.checkout_dir))
        self.is_loadmodel = True

    @abc.abstractmethod
    def fit(self, train_data, valid_data):
        """
        Parameters

        ----------

        train_data : {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'label_n': n of label space
                      }

            The input train samples dict.
 
        valid_data : {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'label_n': n of label space
                      }

            The input valid samples dict.


        Returns

        -------

        self : object

            Fitted estimator.

        """

        pass

#     @abc.abstractmethod
#     def inference(self, test_data):
#         Parameters

#         ----------

#         test_data : {
#                       'x':list[episode_file_path], 
#                       'y':list[label], 
#                       'l':list[seq_len], 
#                       'feat_n': n of feature space, 
#                       'label_n': n of label space
#                       }

#             The input test samples dict.
 
#         pass

    def inference(self, test_data):
        """
        Parameters

        ----------

        test_data : {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'label_n': n of label space
                      }

            The input test samples dict.
 
 
        """
        self._data_check([test_data])
        test_reader = self._get_reader(test_data, 'test')
        self._test_model(test_reader)

    @abc.abstractmethod
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

        pass

    def get_results(self):
        
        """
        
        Load saved prediction results in current ExpID
            truth_value: proj_root/experiments_records/*****(exp_id)/results/y.xxx
            predict_value: proj_root/experiments_records/*****(exp_id)/results/hat_y.xxx
            xxx represents the loaded model
        
        """
        try:
            hat_y = pickle.load(open(os.path.join(self.result_dir, 'hat_y.'+self._loaded_epoch),'rb'))
        except IOError:
            print ('Error: cannot find file {0} or load failed'.format(os.path.join(self.result_dir, 'hat_y.'+self._loaded_epoch)))
        try:
            y = pickle.load(open(os.path.join(self.result_dir, 'y.'+self._loaded_epoch),'rb'))
        except IOError:
            print ('Error: cannot find file {0} or load failed'.format(os.path.join(self.result_dir, 'y.'+self._loaded_epoch)))

        results = {'hat_y': hat_y, 'y': y}
        
        return results
