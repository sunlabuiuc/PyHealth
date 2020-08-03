# -*- coding: utf-8 -*-
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
from ..data import rnn_reader
from ..utils.loss import loss_dict
from ..utils.checklist import *

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
        
        # make saving directory if needed
        if not os.path.isdir(self.checkout_dir):
            os.makedirs(self.checkout_dir)
            
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

    @abc.abstractmethod
    def _build_model(self):
        pass

    def _get_device(self):
        if self.use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print ('use GPU recource')
            else:
                device = torch.device("cpu")
                print ('not find effcient GPU, use CPU recource')
        else:
            device = torch.device("cpu")
            print ('use CPU recource')
        return device

    def _get_lossname(self, loss_name):
        if self.task in ['phenotyping']:
            if loss_name == None or loss_name == '':
                _loss_name = 'CELossSigmoid'
            elif loss_name in loss_dict['phenotyping'].keys():
                _loss_name = loss_name
            else:
                raise Exception('input correct lossfun name')
        elif self.task in ['mortality']:
            if loss_name == None or loss_name == '':
                _loss_name = 'BCELossSigmoid'
            elif loss_name in loss_dict['mortality'].keys():
                _loss_name = loss_name
            else:
                raise Exception('input correct lossfun name')
        return _loss_name

    def _get_optimizer(self):
        if self.optimizer_name == 'adam':
             return optim.Adam(self.predictor.parameters(),
                               lr=self.learn_ratio,
                               weight_decay=self.weight_decay)
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

        _dataset = rnn_reader.DatasetReader(data)
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

    def _load_predictor_config(self):
        temp_path = os.path.join(self.checkout_dir, 'predictor_config.json')
        assert os.path.exists(temp_path), 'cannot find predictor_config.json, please it in dir {0}'.format(self.checkout_dir)
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

    def _train(self, train_loader):
        
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
            cur_masks = databatch['cur_M']
            masks = databatch['M']
            targets = databatch['Y']
            timetick = databatch['T']
            inputs = Variable(inputs).float().to(self.device)
            masks = Variable(masks).float().to(self.device)
            cur_masks = Variable(cur_masks).float().to(self.device)
            targets = Variable(targets).float().to(self.device)
            timetick = Variable(timetick).float().to(self.device)
            data_input = {'X':inputs,'cur_M':cur_masks,'M':masks, 'T':timetick}
            all_h, h = self.predictor(data_input)
            if self.target_repl:
                data_output = {'all_hat_y': all_h, 'hat_y':h,'y':targets,'mask':masks} 
            else:
                data_output = {'hat_y':h,'y':targets}
            loss = self.criterion(data_output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_v.append(loss.cpu().data.numpy())
        self.acc['train'].append(np.mean(np.array(loss_v)))

    def _valid(self, valid_loader):
        """
        Parameters

        ----------

        valid_loader : dataloader of valid data
        
            Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

            refer to torch.utils.data.dataloader

        """

        self.predictor.eval()
        loss_v = []
        for batch_idx, databatch in enumerate(valid_loader):
            inputs = databatch['X']
            cur_masks = databatch['cur_M']
            masks = databatch['M']
            targets = databatch['Y']
            timetick = databatch['T']
            inputs = Variable(inputs).float().to(self.device)
            masks = Variable(masks).float().to(self.device)
            cur_masks = Variable(cur_masks).float().to(self.device)
            targets = Variable(targets).float().to(self.device)
            timetick = Variable(timetick).float().to(self.device)
            data_input = {'X':inputs,'cur_M':cur_masks,'M':masks, 'T':timetick}
            all_h, h = self.predictor(data_input)
            if self.target_repl:
                data_output = {'all_hat_y': all_h, 'hat_y':h,'y':targets,'mask':masks} 
            else:
                data_output = {'hat_y':h,'y':targets}
            loss = self.criterion(data_output)
            loss_v.append(loss.cpu().data.numpy())
        self.acc['valid'].append(np.mean(np.array(loss_v)))

    def _test(self, test_loader):
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
            cur_masks = databatch['cur_M']
            masks = databatch['M']
            targets = databatch['Y']
            timetick = databatch['T']
            inputs = Variable(inputs).float().to(self.device)
            masks = Variable(masks).float().to(self.device)
            cur_masks = Variable(cur_masks).float().to(self.device)
            targets = Variable(targets).float().to(self.device)
            timetick = Variable(timetick).float().to(self.device)
            data_input = {'X':inputs,'cur_M':cur_masks,'M':masks, 'T':timetick}
            _, h = self.predictor(data_input)
            if self.task in ['phenotyping']:
                prob_h = F.softmax(h, dim = -1)
            else:
                prob_h = F.sigmoid(h)
            pre_v.append(h.cpu().detach().numpy())
            prob_v.append(prob_h.cpu().detach().numpy())
            real_v.append(targets.cpu().detach().numpy())
        pickle.dump(np.concatenate(pre_v, 0), open(os.path.join(self.result_dir, 'hat_ori_y.'+self._loaded_epoch),'wb'))
        pickle.dump(np.concatenate(prob_v, 0), open(os.path.join(self.result_dir, 'hat_y.'+self._loaded_epoch),'wb'))
        pickle.dump(np.concatenate(real_v, 0), open(os.path.join(self.result_dir, 'y.'+self._loaded_epoch),'wb'))
     
    @abc.abstractmethod
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

        pass

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

    @abc.abstractmethod
    def get_results(self):
        pass

    @abc.abstractmethod
    def inference(self, test_data):
        """
        Parameters

        ----------

        test_data : {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                      }

            The input test samples dict.
 
 
        """

        pass
