import os
import csv
import pickle
import random
import numpy as np
import pandas as pd
import tqdm
from tqdm._tqdm import trange
import time
from ..utils.checklist import check_expdata_dir

class cms:

    def __init__(self, expdata_id, root_dir='.'):

        """
        experiment data generat class for cms datasets 


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
  
        """
        self.expdata_id = expdata_id
        check_expdata_dir(expdata_id =  expdata_id)
        self.root_dir = root_dir
        self.expdata_dir = os.path.join(self.root_dir, 'experiments_data', self.expdata_id)

        print(
            'Current ExpData_ID: {0} --- Target for CMS'.format(
                self.expdata_id))

    def get_exp_data(self, 
                     sel_task='phenotyping', 
                     shuffle=True, 
                     split_ratio=[0.64, 0.16, 0.2],
                     data_root = '',
                     n_limit = -1):
        """
        Parameters

        ----------
             
        task : str, optional (default='phenotyping')
            name of current healthcare task
 
        shuffle : bool, optional (default=True) 
            determine whether shuffle data or not
            
        split_ratio : list, optional (default=[0.64,0.16,0.2])
            used for split whole data into train/valid/test
        
        data_root : str, optional (default='')
            if data_root=='', use data in ./datasets; else use data in data_root
        
        n_limit : int, optional (default = -1)
            used for sample N-data not for all data, if n_limit==-1, use all data 
        """
        self.sel_task = sel_task

        if data_root == '':
            data_root = os.path.join(self.root_dir, 'datasets')

        all_list = []
        l_list = []
        episode_dir = os.path.join(data_root, 'cms', 'x_data')
        feat_n, label_n = 0, 0
        label_seq = pd.read_csv(os.path.join(data_root, 'cms', 'y_data',
                               self.sel_task + '.csv')).values
        for row_id in trange(len(label_seq)):
            if n_limit>0 and row_id>n_limit:
                break
            time.sleep(0.01)
            row = label_seq[row_id, :]
            concrete_path = os.path.join(episode_dir, row[0])
            if os.path.exists(concrete_path) is False:
                continue
            seq_l, feat_n_all = pd.read_csv(concrete_path).shape
            if seq_l < 2:
                continue
            all_list.append([concrete_path] + [seq_l] + row[1:].astype(float).tolist())
            label_n = len(row[1:])        
        feat_n = feat_n_all - 1
        # shuffle the list
        if shuffle:
            random.shuffle(all_list)
        N = len(all_list)
        x_list = []
        y_list = []
        l_list = []
        for item in all_list:
            x_list.append(item[0])
            l_list.append(item[1])
            y_list.append(np.array(item[2:]).astype(float))

        train_ratio = split_ratio[0]
        valid_ratio = split_ratio[1]

        training_x = x_list[: int(train_ratio * N)]
        validing_x = x_list[int(train_ratio * N): int(
            (train_ratio + valid_ratio) * N)]
        testing_x = x_list[int((train_ratio + valid_ratio) * N):]

        training_y = y_list[: int(train_ratio * N)]
        validing_y = y_list[int(train_ratio * N): int(
            (train_ratio + valid_ratio) * N)]
        testing_y = y_list[int((train_ratio + valid_ratio) * N):]

        training_l = l_list[: int(train_ratio * N)]
        validing_l = l_list[int(train_ratio * N): int(
            (train_ratio + valid_ratio) * N)]
        testing_l = l_list[int((train_ratio + valid_ratio) * N):]

        if os.path.exists(self.expdata_dir) is False:
            os.makedirs(self.expdata_dir)

        pickle.dump(training_x, open(
            os.path.join(self.expdata_dir, 'train_x.pkl'), 'wb'))
        pickle.dump(validing_x, open(
            os.path.join(self.expdata_dir, 'valid_x.pkl'), 'wb'))
        pickle.dump(testing_x, open(
            os.path.join(self.expdata_dir, 'test_x.pkl'), 'wb'))
        print ('finished X generate')
        pickle.dump(training_y, open(
            os.path.join(self.expdata_dir, 'train_y.pkl'), 'wb'))
        pickle.dump(validing_y, open(
            os.path.join(self.expdata_dir, 'valid_y.pkl'), 'wb'))
        pickle.dump(testing_y, open(
            os.path.join(self.expdata_dir, 'test_y.pkl'), 'wb'))
        print ('finished Y generate')
        pickle.dump(training_l, open(
            os.path.join(self.expdata_dir, 'train_l.pkl'), 'wb'))
        pickle.dump(validing_l, open(
            os.path.join(self.expdata_dir, 'valid_l.pkl'), 'wb'))
        pickle.dump(testing_l, open(
            os.path.join(self.expdata_dir, 'test_l.pkl'), 'wb'))
        print ('finished L generate')

        expdata_statistic = {
            'task':self.sel_task,
            'raio': split_ratio,
            'feat_n': feat_n,
            'label_n': label_n,
            'len_train': len(training_x),
            'len_valid': len(validing_x),
            'len_test': len(testing_x)            
        }
        pickle.dump(expdata_statistic, open(
            os.path.join(self.expdata_dir, 'expdata_statistic.pkl'), 'wb'))

        self.train = {'x': training_x, 'y': training_y, 'l': training_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.valid = {'x': validing_x, 'y': validing_y, 'l': validing_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.test = {'x': testing_x, 'y': testing_y, 'l': testing_l,
                     'feat_n': feat_n, 'label_n': label_n}

        print('generate finished')
        print('target Task:', expdata_statistic['task'])
        print('N of features:', expdata_statistic['feat_n'])
        print('N of labels:', expdata_statistic['label_n'])        
        print('N of TrainData:', expdata_statistic['len_train'])
        print('N of ValidData:', expdata_statistic['len_valid'])
        print('N of TestData:', expdata_statistic['len_test'])

    def load_exp_data(self):
        if os.path.exists(self.expdata_dir) is False:
            raise Exception('cannot find exp data dir {0}'.format(self.expdata_dir))

        training_x = pickle.load(open(
            os.path.join(self.expdata_dir, 'train_x.pkl'), 'rb'))
        validing_x = pickle.load(open(
            os.path.join(self.expdata_dir, 'valid_x.pkl'), 'rb'))
        testing_x = pickle.load(open(
            os.path.join(self.expdata_dir, 'test_x.pkl'), 'rb'))

        training_y = pickle.load(open(
            os.path.join(self.expdata_dir, 'train_y.pkl'), 'rb'))
        validing_y = pickle.load(open(
            os.path.join(self.expdata_dir, 'valid_y.pkl'), 'rb'))
        testing_y = pickle.load(open(
            os.path.join(self.expdata_dir, 'test_y.pkl'), 'rb'))

        training_l = pickle.load(open(
            os.path.join(self.expdata_dir, 'train_l.pkl'), 'rb'))
        validing_l = pickle.load(open(
            os.path.join(self.expdata_dir, 'valid_l.pkl'), 'rb'))
        testing_l = pickle.load(open(
            os.path.join(self.expdata_dir, 'test_l.pkl'), 'rb'))

        expdata_statistic = pickle.load(open(
            os.path.join(self.expdata_dir, 'expdata_statistic.pkl'), 'rb'))

        feat_n = expdata_statistic['feat_n']
        label_n = expdata_statistic['label_n']
        self.train = {'x': training_x, 'y': training_y, 'l': training_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.valid = {'x': validing_x, 'y': validing_y, 'l': validing_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.test = {'x': testing_x, 'y': testing_y, 'l': testing_l,
                     'feat_n': feat_n, 'label_n': label_n}

        print('load finished')
        print('target Task:', expdata_statistic['task'])
        print('N of features:', expdata_statistic['feat_n'])
        print('N of labels:', expdata_statistic['label_n'])        
        print('N of TrainData:', expdata_statistic['len_train'])
        print('N of ValidData:', expdata_statistic['len_valid'])
        print('N of TestData:', expdata_statistic['len_test'])

    def show_data(self, k=3):
        """
        Parameters

        ----------
        k : int, optional (default=3) 
            fetch k sample data for show
            
  
        """

        print('------------Train--------------')
        print('x_data', self.train['x'][:k])
        print('y_data', self.train['y'][:k])
        print('l_data', self.train['l'][:k])
        print('------------Valid--------------')
        print('x_data', self.valid['x'][:k])
        print('y_data', self.valid['y'][:k])
        print('l_data', self.valid['l'][:k])
        print('------------Test--------------')
        print('x_data', self.test['x'][:k])
        print('y_data', self.test['y'][:k])
        print('l_data', self.test['l'][:k])


class mimic:

    def __init__(self, expdata_id, root_dir='.'):

        """
        experiment data generat class for cms datasets 


        Parameters

        ----------
        exp_id : str, optional (default='init.test') 
            name of current experiment
  
        """
        self.expdata_id = expdata_id
        check_expdata_dir(expdata_id =  expdata_id)
        self.root_dir = root_dir
        self.expdata_dir = os.path.join(self.root_dir, 'experiments_data', self.expdata_id)

        print(
            'Current ExpData_ID: {0} --- Target for MIMIC'.format(
                self.expdata_id))

    def get_exp_data(self, 
                     sel_task='phenotyping', 
                     shuffle=True, 
                     split_ratio=[0.64, 0.16, 0.2],
                     data_root = '',
                     n_limit = -1):
        """
        Parameters

        ----------
             
        task : str, optional (default='phenotyping')
            name of current healthcare task
 
        shuffle : bool, optional (default=True) 
            determine whether shuffle data or not
            
        split_ratio : list, optional (default=[0.64,0.16,0.2])
            used for split whole data into train/valid/test
        
        data_root : str, optional (default='')
            if data_root=='', use data in ./datasets; else use data in data_root
        
        n_limit : int, optional (default = -1)
            used for sample N-data not for all data, if n_limit==-1, use all data 
        """
        self.sel_task = sel_task

        if data_root == '':
            data_root = os.path.join(self.root_dir, 'datasets')

        all_list = []
        l_list = []
        episode_dir = os.path.join(data_root, 'mimic', 'x_data')
        feat_n, label_n = 0, 0
        label_seq = pd.read_csv(os.path.join(data_root, 'mimic', 'y_data',
                               self.sel_task + '.csv')).values
        for row_id in trange(len(label_seq)):
            if n_limit>0 and row_id>n_limit:
                break
            time.sleep(0.01)
            row = label_seq[row_id, :]
            concrete_path = os.path.join(episode_dir, row[0])
            if os.path.exists(concrete_path) is False:
                continue
            seq_l, feat_n_all = pd.read_csv(concrete_path).shape
            if seq_l < 2:
                continue
            all_list.append([concrete_path] + [seq_l] + row[1:].astype(float).tolist())
            label_n = len(row[1:])        
        feat_n = feat_n_all - 1
        # shuffle the list
        if shuffle:
            random.shuffle(all_list)
        N = len(all_list)
        x_list = []
        y_list = []
        l_list = []
        for item in all_list:
            x_list.append(item[0])
            l_list.append(item[1])
            y_list.append(np.array(item[2:]).astype(float))

        train_ratio = split_ratio[0]
        valid_ratio = split_ratio[1]

        training_x = x_list[: int(train_ratio * N)]
        validing_x = x_list[int(train_ratio * N): int(
            (train_ratio + valid_ratio) * N)]
        testing_x = x_list[int((train_ratio + valid_ratio) * N):]

        training_y = y_list[: int(train_ratio * N)]
        validing_y = y_list[int(train_ratio * N): int(
            (train_ratio + valid_ratio) * N)]
        testing_y = y_list[int((train_ratio + valid_ratio) * N):]

        training_l = l_list[: int(train_ratio * N)]
        validing_l = l_list[int(train_ratio * N): int(
            (train_ratio + valid_ratio) * N)]
        testing_l = l_list[int((train_ratio + valid_ratio) * N):]

        if os.path.exists(self.expdata_dir) is False:
            os.makedirs(self.expdata_dir)

        pickle.dump(training_x, open(
            os.path.join(self.expdata_dir, 'train_x.pkl'), 'wb'))
        pickle.dump(validing_x, open(
            os.path.join(self.expdata_dir, 'valid_x.pkl'), 'wb'))
        pickle.dump(testing_x, open(
            os.path.join(self.expdata_dir, 'test_x.pkl'), 'wb'))
        print ('finished X generate')
        pickle.dump(training_y, open(
            os.path.join(self.expdata_dir, 'train_y.pkl'), 'wb'))
        pickle.dump(validing_y, open(
            os.path.join(self.expdata_dir, 'valid_y.pkl'), 'wb'))
        pickle.dump(testing_y, open(
            os.path.join(self.expdata_dir, 'test_y.pkl'), 'wb'))
        print ('finished Y generate')
        pickle.dump(training_l, open(
            os.path.join(self.expdata_dir, 'train_l.pkl'), 'wb'))
        pickle.dump(validing_l, open(
            os.path.join(self.expdata_dir, 'valid_l.pkl'), 'wb'))
        pickle.dump(testing_l, open(
            os.path.join(self.expdata_dir, 'test_l.pkl'), 'wb'))
        print ('finished L generate')

        expdata_statistic = {
            'task':self.sel_task,
            'raio': split_ratio,
            'feat_n': feat_n,
            'label_n': label_n,
            'len_train': len(training_x),
            'len_valid': len(validing_x),
            'len_test': len(testing_x)            
        }
        pickle.dump(expdata_statistic, open(
            os.path.join(self.expdata_dir, 'expdata_statistic.pkl'), 'wb'))

        self.train = {'x': training_x, 'y': training_y, 'l': training_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.valid = {'x': validing_x, 'y': validing_y, 'l': validing_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.test = {'x': testing_x, 'y': testing_y, 'l': testing_l,
                     'feat_n': feat_n, 'label_n': label_n}

        print('generate finished')
        print('target Task:', expdata_statistic['task'])
        print('N of features:', expdata_statistic['feat_n'])
        print('N of labels:', expdata_statistic['label_n'])        
        print('N of TrainData:', expdata_statistic['len_train'])
        print('N of ValidData:', expdata_statistic['len_valid'])
        print('N of TestData:', expdata_statistic['len_test'])

    def load_exp_data(self):
        if os.path.exists(self.expdata_dir) is False:
            raise Exception('cannot find exp data dir {0}'.format(self.expdata_dir))

        training_x = pickle.load(open(
            os.path.join(self.expdata_dir, 'train_x.pkl'), 'rb'))
        validing_x = pickle.load(open(
            os.path.join(self.expdata_dir, 'valid_x.pkl'), 'rb'))
        testing_x = pickle.load(open(
            os.path.join(self.expdata_dir, 'test_x.pkl'), 'rb'))

        training_y = pickle.load(open(
            os.path.join(self.expdata_dir, 'train_y.pkl'), 'rb'))
        validing_y = pickle.load(open(
            os.path.join(self.expdata_dir, 'valid_y.pkl'), 'rb'))
        testing_y = pickle.load(open(
            os.path.join(self.expdata_dir, 'test_y.pkl'), 'rb'))

        training_l = pickle.load(open(
            os.path.join(self.expdata_dir, 'train_l.pkl'), 'rb'))
        validing_l = pickle.load(open(
            os.path.join(self.expdata_dir, 'valid_l.pkl'), 'rb'))
        testing_l = pickle.load(open(
            os.path.join(self.expdata_dir, 'test_l.pkl'), 'rb'))

        expdata_statistic = pickle.load(open(
            os.path.join(self.expdata_dir, 'expdata_statistic.pkl'), 'rb'))

        feat_n = expdata_statistic['feat_n']
        label_n = expdata_statistic['label_n']
        self.train = {'x': training_x, 'y': training_y, 'l': training_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.valid = {'x': validing_x, 'y': validing_y, 'l': validing_l,
                      'feat_n': feat_n, 'label_n': label_n}
        self.test = {'x': testing_x, 'y': testing_y, 'l': testing_l,
                     'feat_n': feat_n, 'label_n': label_n}

        print('load finished')
        print('target Task:', expdata_statistic['task'])
        print('N of features:', expdata_statistic['feat_n'])
        print('N of labels:', expdata_statistic['label_n'])        
        print('N of TrainData:', expdata_statistic['len_train'])
        print('N of ValidData:', expdata_statistic['len_valid'])
        print('N of TestData:', expdata_statistic['len_test'])

    def show_data(self, k=3):
        """
        Parameters

        ----------
        k : int, optional (default=3) 
            fetch k sample data for show
            
  
        """

        print('------------Train--------------')
        print('x_data', self.train['x'][:k])
        print('y_data', self.train['y'][:k])
        print('l_data', self.train['l'][:k])
        print('------------Valid--------------')
        print('x_data', self.valid['x'][:k])
        print('y_data', self.valid['y'][:k])
        print('l_data', self.valid['l'][:k])
        print('------------Test--------------')
        print('x_data', self.test['x'][:k])
        print('y_data', self.test['y'][:k])
        print('l_data', self.test['l'][:k])
