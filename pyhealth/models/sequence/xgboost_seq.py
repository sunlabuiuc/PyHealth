# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os
import pickle
import warnings
import numpy as np
import json
import joblib
from xgboost import XGBClassifier
from xgboost import XGBRFRegressor
from pyhealth.utils.check import *
from pyhealth.data.data_reader.sequence import ml_reader
from sklearn.multioutput import MultiOutputClassifier
warnings.filterwarnings('ignore')

class XGBoostSequence:

    def __init__(self, 
                 expmodel_id = 'test.new', 
                 n_estimators=100, 
                 criterion='gini', 
                 max_depth=None, 
                 min_samples_split=2, 
                 min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, 
                 max_features='auto', 
                 max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, 
                 min_impurity_split=None, 
                 bootstrap=True, 
                 oob_score=False, 
                 n_jobs=None, 
                 random_state=None, 
                 verbose=0, 
                 warm_start=False, 
                 class_weight=None, 
                 ccp_alpha=0.0, 
                 max_samples=None
                ):
        """
        XGboost from public XGBoostSequence Lib.


        Parameters

        ----------

        """
        check_model_dir(expmodel_id =  expmodel_id)
        self.checkout_dir = os.path.join('./experiments_records', expmodel_id, 'checkouts')
        self.result_dir = os.path.join('./experiments_records', expmodel_id, 'results')
        # make saving directory if needed
        if not os.path.isdir(self.checkout_dir):
            os.makedirs(self.checkout_dir)
            
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        self.expmodel_id = expmodel_id
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score 
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.task_type = None
        # self._args_check()

    def _data_check(self, datalist):

        """
        
        Target to 1) check train_data/valid_data valid, if not give tips about data problem
                  2) check loss function valid, if not recommend proper loss func
        
        Parameters

        ----------

        datalist = [data1 = {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                    },
                    data2 = {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                    }, ...
                    ]
        Returns

        -------

        self : object


        """
        
        feat_n_check = set([])
        label_n_check = set([])
        task_type_check = set([])
        for each_data in datalist:
            for each_x_path in each_data['x']:
                if os.path.exists(each_x_path) is False:
                    raise Exception('episode file not exist')
            feat_n_check.add(each_data['feat_n'])
            label_n_check.add(np.shape(np.array(each_data['y']))[1])
            task_type_check.add(label_check(each_data['y'], hat_y = None, assign_task_type = self.task_type))
            
        if len(feat_n_check) != 1:
            raise Exception('feat_n is inconformity in data')
        if len(task_type_check) != 1:
            raise Exception('task_type is inconformity in data')
        
        pre_task_type = list(task_type_check)[0]
        if self.task_type == None:
            self.task_type = pre_task_type
        elif self.task_type == pre_task_type:
            pass
        else:
            raise Exception('predifine task-type {0}, but data support task-type {1}'.format(self.task_type, pre_task_type))
        print ('current task can beed seen as {0}'.format(self.task_type))

    def _build_model(self):
        """
        
        Build the crucial components for model training 
 
        
        """
        
        _config = {
            'n_estimators': self.n_estimators, 
            'criterion': self.criterion, 
            'max_depth': self.max_depth, 
            'min_samples_split': self.min_samples_split, 
            'min_samples_leaf': self.min_samples_leaf, 
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf, 
            'max_features': self.max_features, 
            'max_leaf_nodes': self.max_leaf_nodes, 
            'min_impurity_split': self.min_impurity_split, 
            'bootstrap': self.bootstrap, 
            'oob_score': self.oob_score, 
            'n_jobs': self.n_jobs, 
            'random_state': self.random_state, 
            'verbose': self.verbose, 
            'warm_start': self.warm_start, 
            'ccp_alpha': self.ccp_alpha, 
            'max_samples': self.max_samples
        }
        if self.task_type == 'binaryclass':
            self.predictor = XGBClassifier(**_config, objective='binary:logistic')
        elif self.task_type == 'multiclass':
            self.predictor = XGBClassifier(**_config)
        elif self.task_type == 'multilabel':
            xgb_estimator = XGBClassifier(**_config, objective='binary:logistic')
            self.predictor = MultiOutputClassifier(xgb_estimator)
        elif self.task_type == 'regression':
            self.predictor = XGBRFRegressor(**_config)
        self._save_config(_config, 'predictor')
        _config = {'tasktype': self.task_type}
        self._save_config(_config, 'tasktype')

    def _data_check(self, datalist):

        """
        
        Target to 1) check train_data/valid_data valid, if not give tips about data problem
                  2) check loss function valid, if not recommend proper loss func
        
        Parameters

        ----------

        datalist = [data1 = {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                    },
                    data2 = {
                      'x':list[episode_file_path], 
                      'y':list[label], 
                      'l':list[seq_len], 
                      'feat_n': n of feature space, 
                      'label_n': n of label space
                    }, ...
                    ]
        Returns

        -------

        self : object


        """

        feat_n_check = set([])
        label_n_check = set([])
        task_type_check = set([])
        for each_data in datalist:
            for each_x_path in each_data['x']:
                if os.path.exists(each_x_path) is False:
                    raise Exception('episode file not exist')
            feat_n_check.add(each_data['feat_n'])
            label_n_check.add(np.shape(np.array(each_data['y']))[1])
            task_type_check.add(label_check(each_data['y'], hat_y = None, assign_task_type = self.task_type))
            
        if len(feat_n_check) != 1:
            raise Exception('feat_n is inconformity in data')
        if len(task_type_check) != 1:
            raise Exception('task_type is inconformity in data')

        pre_task_type = list(task_type_check)[0]
        if self.task_type == None:
            self.task_type = pre_task_type
        elif self.task_type == pre_task_type:
            pass
        else:
            raise Exception('predifine task-type {0}, but data support task-type {1}'.format(self.task_type, pre_task_type))

    def fit(self, data_dict, X = None, y = None, assign_task_type = None):
        
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
        self.task_type = assign_task_type
        if data_dict != None:
            self._data_check([data_dict])
            data = ml_reader.DatasetReader(data_dict, task_type = self.task_type).get_data()
            _X = np.array(data['X'])
            _y = np.array(data['Y'])
        elif X != None and y != None:
            self._data_check([{'X': X, 'Y': Y}])
            _X = X
            _y = Y
        else:
            raise Exception('fill in correct data for model train')
        
        print (np.shape(_X),np.shape(_y))
        self._build_model()
        self.predictor.fit(_X, _y)
        model_path = os.path.join(self.checkout_dir, 'best.model')
        joblib.dump(self.predictor,  model_path)

    def _save_config(self, config, config_type):
        temp_path = os.path.join(self.checkout_dir, "{0}_config.json".format(config_type))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(config, indent=4))

    def _load_config(self, config_type):
        temp_path = os.path.join(self.checkout_dir, '{0}_config.json'.format(config_type))
        assert os.path.exists(temp_path), 'cannot find {0}_config.json, please it in dir {1}'.format(config_type, self.checkout_dir)
        with open(temp_path, 'r') as f:
            config = json.load(f)
        return config

    def load_model(self):
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
        model_path = os.path.join(self.checkout_dir, 'best.model')
        self.task_type = self._load_config('tasktype')['tasktype']
        self.predictor = joblib.load(model_path)

    def inference(self, data_dict, X = None, y = None):
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

        if data_dict != None:
            self._data_check([data_dict])
            data = ml_reader.DatasetReader(data_dict, task_type = self.task_type).get_data()
            _X = data['X']
            _y = data['Y']
        elif X != None and y != None:
            self._data_check({'X': X, 'Y': y})
            _X = X
            _y = y
        else:
            raise Exception('fill in correct data for model inference')

        if self.task_type in ['binaryclass','regression']:
            real_v = _y.reshape(-1, 1)
            prob_v = self.predictor.predict_proba(_X)[:, 1].reshape(-1, 1)
        elif self.task_type in ['multiclass']:
            real_v = np.array(_y)
            prob_v = self.predictor.predict_proba(_X).reshape(-1, np.shape(real_v)[1])
        elif self.task_type in ['multilabel']:
            real_v = np.array(_y)
            prob_v = []
            _prob_v = self.predictor.predict_proba(_X)
            for each_class in _prob_v:
                if len(each_class) == 1:
                    each_class = np.array([each_class])
                prob_v.append(each_class[:, 1:2])
            prob_v = np.concatenate(prob_v, 1)
            
        pickle.dump(prob_v, open(os.path.join(self.result_dir, 'hat_y'),'wb'))
        pickle.dump(real_v, open(os.path.join(self.result_dir, 'y'),'wb'))

    def get_results(self):
        
        """
        
        Load saved prediction results in current ExpID
            truth_value: proj_root/experiments_records/*****(exp_id)/results/y
            predict_value: proj_root/experiments_records/*****(exp_id)/results/hat_y
            xxx represents the loaded model
        
        """
        try:
            hat_y = pickle.load(open(os.path.join(self.result_dir, 'hat_y'),'rb'))
        except IOError:
            print ('Error: cannot find file {0} or load failed'.format(os.path.join(self.result_dir, 'hat_y')))
        try:
            y = pickle.load(open(os.path.join(self.result_dir, 'y'),'rb'))
        except IOError:
            print ('Error: cannot find file {0} or load failed'.format(os.path.join(self.result_dir, 'y')))

        results = {'hat_y': hat_y, 'y': y}
        
        return results
