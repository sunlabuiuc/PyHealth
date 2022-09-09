import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class BaseModel:

    def __init__(self, dataset, model, feature_mode=0):
        super(BaseModel, self).__init__()

        self.dataset = dataset
        self.feature_mode = feature_mode

        self.model = None
        self.predictor = None
        self.X = None
        self.y = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None

        self.class_num_constraint = 0

        # Import model
        if model == 'XGB':
            self.model = XGBClassifier(objective='binary:logistic', tree_method='gpu_hist')
        elif model == 'SVM':
            self.model = SVC(kernel='linear', probability=True)
            self.class_num_constraint = 1
        elif model == 'LR':
            self.model = LogisticRegression(random_state=101)
            self.class_num_constraint = 1
        elif model == 'DT':
            self.model = DecisionTreeClassifier(random_state=101)

        # TODO: Add more models

        # For different tasks, we use different loss and output format
        if dataset.task() == 'DrugRec':
            self.predictor = MultiOutputClassifier(self.model)

        # TODO: Add more tasks

        self.voc_size = dataset.voc_size

    def preprocess(self):
        X = None
        y = None

        # Do different preprocesses to different tasks
        if self.dataset.task() == 'DrugRec':
            if self.feature_mode == 0:
                X, y = self.data_normalized_codes_count(dataset=self.dataset)
            elif self.feature_mode == 1:
                X, y = self.data_multi_hot(dataset=self.dataset)
            # elif self.feature_mode == 3:
            # TODO: pretrain

        return X, y

    # One sample is one visit, regardless of patients
    @staticmethod
    def data_multi_hot(dataset):
        cons = []
        pros = []
        drgs = []

        for i in range(len(dataset)):
            conditions_ = dataset[i]['conditions']
            procedures_ = dataset[i]['procedures']
            drugs_ = dataset[i]['drugs']

            for j in range(len(conditions_)):
                condition = conditions_[j]
                tmp = np.zeros(dataset.voc_size[0])
                for k in range(len(condition)):
                    tmp[condition[k]] = 1
                cons.append(tmp)

            for m in range(len(procedures_)):
                procedure = procedures_[m]
                tmp = np.zeros(dataset.voc_size[1])
                for n in range(len(procedure)):
                    tmp[procedure[n]] = 1
                pros.append(tmp)

            for p in range(len(drugs_)):
                drug = drugs_[p]
                tmp = np.zeros(dataset.voc_size[2])
                for q in range(len(drug)):
                    tmp[drug[q]] = 1
                drgs.append(tmp)

        conditions = np.array(cons, dtype=int)
        procedures = np.array(pros, dtype=int)
        drugs = np.array(drgs, dtype=int)

        x_emb = []
        for i in range(len(conditions)):
            tmp = np.concatenate((conditions[i], procedures[i]))
            x_emb.append(tmp)

        X = np.array(x_emb, dtype=int)

        y = drugs

        return X, y

    # One sample is one patient's all visits
    # For X, we compute the counts of medical codes (conditions and procedures) for each patient based on all
    # her visits as input variables and normalize the vector to Zero Mean Unit Variance
    # For y, we multi-hot all drugs appeared in all visits of one patient, regardless of appearance times
    @staticmethod
    def data_normalized_codes_count(dataset):
        cons = []
        pros = []
        drgs = []

        for i in range(len(dataset)):
            conditions_ = dataset[i]['conditions']
            procedures_ = dataset[i]['procedures']
            drugs_ = dataset[i]['drugs']

            tmp_cons = np.zeros(dataset.voc_size[0])
            tmp_pros = np.zeros(dataset.voc_size[1])
            tmp_drgs = np.zeros(dataset.voc_size[2])

            for j in range(len(conditions_)):
                condition = conditions_[j]
                for k in range(len(condition)):
                    if condition[k] != 0:
                        tmp_cons[condition[k]] += 1

            for m in range(len(procedures_)):
                procedure = procedures_[m]
                for n in range(len(procedure)):
                    if procedure[n] != 0:
                        tmp_pros[procedure[n]] += 1

            for p in range(len(drugs_)):
                drug = drugs_[p]
                for q in range(len(drug)):
                    tmp_drgs[drug[q]] = 1

            normed_cons = (tmp_cons - tmp_cons.mean(axis=0)) / tmp_cons.std(axis=0)
            normed_pros = (tmp_pros - tmp_pros.mean(axis=0)) / tmp_pros.std(axis=0)

            cons.append(normed_cons)
            pros.append(normed_pros)
            drgs.append(tmp_drgs)

        conditions = np.array(cons, dtype=float)
        procedures = np.array(pros, dtype=float)
        drugs = np.array(drgs, dtype=int)

        x_emb = []
        for i in range(len(conditions)):
            tmp = np.concatenate((conditions[i], procedures[i]))
            x_emb.append(tmp)

        X = np.array(x_emb, dtype=float) / 2

        y = drugs

        return X, y

    def train(self, split_ratio=0.9, k=5):
        self.X, self.y = self.preprocess()
        idx = (int)(len(self.X) * split_ratio)
        self.X_train, self.X_test = self.X[:idx], self.X[idx:]
        self.y_train, self.y_test = self.y[:idx], self.y[idx:]

        val_preds = np.zeros(self.y_train.shape)
        test_preds = np.zeros((self.X_test.shape[0], self.y_test.shape[1]))
        val_losses = []
        kf = KFold(n_splits=k)

        for fn, (trn_idx, val_idx) in enumerate(kf.split(self.X_train, self.y_train)):
            print('Starting fold: ', fn)
            X_train_, X_val = self.X_train[trn_idx], self.X_train[val_idx]
            y_train_, y_val = self.y_train[trn_idx], self.y_train[val_idx]

            # For models such as SVM, LR, at least 2 classes should be detected for each column of trainning data
            # So, the existence of columns that contain all zeros would lead to an exception
            # To address this issue, we make a piece of 'unclean' data for training
            if self.class_num_constraint == 1:
                for i in range(len(y_train_[0])):
                    if y_train_[:, i].sum() == 0:
                        y_train_[0, i] = 1

            self.predictor.fit(X_train_, y_train_)
            val_pred = self.predictor.predict_proba(X_val)  # list of preds per class
            val_pred = np.array(val_pred)[:, :, 1].T  # take the positive class
            val_preds[val_idx] = val_pred

            loss = log_loss(np.ravel(y_val), np.ravel(val_pred))
            val_losses.append(loss)
            #             preds = self.predictor.predict_proba(self.X_test)
            #             preds = np.array(preds)[:, :, 1].T  # take the positive class
            #             test_preds += preds / 5

            print('Mean loss across folds: ', np.mean(val_losses))
            print('STD  loss across folds: ', np.std(val_losses))

    def predict(self, X_test=None):
        if X_test is None:
            res = self.predictor.predict(self.X_test)
            print('BCE loss: ', log_loss(res, self.y_test))
            return res
        else:
            return self.predictor.predict(X_test)
