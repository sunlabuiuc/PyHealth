import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class BaseModel:

    def __init__(self, dataset, model, emb_dim=64):
        super(BaseModel, self).__init__()

        self.dataset = dataset
        self.emb_dim = emb_dim

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
        if model == 'XGBoost':
            self.model = XGBClassifier(objective='binary:logistic', tree_method='gpu_hist')
        elif model == 'SVM':
            self.model = SVC(kernel='linear', probability=True)

        # TODO: Add more models

        # For different tasks, we use different loss and output format
        if dataset.task() == 'DrugRec':
            self.predictor = MultiOutputClassifier(self.model, n_jobs=-1)

        # TODO: Add more tasks

        self.voc_size = dataset.voc_size

    def preprocess(self):
        X = None
        y = None

        # Do different preprocesses to different tasks
        if self.dataset.task() == 'DrugRec':
            cons = []
            pros = []
            drgs = []

            for i in range(len(self.dataset)):
                conditions_ = self.dataset[i]['conditions']
                procedures_ = self.dataset[i]['procedures']
                drugs_ = self.dataset[i]['drugs']

                for j in range(len(conditions_)):
                    condition = conditions_[j]
                    tmp = np.zeros(self.voc_size[0])
                    for k in range(len(condition)):
                        tmp[condition[k]] = 1
                    cons.append(tmp)

                for m in range(len(procedures_)):
                    procedure = procedures_[m]
                    tmp = np.zeros(self.voc_size[1])
                    for n in range(len(procedure)):
                        tmp[procedure[n]] = 1
                    pros.append(tmp)

                for p in range(len(drugs_)):
                    drug = procedures_[p]
                    tmp = np.zeros(self.voc_size[2])
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

    def train(self, split_ratio=0.9):
        self.X, self.y = self.preprocess()
        idx = (int)(len(self.X) * split_ratio)
        self.X_train, self.X_test = self.X[:idx], self.X[idx:]
        self.y_train, self.y_test = self.y[:idx], self.y[idx:]

        val_preds = np.zeros(self.y_train.shape)
        test_preds = np.zeros((self.X_test.shape[0], self.y_test.shape[1]))
        val_losses = []
        kf = KFold(n_splits=5)

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
            preds = self.predictor.predict_proba(self.X_test)
            preds = np.array(preds)[:, :, 1].T  # take the positive class
            test_preds += preds / 5

            print('Mean loss across folds: ', np.mean(val_losses))
            print('STD  loss across folds: ', np.std(val_losses))

    def predict(self, X_test=None):
        if X_test is None:
            res = self.predictor.predict(self.X_test)
            print('BCE loss: ', log_loss(res, self.y_test))
            return res
        else:
            return self.predictor.predict(X_test)
