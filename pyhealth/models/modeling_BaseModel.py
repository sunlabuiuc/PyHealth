import numpy as np
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import torch


class BaseModel:

    def __init__(self, dataset, model, emb_dim=64):
        super(BaseModel, self).__init__()

        self.dataset = dataset
        self.emb_dim = emb_dim

        self.model = None
        self.predictor = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None

        # Import model
        if model == 'XGBoost':
            self.model = XGBClassifier()
        elif model == 'SVM':
            self.model = LinearSVC()

        # TODO: Add more models

        # For different tasks, we use different loss and output format
        if dataset.task() == 'DrugRec':
            self.model = self.model(objective='binary:logistic', tree_method='gpu_hist')
            self.predictor = MultiOutputClassifier(self.model)

        # TODO: Add more tasks

        self.voc_size = dataset.voc_size

        self.condition_embedding = nn.Sequential(
            nn.Embedding(self.voc_size[0], 64, padding_idx=0),
            nn.Dropout(0.5)
        )

        self.procedure_embedding = nn.Sequential(
            nn.Embedding(self.voc_size[1], 64, padding_idx=0),
            nn.Dropout(0.5)
        )

    def preprocess(self):
        X = None
        y = None

        # Do different preprocesses to different tasks
        if self.dataset.task() == 'DrugRec':
            visit_embs = []
            for i in range(len(self.dataset)):
                # visit embedding
                condition_emb = self.condition_embedding(self.dataset[i]['conditions']).sum(dim=1).data
                procedure_emb = self.procedure_embedding(self.dataset[i]['procedures']).sum(dim=1).data
                visit_embs.append(condition_emb + procedure_emb)

            x_emb = []
            y_emb = []
            for patient in range(len(visit_embs)):
                for visit in range(len(visit_embs[patient])):
                    x_emb.append(visit_embs[patient][visit].numpy())

                    # drug multi-hot
                    drugs_index = self.dataset[patient]['drugs'][visit]
                    drugs_multihot = torch.zeros(1, self.voc_size[2])
                    drugs_multihot[0][drugs_index] = 1
                    y_emb.append(drugs_multihot[0].numpy())

            X = np.array(x_emb, dtype=float)
            y = np.array(y_emb, dtype=int)

        return X, y

    def train(self, split_ratio=0.9):
        X, y = self.preprocess()
        idx = (int)(len(X) * split_ratio)
        self.X_train, self.X_test = X[:idx], X[idx:]
        self.y_train, self.y_test = y[:idx], y[idx:]

        val_preds = np.zeros(self.y_train.shape)
        test_preds = np.zeros((self.X_test.shape[0], self.y_test.shape[1]))
        val_losses = []
        kf = KFold(n_splits=5)

        for fn, (trn_idx, val_idx) in enumerate(kf.split(self.X_train, self.y_train)):
            print('Starting fold: ', fn)
            X_train_, X_val = self.X_train[trn_idx], self.X_train[val_idx]
            y_train_, y_val = self.y_train[trn_idx], self.y_train[val_idx]

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

    def predict(self, X_test):
        if X_test is None:
            res = self.predictor.predict(self.X_test)
            print('BCE loss: ', log_loss(res, self.y_test))
            return res
        else:
            return self.predictor.predict(X_test)
