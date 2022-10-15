from typing import List, Tuple, Union, Dict, Optional
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from pyhealth.datasets import BaseDataset
from pyhealth.tokenizer import Tokenizer
from sklearn.base import ClassifierMixin
from itertools import chain


class ClassicML:
    def __init__(
        self,
        dataset: BaseDataset,
        tables: List[str],
        target: str,
        classifier: ClassifierMixin,
        mode: str,
        **kwargs
    ):
        """Call classical ML models
        Args:
            dataset: the dataset object
            tables: a list of table names to be used
            target: the target table name
            classifier: the classifier object from sklearn
            mode: the mode of the model, can be "multilabel", "binary", "multiclass"
        """
        super(ClassicML, self).__init__()

        self.tables = tables
        self.target = target
        self.classifier = classifier
        self.mode = mode
        self.label_tokenizer = None
        self.valid_label = None
        self.predictor = None
        self.pca = None

        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain), special_tokens=["<pad>", "<unk>"]
            )

        if self.mode == "multilabel":
            self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))
            self.valid_label = np.zeros(self.label_tokenizer.get_vocabulary_size())
            self.predictor = MultiOutputClassifier(self.classifier)

        elif self.mode in ["binary", "multiclass"]:
            self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))
            self.predictor = self.classifier

    @staticmethod
    def code2vec(self, **kwargs):
        """convert the batch medical codes to vectors
        Args:
            **kwargs: the key-value pair of batch data

        Parameters
        ----------
        self
        """
        batch_X = []
        for domain in self.tables:
            cur_X = np.zeros(
                (len(kwargs[domain]), self.tokenizers[domain].get_vocabulary_size())
            )
            for idx, sample in enumerate(kwargs[domain]):
                if type(sample[0]) == list:
                    sample = np.unique(list(chain(*sample)))
                cur_X[
                    idx, self.tokenizers[domain].convert_tokens_to_indices(sample)
                ] = 1

            batch_X.append(cur_X)

        if self.mode in ["multilabel"]:
            kwargs[self.target] = self.label_tokenizer.batch_encode_2d(
                kwargs[self.target], padding=False, truncation=False
            )
            batch_y = np.zeros(
                (len(kwargs[self.target]), self.label_tokenizer.get_vocabulary_size())
            )
            for idx, sample in enumerate(kwargs[self.target]):
                batch_y[idx, sample] = 1

        elif self.mode in ["binary", "multiclass"]:
            batch_y = self.label_tokenizer.convert_tokens_to_indices(
                kwargs[self.target]
            )
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        return np.concatenate(batch_X, axis=1), batch_y

    def fit(
        self,
        train_loader,
        reduce_dim=100,
        val_loader=None,
        val_metric=None,
    ):
        """fit the classical ML model
        train_loader: the train data loader
        reduce_dim: the dimension of the reduced data
        val_loader (not used): the validation data loader
        val_metric (not used): the validation metric
        """
        # X (diagnosis and procedure), y (drugs)
        X, y = [], []

        # load the X and y batch by batch
        for batch in train_loader:
            cur_X, cur_y = self.code2vec(self, **batch)
            X.append(cur_X)
            y.append(cur_y)

        # train the model
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        print("X and y shape:", np.shape(X), np.shape(y))

        # PCA to 100-dim
        if reduce_dim is not None:
            self.pca = PCA(n_components=reduce_dim)
        X = self.pca.fit_transform(X)

        if self.mode == "multilabel":
            # obtain the valid pos of y that has both 0 and 1
            self.valid_label = np.where(y.sum(0) > 0)[0]
            # fit
            self.predictor.fit(X, y[:, self.valid_label])

        elif self.mode in ["binary", "multiclass"]:
            # fit
            self.predictor.fit(X, y)
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def __call__(self, **kwargs):
        """predict the batch data"""

        X, y = self.code2vec(self, **kwargs)
        X = self.pca.transform(X)
        cur_prob = self.predictor.predict_proba(X)
        y_prob = np.zeros((X.shape[0], self.label_tokenizer.get_vocabulary_size()))
        y_true = y

        if self.mode == "multilabel":
            cur_prob = np.array(cur_prob)[:, :, -1].T
            y_prob[:, self.valid_label] = cur_prob
            y_pred = (y_prob > 0.5).astype(int)

        elif self.mode == "binary":
            y_prob = cur_prob
            y_pred = (y_prob > 0.5).astype(int)
            y_true = np.zeros([len(y), 2])
            for i in range(len(y_true)):
                y_true[i][y[i]] = 1

        elif self.mode == "multiclass":
            y_prob = cur_prob
            y_pred = np.argmax(y_prob, axis=1)[0]

        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        return {
            "loss": 1.0,
            "y_prob": y_prob,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def load(self, path):
        with open(path, "rb") as f:
            self.predictor, self.pca, self.valid_label = pickle.load(f)
