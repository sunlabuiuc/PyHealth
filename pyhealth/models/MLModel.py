from typing import List, Tuple, Union, Dict, Optional
import pickle
import os
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from pyhealth.utils import set_logger
from pyhealth.data import BaseDataset
from pyhealth.tokenizer import Tokenizer
from sklearn.base import ClassifierMixin


class MLTask:

    def __init__(
            self,
            dataset: BaseDataset,
            tables: Union[List[str], Tuple[str]],
            target: str,
            classifier: ClassifierMixin,
            mode: str,
            enable_logging: bool = True,
            output_path: Optional[str] = None,
            exp_name: Optional[str] = None,
            **kwargs
    ):
        super(MLTask, self).__init__()

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

        elif self.mode == "binary":
            self.predictor = self.classifier

        if enable_logging:
            self.exp_path = set_logger(output_path, None)
        else:
            self.exp_path = None

    def fit(
            self,
            train_loader,
            reduce_dim=100,
            val_loader=None,
            val_metric=None,
    ):
        # X (diagnosis and procedure), y (drugs)
        X, y = [], []

        # load the X and y batch by batch
        for batch in train_loader:
            cur_X, cur_y = code2vec(self.tables, self.target, self.mode, batch, self.tokenizers, self.label_tokenizer)
            X.append(cur_X)
            y.append(cur_y)

        # train the model
        X = np.concatenate(X, axis=0)

        # PCA to 100-dim
        if reduce_dim is not None:
            self.pca = PCA(n_components=reduce_dim)
        X = self.pca.fit_transform(X)

        if self.mode == "multilabel":
            # index 0 and 1 are invalid targets
            y = np.concatenate(y, axis=0)[:, 2:]
            # obtain the valid pos of y that has both 0 and 1
            self.valid_label = np.where(y.sum(0) > 0)[0]
            print(np.shape(X), np.shape(y))
            # fit
            self.predictor.fit(X, y[:, self.valid_label])

        elif self.mode == "binary":
            y = np.concatenate(y, axis=0)
            print(np.shape(X), np.shape(y))
            # fit
            self.predictor.fit(X, y)

        # save the model
        if self.exp_path is not None:
            with open(os.path.join(self.exp_path, "best.ckpt"), "wb") as f:
                pickle.dump([self.predictor, self.pca, self.valid_label], f)
            print("best_model_path:", os.path.join(self.exp_path, "best.ckpt"))

    def __call__(
        self, tables, target, batch, padding_mask=None, device=None, **kwargs
    ):
        X, y = code2vec(tables, target, self.mode, batch, self.tokenizers, self.label_tokenizer)
        X = self.pca.transform(X)
        cur_prob = self.predictor.predict_proba(X)
        cur_prob = np.array(cur_prob)[:, :, -1].T
        y_prob = np.zeros((X.shape[0], self.label_tokenizer.get_vocabulary_size() - 2))

        y_prob[:, self.valid_label] = cur_prob

        return {"loss": 1.0, "y_prob": y_prob, "y_true": y[:, 2:]}

    def load(self, path):
        with open(path, "rb") as f:
            self.predictor, self.pca, self.valid_label = pickle.load(f)

    def eval(self, test_loader):
        X, y_true, y_prob, y_pred = [], [], [], []
        for batch in test_loader:
            cur_X, cur_y = code2vec(self.tables, self.target, self.mode, batch, self.tokenizers, self.label_tokenizer)
            X.append(cur_X)
            y_true.append(cur_y)

        X = np.concatenate(X, axis=0)
        X = self.pca.transform(X)
        cur_prob = self.predictor.predict_proba(X)
        print(cur_prob, type(cur_prob))

        if self.mode == "multilabel":
            cur_prob = np.array(cur_prob)[:, :, -1].T
            y_true = np.concatenate(y_true, axis=0)[:, 2:]
            y_prob = np.zeros((X.shape[0], self.label_tokenizer.get_vocabulary_size() - 2))
            y_prob[:, self.valid_label] = cur_prob
            y_pred = (y_prob > 0.5).astype(int)

        elif self.mode == "binary":
            y_true = np.concatenate(y_true, axis=0)
            y_gt = np.zeros([len(y_true), 2])
            for i in range(len(y_gt)):
                y_gt[y_true[i]] = 1
            y_true = y_gt
            y_prob = cur_prob
            y_pred = (y_prob > 0.5).astype(int)

        return y_true, y_prob, y_pred


class MLModel:
    """MLModel Class, use "task" as key to identify specific MLModel model and route there"""
    def __init__(self, **kwargs):
        super(MLModel, self).__init__()
        self.model = MLTask(**kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self.model.load(*args, **kwargs)


def code2vec(
        tables: Union[List[str], Tuple[str]],
        target: str,
        mode: str,
        batch: dict,
        domain_tokenizers: Dict[str, Tokenizer],
        label_tokenizer: Tokenizer = None,
):
    cur_domain = []
    for domain in tables:
        cur_tmp = np.zeros(
            (len(batch[domain]), domain_tokenizers[domain].get_vocabulary_size())
        )
        for idx, sample in enumerate(batch[domain]):
            cur_tmp[idx, domain_tokenizers[domain].convert_tokens_to_indices(sample[-1:][0])] = 1

        cur_domain.append(cur_tmp)

    cur_label = None
    if mode == "multilabel":
        cur_label = np.zeros(
            (len(batch[target]), label_tokenizer.get_vocabulary_size())
        )
        for idx, sample in enumerate(batch[target]):
            cur_label[idx, label_tokenizer.convert_tokens_to_indices(sample[-1:][0])] = 1
    elif mode == "binary":
        cur_label = batch[target]

    cur_X = np.concatenate(cur_domain, axis=1)
    cur_y = cur_label

    return cur_X, cur_y
