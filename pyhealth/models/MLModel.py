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
        super(MLTask, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            classifier=classifier,
            mode=mode
        )

        self.tables = tables
        self.target = target
        self.classifier = classifier
        self.predictor = MultiOutputClassifier(self.classifier)
        self.pca = None

        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain), special_tokens=["<pad>", "<unk>"]
            )

        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))

        self.valid_label = np.zeros(self.label_tokenizer.get_vocabulary_size())

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
            cur_X, cur_y = code2vec(self.tables, self.target, self.tokenizers, self.label_tokenizer, batch)
            X.append(cur_X)
            y.append(cur_y)

        # train the model
        X = np.concatenate(X, axis=0)
        # index 0 and 1 are invalid drugs
        y = np.concatenate(y, axis=0)[:, 2:]

        # obtain the valid pos of y that has both 0 and 1
        self.valid_label = np.where(y.sum(0) > 0)[0]

        print(X.shape, y.shape)

        # PCA to 100-dim
        if reduce_dim is not None:
            self.pca = PCA(n_components=reduce_dim)
        X = self.pca.fit_transform(X)

        # fit
        self.predictor.fit(X, y[:, self.valid_label])

        # save the model
        if self.exp_path is not None:
            with open(os.path.join(self.exp_path, "best.ckpt"), "wb") as f:
                pickle.dump([self.predictor, self.pca, self.valid_label], f)
            print("best_model_path:", os.path.join(self.exp_path, "best.ckpt"))

    def __call__(
        self, tables, target, tokenizers, label_tokenizer, batch, padding_mask=None, device=None, **kwargs
    ):
        X, y = code2vec(tables, target, tokenizers, label_tokenizer, batch)
        X = self.pca.transform(X)
        cur_prob = self.predictor.predict_proba(X)
        cur_prob = np.array(cur_prob)[:, :, -1].T
        y_prob = np.zeros((X.shape[0], self.label_tokenizer.get_vocabulary_size() - 2))

        y_prob[:, self.valid_label] = cur_prob

        return {"loss": 1.0, "y_prob": y_prob, "y_true": y[:, 2:]}

    def load(self, path):
        with open(path, "rb") as f:
            self.predictor, self.pca, self.valid_label = pickle.load(f)


class MLModel:
    """MLModel Class, use "task" as key to identify specific MLModel model and route there"""
    def __init__(self, **kwargs):
        super(MLModel, self).__init__()
        self.model = MLTask(**kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self.model.load(*args, **kwargs)


def code2vec(
        tables: Union[List[str], Tuple[str]],
        target: str,
        domain_tokenizers: Dict[str, Tokenizer],
        label_tokenizer: Tokenizer,
        batch: dict,
):
    cur_domain = []
    for domain in tables:
        cur_tmp = np.zeros(
            (len(batch[domain]), domain_tokenizers[domain].get_vocabulary_size())
        )
        for idx, sample in enumerate(batch[domain]):
            cur_tmp[idx, domain_tokenizers[domain].convert_tokens_to_indices(sample[-1:][0])] = 1

        cur_domain.append(cur_tmp)

    cur_label = np.zeros(
        (len(batch[target]), label_tokenizer.get_vocabulary_size())
    )
    for idx, sample in enumerate(batch[target]):
        cur_label[idx, label_tokenizer.convert_tokens_to_indices(sample[-1:][0])] = 1

    cur_X = None
    if len(cur_domain) == 2:
        cur_X = np.concatenate([cur_domain[0], cur_domain[1]], axis=1)
    else:
        for i in range(len(cur_domain)):
            if cur_X is None:
                cur_X = np.concatenate([cur_domain[i], cur_domain[i + 1]], axis=1)
            else:
                cur_X = np.concatenate([cur_X, cur_domain[i + 1]], axis=1)

    cur_y = cur_label

    return cur_X, cur_y
