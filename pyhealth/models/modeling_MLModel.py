from logging.config import valid_ident
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class MLDrugRecommendation:
    def __init__(self, classifier, voc_size, tokenizers):
        super(MLDrugRecommendation, self).__init__()

        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]

        self.classifier = classifier
        self.predictor = MultiOutputClassifier(self.classifier)

        # valid y index (store the pos that has both 0 and 1)
        self.valid_label = np.zeros(self.drug_tokenizer.get_vocabulary_size())

    def _code2vec(self, conditions, procedures, drugs):
        cur_diag = np.zeros(
            (len(conditions), self.condition_tokenizer.get_vocabulary_size())
        )
        for idx, sample in enumerate(conditions):
            cur_diag[idx, self.condition_tokenizer(sample[-1:])[0]] = 1

        cur_prod = np.zeros(
            (len(procedures), self.procedure_tokenizer.get_vocabulary_size())
        )
        for idx, sample in enumerate(procedures):
            cur_prod[idx, self.procedure_tokenizer(sample[-1:])[0]] = 1

        cur_drug = np.zeros((len(drugs), self.drug_tokenizer.get_vocabulary_size()))
        for idx, sample in enumerate(drugs):
            cur_drug[idx, self.drug_tokenizer(sample[-1:])[0]] = 1

        return np.concatenate([cur_diag, cur_prod], axis=1), cur_drug

    def fit(self, train_loader, evaluate_fn=None, eval_loader=None, monitor=None):
        # X (diagnosis and procedure), y (drugs)
        X, y = [], []

        # load the X and y batch by batch
        for batch in train_loader:
            conditions = batch["conditions"]
            procedures = batch["procedures"]
            drugs = batch["drugs"]
            cur_X, cur_y = self._code2vec(conditions, procedures, drugs)

            X.append(cur_X)
            y.append(cur_y)

        # train the model
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        # obtain the valid pos of y that has both 0 and 1
        self.valid_label = np.where(y.sum(0) > 0)[0]
        self.predictor.fit(X, y[:, self.valid_label])

    def __call__(
        self, conditions, procedures, drugs, padding_mask=None, device=None, **kwargs
    ):
        X, y = self._code2vec(conditions, procedures, drugs)
        cur_prob = self.predictor.predict_proba(X)
        cur_prob = np.array(cur_prob)[:, :, -1].T
        y_prob = np.zeros((X.shape[0], self.drug_tokenizer.get_vocabulary_size()))

        y_prob[:, self.valid_label] = cur_prob

        return {"loss": 1.0, "y_prob": y_prob, "y_true": y}

class MLModel:
    """MLModel Class, use "task" as key to identify specific MLModel model and route there"""

    def __init__(self, **kwargs):
        super(MLModel, self).__init__()
        task = kwargs["task"]
        kwargs.pop("task")
        if task == "drug_recommendation":
            self.model = MLDrugRecommendation(**kwargs)
        else:
            raise NotImplementedError

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
