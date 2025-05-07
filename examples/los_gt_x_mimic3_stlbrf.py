from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import get_dataloader
from pyhealth.models import STLBRF, CaliForest
from pyhealth.tasks.length_of_stay_prediction import (
    LengthOfStayGreaterThanXPredictionMIMIC3,
)
from sklearn.datasets import make_hastie_10_2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import get_dataloader
import numpy as np


def preprocess_data():
    """
    kwargs = {
        'hadm_id': ['148506', '125387'],
        'patient_id': ['20277', '20277'],
        'conditions': tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]]),
        'procedures': tensor([[ 29,  13,  40,   0,   0,   0,   0,   0,   0], [  3,  32,  21,  33,  34,  35,  36,  37,  38]]),
        'drugs': tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  8,  7,  5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  8,  7,  5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]),
        'mortality': tensor([0., 1.])
    }
    feature_keys = ['conditions', 'procedures', 'drugs']
    """
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        # code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: set task
    sample_dataset = base_dataset.set_task(
        LengthOfStayGreaterThanXPredictionMIMIC3(threshold=3)
    )
    total_samples = len(sample_dataset.samples)
    print(f"Total samples: {total_samples}")

    train_dataloader = get_dataloader(
        sample_dataset, batch_size=len(sample_dataset.samples), shuffle=True
    )
    all_data = next(iter(train_dataloader))

    feature_keys = list(sample_dataset.input_schema.keys())
    label_keys = list(sample_dataset.output_schema.keys())

    X = np.concatenate([all_data[k].numpy() for k in feature_keys], axis=1)
    labels = [all_data[key].numpy() for key in label_keys]
    y = np.array(labels).reshape(-1, 1).ravel()

    return X, y


def hastie_data():
    np.random.seed(42)
    poly = PolynomialFeatures()
    X, y = make_hastie_10_2(n_samples=10000)
    X = poly.fit_transform(X)
    y[y < 0] = 0
    return X, y


if __name__ == "__main__":
    X, y = preprocess_data()

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # STEP 3: define model
    model = STLBRF()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # f1 score
    f1 = roc_auc_score(y_test, y_pred)
    print(f"roc_auc score: {f1}")
