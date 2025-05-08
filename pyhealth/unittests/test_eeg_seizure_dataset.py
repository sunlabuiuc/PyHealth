import pickle
import numpy as np
import pytest

from pyhealth.datasets.eeg_seizure_dataset import EEGSeizureDataset


def make_dummy(tmp_path):
    arr = np.zeros((5, 19, 800), dtype=float)
    data = {
        "eeg": arr,
        "label": np.array([0, 1, 0, 1, 0], dtype=int),
        "tag": np.zeros(5),
        "subj": np.array([1, 1, 2, 2, 3], dtype=int),
    }
    p = tmp_path / "dummy.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    return str(p)


def test_length_and_getitem(tmp_path):
    fp = make_dummy(tmp_path)
    ds = EEGSeizureDataset(fp)

    assert len(ds) == 5

    X, y, subj = ds[1]
    assert isinstance(X, np.ndarray)
    assert X.shape == (19, 800)
    assert y == 1
    assert subj == 1
