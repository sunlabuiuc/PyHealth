import pickle
import numpy as np
import pytest

from pyhealth.tasks.seizure_detection_task import SeizureDetectionTask


def make_dummy(tmp_path):
    arr = np.random.randn(20, 19, 800).astype(float)
    labels = np.array([0, 1] * 10, dtype=int)
    tags = np.zeros(20)
    subjs = np.repeat(np.arange(4), 5)  # 4 subjects × 5 samples each
    data = {"eeg": arr, "label": labels, "tag": tags, "subj": subjs}
    p = tmp_path / "dummy.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    return str(p)


def test_task_runs(tmp_path):
    pkl = make_dummy(tmp_path)

    # holdout 1 subject for test, 1 epoch only
    task = SeizureDetectionTask(pkl, holdout_subjects=1, epochs=1, batch_size=4)
    history = task.train()
    assert "loss" in history and "acc" in history

    metrics = task.evaluate()
    assert "test_acc" in metrics
    assert 0.0 <= metrics["test_acc"] <= 1.0
