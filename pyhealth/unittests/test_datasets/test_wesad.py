import os
import tempfile
import pickle
import numpy as np
from pyhealth.datasets import WESADDataset

def create_fake_wesad_pkl(path):
    signal = {
        "chest": {
            "ACC": np.random.randn(1000, 3),
            "EDA": np.random.randn(125),     # 1000 / 8
            "Temp": np.random.randn(125),
            "ECG": np.random.randn(21875),   # 1000 * 700 / 32
            "Resp": np.random.randn(21875),
        }
    }
    sample = {
        "subject": "S99",
        "signal": signal,
        "label": np.random.randn(21875),
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "S99.pkl"), "wb") as f:
        pickle.dump(sample, f)


def test_wesad_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        subj_dir = os.path.join(tmpdir, "S99")
        os.makedirs(subj_dir, exist_ok=True)
        create_fake_wesad_pkl(subj_dir)

        dataset = WESADDataset(root=tmpdir, segment_length=240)
        assert len(dataset.global_event_df) > 0, "Dataset should not be empty"
        sample = dataset.global_event_df[0]
        assert "data" in sample and "subject" in sample
        assert sample["data"].shape == (240, 7), "Segment shape should be (240, 7)"
        print("Test passed.")


if __name__ == "__main__":
    test_wesad_dataset()
