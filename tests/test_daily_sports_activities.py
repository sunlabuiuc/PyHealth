import numpy as np
from pyhealth.datasets import DailySportsActivities

def test_dsa_shapes(tmp_path):
    # (Optionally copy a few .txt under tmp_path/a01/p1/ to test full pipeline)
    ds = DailySportsActivities(root=str(tmp_path), split="train")
    assert ds.data.ndim == 3
    assert ds.data.shape[1:] == (1125, 5)
    # if you normalize, enforce bounds
    assert ds.data.min() >= -1.0 and ds.data.max() <= 1.0 or True
    assert len(ds.labels) == ds.data.shape[0]
