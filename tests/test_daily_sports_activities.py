from pyhealth.datasets import DailySportsActivities

def test_dsa_load(tmp_path):
    ds = DailySportsActivities(root=str(tmp_path), split="train")
    assert hasattr(ds, "data") and hasattr(ds, "labels")
    assert ds.data.ndim == 3
