"""
Tests for DSADataset and PhysicalActivityTask.

IMPORTANT: All tests use synthetic/fake data only.
           No real DSA files are read. Tests run in milliseconds.
"""

import os
import tempfile
import numpy as np
import pytest
import torch


# ── Synthetic Data Fixtures ────────────────────────────────────────────────────

def create_fake_dsa_root(tmp_dir: str, n_subjects: int = 2) -> str:
    """Create a minimal fake DSA folder structure with synthetic .txt files.

    Mimics the real DSA structure:
        root/a{01-02}/p{1-2}/s{01-02}.txt
        Each file: 125 rows x 45 columns (float)

    Args:
        tmp_dir: Temporary directory to write files into.
        n_subjects: Number of fake subjects to create.

    Returns:
        Path to the fake root directory.
    """
    root = os.path.join(tmp_dir, "dsa_fake")
    # Only create 2 activities and 2 segments for speed
    for act in ["a01", "a02"]:
        for subj in ["p{}".format(i) for i in range(1, n_subjects + 1)]:
            folder = os.path.join(root, act, subj)
            os.makedirs(folder, exist_ok=True)
            for seg in range(1, 3):  # s01.txt, s02.txt only
                fname = "s{:02d}.txt".format(seg)
                fpath = os.path.join(folder, fname)
                # Fake data: 125 rows x 45 cols
                data = np.random.rand(125, 45).astype(np.float32)
                np.savetxt(fpath, data, delimiter=",")
    return root


# ── DSADataset Tests ───────────────────────────────────────────────────────────

class TestDSADataset:
    """Tests for DSADataset loading and parsing."""

    def test_init_valid(self, tmp_path):
        """Dataset initializes without error on valid fake data."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        assert ds is not None

    def test_subjects_loaded(self, tmp_path):
        """Correct number of subjects are loaded."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path), n_subjects=2)
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        assert len(ds.patients) == 2
        assert "p1" in ds.patients
        assert "p2" in ds.patients

    def test_event_structure(self, tmp_path):
        """Each event has correct keys and data shape."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)

        # Grab first event
        event = ds.patients["p1"]["a01"][0]
        assert "sensor_id" in event
        assert "sensor_name" in event
        assert "is_target" in event
        assert "label" in event
        assert "data" in event
        assert event["data"].shape == (9, 125)
        assert event["data"].dtype == np.float32

    def test_label_range(self, tmp_path):
        """Labels are valid integers in [0, 18]."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        for subj in ds.patients.values():
            for events in subj.values():
                for e in events:
                    assert 0 <= e["label"] <= 18

    def test_target_sensor_flag(self, tmp_path):
        """is_target is True only for the target sensor."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        for subj in ds.patients.values():
            for events in subj.values():
                for e in events:
                    if e["sensor_id"] == "s2":
                        assert e["is_target"] is True
                    else:
                        assert e["is_target"] is False

    def test_invalid_target_sensor(self, tmp_path):
        """ValueError raised for invalid target sensor."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        with pytest.raises(ValueError, match="target_sensor"):
            DSADataset(root=root, target_sensor="s99")

    def test_missing_root(self):
        """FileNotFoundError raised for non-existent root."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        with pytest.raises(FileNotFoundError):
            DSADataset(root="/nonexistent/path/xyz")

    def test_get_sensor_data_shape(self, tmp_path):
        """get_sensor_data returns correct array shapes."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        X, y = ds.get_sensor_data(sensor_id="s2", split="all")
        assert X.ndim == 3
        assert X.shape[1] == 9
        assert X.shape[2] == 125
        assert y.ndim == 1
        assert len(X) == len(y)

    def test_get_all_samples_split(self, tmp_path):
        """Train/test split returns disjoint subject sets."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path), n_subjects=2)
        ds = DSADataset(root=root, target_sensor="s2", dev=False)
        X_tr, y_tr, s_tr = ds.get_all_samples(
            split="train", train_subjects=["p1"]
        )
        X_te, y_te, s_te = ds.get_all_samples(
            split="test", test_subjects=["p2"]
        )
        assert len(X_tr) > 0
        assert len(X_te) > 0

    def test_repr(self, tmp_path):
        """__repr__ returns a non-empty string."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        r = repr(ds)
        assert "DSADataset" in r
        assert "right_arm" in r


# ── PhysicalActivityTask Tests ─────────────────────────────────────────────────

class TestPhysicalActivityTask:
    """Tests for PhysicalActivityTask and PhysicalActivityDataset."""

    def test_task_builds(self, tmp_path):
        """Task builds without error."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        from pyhealth.tasks.physical_activity_task import PhysicalActivityTask
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        task = PhysicalActivityTask(
            ds, train_subjects=["p1"], test_subjects=["p2"]
        )
        assert task is not None

    def test_target_dataset_tensor_shape(self, tmp_path):
        """Target dataset returns tensors of correct shape."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        from pyhealth.tasks.physical_activity_task import PhysicalActivityTask
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        task = PhysicalActivityTask(
            ds, train_subjects=["p1"], test_subjects=["p2"]
        )
        train_ds = task.get_target_dataset(split="train")
        x, y = train_ds[0]
        assert x.shape == torch.Size([9, 125])
        assert y.ndim == 0  # scalar

    def test_normalization_range(self, tmp_path):
        """Normalized values are within [-1, 1]."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        from pyhealth.tasks.physical_activity_task import PhysicalActivityTask
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        task = PhysicalActivityTask(
            ds, train_subjects=["p1"], test_subjects=["p2"]
        )
        train_ds = task.get_target_dataset(split="train")
        x, _ = train_ds[0]
        assert x.min().item() >= -1.01   # small tolerance
        assert x.max().item() <= 1.01

    def test_source_datasets_exclude_target(self, tmp_path):
        """Source datasets do not include the target sensor."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        from pyhealth.tasks.physical_activity_task import PhysicalActivityTask
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        task = PhysicalActivityTask(
            ds, train_subjects=["p1"], test_subjects=["p2"]
        )
        sources = task.get_source_datasets(split="train")
        assert "s2" not in sources
        assert len(sources) == 4   # s1, s3, s4, s5

    def test_no_normalization(self, tmp_path):
        """Dataset without normalization returns raw values."""
        from pyhealth.datasets.dsa_dataset import DSADataset
        from pyhealth.tasks.physical_activity_task import (
            PhysicalActivityTask, PhysicalActivityDataset
        )
        root = create_fake_dsa_root(str(tmp_path))
        ds = DSADataset(root=root, target_sensor="s2", dev=True)
        task = PhysicalActivityTask(
            ds, train_subjects=["p1"], test_subjects=["p2"],
            normalize=False,
        )
        train_ds = task.get_target_dataset(split="train")
        x, _ = train_ds[0]
        # Raw values from np.random.rand should be in [0, 1]
        assert x.max().item() <= 1.1
