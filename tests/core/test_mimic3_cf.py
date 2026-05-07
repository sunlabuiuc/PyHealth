"""
Unit tests for ``pyhealth.datasets.MIMIC3CirculatoryFailureDataset``.
"""
from pathlib import Path
from pyhealth.datasets import MIMIC3CirculatoryFailureDataset


def test_mimic3_cf_dataset_initialization(monkeypatch):
    captured = {}

    def fake_base_init(
        self,
        root,
        tables,
        dataset_name=None,
        config_path=None,
        **kwargs,
    ):
        self.root = root
        self.tables = tables
        self.dataset_name = dataset_name
        self.config_path = config_path
        captured["tables"] = tables
        captured["dataset_name"] = dataset_name
        captured["config_path"] = config_path

    monkeypatch.setattr(
        "pyhealth.datasets.base_dataset.BaseDataset.__init__",
        fake_base_init,
    )

    dataset = MIMIC3CirculatoryFailureDataset(root="dummy-root")

    assert dataset.dataset_name == "mimic3_cf"
    assert "patients" in dataset.tables
    assert "admissions" in dataset.tables
    assert "icustays" in dataset.tables
    assert "chartevents" in dataset.tables
    assert Path(dataset.config_path).name == "mimic3_cf.yaml"