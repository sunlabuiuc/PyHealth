"""Small PTBXLDataset-backed fixture checks for the H1 ablation example.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., MLHC 2025.
ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning.
https://proceedings.mlr.press/v298/sethi25a.html
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import butter, filtfilt

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/ptbxl_multilabel_classification_protoecgnet_h1.py"
)
spec = importlib.util.spec_from_file_location("protoecgnet_h1_example", EXAMPLE_PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_macro_metrics_report_eligibility_and_keep_zero_targets():
    y = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    p = np.array([[0.9, 0.1, 0.1], [0.2, 0.1, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)
    metrics = example._classification(y, p)
    assert metrics["eligible_labels"] == 2
    assert metrics["total_labels"] == 3
    assert metrics["positive_support"] == [1, 0, 1]
    assert metrics["negative_support"] == [2, 3, 2]
    assert metrics["macro_auroc"] == 1.0
    assert metrics["macro_ap"] == 1.0


def test_macro_metrics_with_no_eligible_label_are_explicit():
    metrics = example._classification(np.zeros((2, 3)), np.ones((2, 3)) * 0.5)
    assert metrics["eligible_labels"] == 0
    assert metrics["macro_auroc"] is None and metrics["macro_ap"] is None


def test_real_dataset_adapter_preserves_zero_targets_and_skips_heldout_waveforms(
    tmp_path,
):
    wfdb = pytest.importorskip("wfdb")
    root = tmp_path / "ptbxl"
    root.mkdir()
    rng = np.random.default_rng(59852)
    rows = []
    for eid, fold, codes in [
        (1, 1, {"NDT": 100.0}),
        (2, 2, {"NST_": 0.0}),
        (3, 8, {}),
        (4, 9, {"NDT": 100.0}),
        (5, 9, {}),
        (6, 10, {"NDT": 100.0}),
    ]:
        rel = f"records100/{eid:05d}_lr"
        rows.append(
            {
                "ecg_id": eid,
                "patient_id": float(100 + eid),
                "age": 45,
                "sex": 1,
                "site": 0,
                "device": "synthetic",
                "recording_date": "2000-01-01 00:00:00",
                "scp_codes": str(codes),
                "strat_fold": fold,
                "filename_lr": rel,
                "filename_hr": f"records500/{eid:05d}_hr",
            }
        )
        if fold == 10:
            continue  # A real attempted held-out read must fail.
        folder = root / "records100"
        folder.mkdir(exist_ok=True)
        wfdb.wrsamp(
            record_name=f"{eid:05d}_lr",
            fs=100,
            units=["mV"] * 12,
            sig_name=example.LEADS,
            p_signal=rng.normal(size=(1000, 12)) * 0.1,
            fmt=["16"] * 12,
            write_dir=str(folder),
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"description": ["synthetic"], "diagnostic": [1]}, index=["NDT"]
    ).to_csv(root / "scp_statements.csv")
    output = tmp_path / "cache"
    output.mkdir()
    train, validation, y = example._ptbxl_splits(root, output, 0, 0)
    assert len(train) == 3 and len(validation) == 2
    assert [train[i]["ecg_id"] for i in range(3)] == [1, 2, 3]
    assert y.shape == (3, 52)
    assert y[0, 0] == 1 and y[1, 1] == 1  # Unknown likelihood is a positive key.
    assert y[2].sum() == 0 and train[2]["labels"].sum() == 0
    assert all(train[i]["strat_fold"] in range(1, 9) for i in range(3))
    assert all(validation[i]["strat_fold"] == 9 for i in range(2))
    raw, _ = wfdb.rdsamp(str(root / "records100/00001_lr"))
    b, a = butter(1, 0.5 / (100 / 2), btype="high")
    expected = torch.from_numpy(
        np.ascontiguousarray(filtfilt(b, a, raw.T, axis=-1), dtype=np.float32)
    )
    assert torch.equal(train[0]["signal"], expected)
    assert not (root / "records100/00006_lr.hea").exists()


def test_observed_loader_records_actual_matched_two_epoch_streams():
    train, _, _ = example._synthetic_splits(598)
    loaders = [
        example._ObservedPairedLoader(
            train, batch_size=4, stream_seed=example.TRAIN_NOISE_STREAM, seed=598
        )
        for _ in range(2)
    ]
    for loader in loaders:
        for _ in range(2):
            for batch in loader:
                assert batch["signal"].shape == batch["noisy_signal"].shape
    assert loaders[0].stream_hashes == loaders[1].stream_hashes
    assert loaders[0].consumed_batches == {"1": 2, "2": 2}
    assert loaders[0].stream_hashes["1"] != loaders[0].stream_hashes["2"]
