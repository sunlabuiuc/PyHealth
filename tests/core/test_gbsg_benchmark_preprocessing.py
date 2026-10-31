"""Authors: Aritra Saha and Ugochukwu Njoku.

Exercise GBSG preprocessing through the actual task, benchmark and Trainer.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html
"""

import csv
import json

import numpy as np
import pytest
import torch

from examples.survival_benchmark.run_survival_mdn import GBSGBenchmark, main


def write_inputs(root, heldout_shift=0):
    """Write six artificial patients and two locally released-style splits."""
    root.mkdir()
    values = np.array([1, 3, 101, 103, 201, 203], dtype=float)
    values[2:] += heldout_shift
    features = np.column_stack([values * (j + 1) for j in range(6)] + [[5] * 6])
    times = np.arange(1, 7, dtype=float)
    events = np.array([0, 1, 1, 1, 1, 1], dtype=float)
    fields = ["source_id", "source_group", "source_row"]
    fields += [f"x{j}" for j in range(7)] + ["time", "event"]
    with (root / "gbsg.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(6):
            writer.writerow(
                {
                    "source_id": f"train_{i}",
                    "source_group": "train",
                    "source_row": i,
                    **{f"x{j}": features[i, j] for j in range(7)},
                    "time": times[i],
                    "event": events[i],
                }
            )
    released = root / "released"
    released.mkdir()
    for split, groups in [(1, [[0, 1], [2, 3], [4, 5]]), (2, [[2, 3], [0, 1], [4, 5]])]:
        for name, indices in zip(("train", "valid", "test"), groups):
            np.savez(
                released / f"{name}_{split}.npz",
                features[indices],
                np.column_stack([times[indices] / 6 + 0.001, events[indices]]),
            )
    return released, features


def by_id(samples):
    return {sample["patient_id"]: sample for sample in samples}


def test_gbsg_standardization_uses_training_rows_and_preserves_labels(tmp_path):
    root = tmp_path / "original"
    released, expected = write_inputs(root)
    raw = GBSGBenchmark(root, released, tmp_path / "raw-cache")
    scaled = GBSGBenchmark(root, released, tmp_path / "scaled-cache", standardize=True)
    ids, raw_samples, raw_meta = raw.split(1)
    scaled_ids, scaled_samples, metadata = scaled.split(1)
    assert ids == scaled_ids
    assert raw_meta["features"] == "raw"
    assert metadata["feature_mean"] == [2, 4, 6, 8, 10, 12, 5]
    assert metadata["feature_scale"] == [1, 2, 3, 4, 5, 6, 1]
    raw_rows, scaled_rows = by_id(raw_samples), by_id(scaled_samples)
    for i in range(6):
        patient = f"train_{i}"
        np.testing.assert_array_equal(raw_rows[patient]["features"], expected[i])
        for label in ("duration", "event"):
            torch.testing.assert_close(
                raw_rows[patient][label], scaled_rows[patient][label]
            )
    np.testing.assert_array_equal(scaled_rows["train_0"]["features"], [-1] * 6 + [0])
    np.testing.assert_array_equal(scaled_rows["train_1"]["features"], [1] * 6 + [0])
    np.testing.assert_array_equal(scaled_rows["train_2"]["features"], [99] * 6 + [0])
    changed = tmp_path / "changed"
    changed_release, _ = write_inputs(changed, heldout_shift=1000)
    altered = GBSGBenchmark(changed, changed_release, tmp_path / "changed-cache", True)
    _, changed_samples, changed_meta = altered.split(1)
    assert metadata == changed_meta
    changed_rows = by_id(changed_samples)
    for patient in ids["train"]:
        torch.testing.assert_close(
            scaled_rows[patient]["features"], changed_rows[patient]["features"]
        )
    _, _, second = scaled.split(2)
    assert second["feature_mean"] == [102, 204, 306, 408, 510, 612, 5]
    assert scaled.split(1)[2] == metadata


def test_gbsg_cli_versions_preprocessing_and_reuses_saved_fits(tmp_path, capsys):
    root = tmp_path / "inputs"
    released, _ = write_inputs(root)
    out = tmp_path / "runs"
    args = [
        "--dataset",
        "gbsg",
        "--root",
        str(root),
        "--released-dir",
        str(released),
        "--out",
        str(out),
        "--epochs",
        "1",
        "--device",
        "cpu",
    ]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        main(args)
        main([*args, "--standardize-gbsg"])
        paths = list(out.glob("*/split_1/seed_1/run_*/record.json"))
        assert len(paths) == 2
        original = {path: path.read_bytes() for path in paths}
        records = [json.loads(value) for value in original.values()]
        assert {r["request"]["gbsg_standardize"] for r in records} == {False, True}
        assert records[0]["model"] == records[1]["model"]
        assert records[0]["train"] == records[1]["train"]
        assert records[0]["request"]["inputs"] == records[1]["request"]["inputs"]
        for record, path in zip(records, paths):
            assert np.isfinite(record["val_nll"])
            assert record["counts"] == {"train": 2, "valid": 2, "test": 2}
            assert "trainer/best.ckpt" in record["artifacts"]
            with np.load(path.parent / "predictions.npz") as saved:
                assert saved["survival_at_observed_time"].shape == (2,)
        main(args)
        main([*args, "--standardize-gbsg"])
        assert all(path.read_bytes() == value for path, value in original.items())
    finally:
        torch.set_num_threads(previous_threads)
    capsys.readouterr()


@pytest.mark.parametrize("dataset", ["synthetic", "support2"])
def test_gbsg_scaling_option_rejects_other_datasets(dataset, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--dataset", dataset, "--standardize-gbsg", "--out", str(tmp_path)])
    assert error.value.code == 2
    assert not (tmp_path / "benchmark.json").exists()
    assert "requires --dataset gbsg" in capsys.readouterr().err
