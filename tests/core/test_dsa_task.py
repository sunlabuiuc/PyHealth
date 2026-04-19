"""Tests for the DSA standalone activity-classification task."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyhealth.datasets.dsa import DSADataset
from pyhealth.tasks.dsa_activity_classification import DSAActivityClassification


def _write_segment_from_array(path: Path, values: np.ndarray) -> None:
    """Write a synthetic DSA segment file from a 2D array."""
    lines = []
    for row in values:
        lines.append(",".join(f"{value:.6f}" for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_patterned_segment(
    path: Path,
    n_rows: int = 125,
    n_cols: int = 45,
    row_scale: float = 1.0,
    col_scale: float = 0.1,
) -> np.ndarray:
    """Write a segment with deterministic, non-constant channel values."""
    row_offsets = np.arange(n_rows, dtype=np.float64).reshape(-1, 1) * row_scale
    col_offsets = np.arange(n_cols, dtype=np.float64).reshape(1, -1) * col_scale
    values = row_offsets + col_offsets
    _write_segment_from_array(path, values)
    return values


def _write_constant_segment(
    path: Path,
    value: float = 7.0,
    n_rows: int = 125,
    n_cols: int = 45,
) -> np.ndarray:
    """Write a segment where every value is identical."""
    values = np.full((n_rows, n_cols), value, dtype=np.float64)
    _write_segment_from_array(path, values)
    return values


def _make_dsa_tree(
    root: Path,
    activities: tuple[str, ...] = ("a01",),
    subjects: tuple[str, ...] = ("p1",),
    segments: tuple[str, ...] = ("s01.txt",),
    constant: bool = False,
) -> np.ndarray:
    """Create a minimal DSA directory tree and return the last segment array."""
    last_values = np.empty((125, 45), dtype=np.float64)
    for activity in activities:
        for subject in subjects:
            for segment in segments:
                seg_dir = root / activity / subject
                seg_dir.mkdir(parents=True, exist_ok=True)
                seg_path = seg_dir / segment
                if constant:
                    last_values = _write_constant_segment(seg_path)
                else:
                    last_values = _write_patterned_segment(seg_path)
    return last_values


class TestDSAActivityClassification(unittest.TestCase):
    """Task-level coverage for DSA activity classification."""

    def test_invalid_selected_unit_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported DSA unit"):
            DSAActivityClassification(dataset_root="/tmp/dsa", selected_units=("bad",))

    def test_invalid_normalization_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported normalization"):
            DSAActivityClassification(
                dataset_root="/tmp/dsa",
                normalization="zscore",
            )

    def test_task_generates_expected_samples_for_all_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_dsa_tree(
                root,
                activities=("a01", "a02"),
                subjects=("p1",),
                segments=("s01.txt", "s02.txt"),
            )
            dataset = DSADataset(root=tmpdir)
            patient = next(dataset.iter_patients())
            task = DSAActivityClassification(dataset_root=tmpdir)

            samples = task(patient)

            self.assertEqual(len(samples), 4)
            sample = samples[0]
            self.assertEqual(sample["patient_id"], "p1")
            self.assertEqual(sample["subject_id"], "p1")
            self.assertEqual(sample["signal"].shape, (125, 45))
            self.assertEqual(sample["num_channels"], 45)
            self.assertEqual(sample["unit_combo"], "T+RA+LA+RL+LL")
            self.assertIn(sample["activity_name"], {"sitting", "standing"})
            self.assertIn(sample["activity_code"], {"A01", "A02"})
            self.assertIn(sample["label"], {0, 1})

    def test_selected_unit_extracts_expected_columns_without_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_values = _make_dsa_tree(root)
            dataset = DSADataset(root=tmpdir)
            patient = next(dataset.iter_patients())
            task = DSAActivityClassification(
                dataset_root=tmpdir,
                selected_units=("RA",),
                normalization="none",
            )

            samples = task(patient)

            self.assertEqual(len(samples), 1)
            expected = raw_values[:, 9:18]
            np.testing.assert_allclose(samples[0]["signal"], expected)
            self.assertEqual(samples[0]["unit_combo"], "RA")
            self.assertEqual(samples[0]["num_channels"], 9)

    def test_minmax_normalization_scales_each_channel_to_minus1_plus1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_dsa_tree(root)
            dataset = DSADataset(root=tmpdir)
            patient = next(dataset.iter_patients())
            task = DSAActivityClassification(
                dataset_root=tmpdir,
                selected_units=("T",),
                normalization="minmax",
            )

            samples = task(patient)
            signal = samples[0]["signal"]

            np.testing.assert_allclose(signal.min(axis=0), -1.0)
            np.testing.assert_allclose(signal.max(axis=0), 1.0)

    def test_constant_channels_become_zeros_after_minmax_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_dsa_tree(root, constant=True)
            dataset = DSADataset(root=tmpdir)
            patient = next(dataset.iter_patients())
            task = DSAActivityClassification(
                dataset_root=tmpdir,
                selected_units=("T",),
                normalization="minmax",
            )

            samples = task(patient)

            np.testing.assert_allclose(samples[0]["signal"], 0.0)

    def test_set_task_builds_sample_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_dsa_tree(
                root,
                activities=("a01", "a02"),
                subjects=("p1", "p2"),
                segments=("s01.txt",),
            )
            dataset = DSADataset(root=tmpdir)
            task = DSAActivityClassification(
                dataset_root=tmpdir,
                selected_units=("T", "RA"),
            )

            sample_dataset = dataset.set_task(task, num_workers=1)

            self.assertEqual(len(sample_dataset), 4)
            self.assertIn("signal", sample_dataset.input_processors)
            self.assertIn("label", sample_dataset.output_processors)

            sample = sample_dataset[0]
            self.assertEqual(tuple(sample["signal"].shape), (125, 18))
            self.assertIn(int(sample["label"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
