# Author: Jathurshan Pradeepkumar
# NetID: jp65
# Paper Title: Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation; Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
# Paper Link: https://pubmed.ncbi.nlm.nih.gov/36878708/ ; https://arxiv.org/pdf/2502.16060
# Description: Unit tests for IIICDataset and EEGSeizureIIIC task

from dataclasses import dataclass
from pathlib import Path
import pickle
import tempfile
from typing import List, Optional
import unittest

import numpy as np
import pandas as pd

from pyhealth.datasets import IIICDataset
from pyhealth.tasks import EEGSeizureIIIC, IIICSeizureClassification


@dataclass
class _DummyEvent:
    record_id: str
    signal_file: str
    label: Optional[int] = None
    split: str = "train"


class _DummyPatient:
    def __init__(self, patient_id: str, events: List[_DummyEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_DummyEvent]:
        if event_type in (None, "signal"):
            return self._events
        return []


class TestIIICDataset(unittest.TestCase):
    """Tests for IIICDataset metadata preparation and loading."""

    def _create_raw_s3_npy_layout(self, root: Path) -> None:
        """Create synthetic BDSP S3 sparcnet_data .npy files."""
        rng = np.random.RandomState(42)
        # 8 segments across 5 patients + 1 duplicate key + 1 low-vote segment + 1 zero-std segment
        keys = np.array(
            [
                "abn10001_20140101_100000_001",
                "abn10001_20140101_100000_002",
                "abn10002_20140102_110000_001",
                "abn10003_20140103_120000_001",
                "abn10004_20140104_130000_001",
                "abn10005_20140105_140000_001",
                "abn10001_20140101_100000_001",  # duplicate key -> should be deduplicated
                "abn10006_20140106_150000_001",  # low vote sum <= 3 -> filtered out
                "abn10007_20140107_160000_001",  # flat zero signal -> filtered out
            ],
            dtype="<U30",
        )
        X = rng.randn(len(keys), 16, 2000).astype(np.float16)
        # Make the last segment completely flat (std == 0)
        X[-1] = 0.0

        Y = np.array(
            [
                [10, 0, 0, 0, 0, 0],  # class 0 (Other)
                [0, 12, 1, 0, 0, 0],  # class 1 (SZ)
                [0, 0, 8, 2, 0, 0],   # class 2 (LPD)
                [0, 0, 0, 9, 0, 0],   # class 3 (GPD)
                [0, 0, 0, 0, 7, 1],   # class 4 (LRDA)
                [0, 0, 0, 0, 0, 11],  # class 5 (GRDA)
                [10, 0, 0, 0, 0, 0],  # duplicate key
                [1, 1, 1, 0, 0, 0],   # sum == 3 (not > 3, filtered out)
                [0, 15, 0, 0, 0, 0],  # flat signal (filtered out)
            ],
            dtype=np.uint8,
        )

        np.save(root / "all_train_X.npy", X)
        np.save(root / "all_train_Y.npy", Y)
        np.save(root / "all_train_key.npy", keys)

    def _create_preextracted_pkl_layout(self, root: Path) -> None:
        """Create synthetic pre-extracted /srv/local/data/jp65/IIIC/ layout."""
        rng = np.random.RandomState(7)
        splits_info = {
            "train": [
                ("abn20001_20150101_000000_10", "abn20001", 0, [8, 0, 0, 0, 0, 0]),
                ("abn20001_20150101_000000_20", "abn20001", 1, [0, 9, 0, 0, 0, 0]),
            ],
            "val": [
                ("abn20002_20150102_000000_10", "abn20002", 2, [0, 0, 7, 0, 0, 0]),
            ],
            "test": [
                ("abn20003_20150103_000000_10", "abn20003", 5, [0, 0, 0, 0, 0, 6]),
            ],
        }

        for split, items in splits_info.items():
            split_dir = root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for key, pid, label, votes in items:
                sig = rng.randn(16, 2000).astype(np.float16)
                with open(split_dir / f"{key}.pkl", "wb") as f:
                    pickle.dump({"signal": sig, "label": label}, f)
                rows.append(
                    {
                        "0": votes[0],
                        "1": votes[1],
                        "2": votes[2],
                        "3": votes[3],
                        "4": votes[4],
                        "5": votes[5],
                        "key": key,
                        "majority": label,
                        "pid": pid,
                    }
                )
            pd.DataFrame(rows).to_csv(root / f"{split}_df.csv", index=False)

    def test_prepare_metadata_from_raw_s3_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_raw_s3_npy_layout(root)

            ds = IIICDataset(root=str(root), cache_dir=str(root / "cache"))
            csv_path = root / "iiic-processed.csv"
            self.assertTrue(csv_path.exists())

            df = pd.read_csv(csv_path)
            # 6 valid unique segments across 5 unique patients (abn10001..abn10005)
            self.assertEqual(len(df), 6)
            self.assertEqual(set(df["patient_id"]), {"abn10001", "abn10002", "abn10003", "abn10004", "abn10005"})

            # Check patient-level split disjointness (no patient in multiple splits)
            for pid, group in df.groupby("patient_id"):
                self.assertEqual(len(group["split"].unique()), 1)

            # Verify train_df.csv, val_df.csv, test_df.csv and .pkl files were created
            self.assertTrue((root / "train_df.csv").exists())
            self.assertTrue((root / "val_df.csv").exists())
            self.assertTrue((root / "test_df.csv").exists())
            for sig_file in df["signal_file"]:
                self.assertTrue(Path(sig_file).is_file())

            # Verify default_task property
            self.assertIsInstance(ds.default_task, EEGSeizureIIIC)

    def test_prepare_metadata_from_preextracted_pkl_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_preextracted_pkl_layout(root)

            ds = IIICDataset(root=str(root), cache_dir=str(root / "cache"))
            csv_path = root / "iiic-processed.csv"
            self.assertTrue(csv_path.exists())

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 4)
            self.assertEqual(set(df["split"]), {"train", "val", "test"})

            sample_dataset = ds.set_task(EEGSeizureIIIC())
            self.assertEqual(len(sample_dataset), 4)
            s0 = sample_dataset[0]
            self.assertEqual(s0["signal"].shape, (16, 2000))
            self.assertEqual(s0["stft"].shape, (16, 100, 19))
            self.assertIn(int(s0["label"]), {0, 1, 2, 5})

    def test_invalid_arguments_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                IIICDataset(root=tmp, subset="invalid_subset")
            with self.assertRaises(ValueError):
                IIICDataset(root=tmp, split_ratios=(0.5, 0.2))


class TestEEGSeizureIIICTask(unittest.TestCase):
    """Tests for EEGSeizureIIIC task signal processing and STFT."""

    def test_task_call_200hz_and_256hz_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.RandomState(0)
            sig = (rng.randn(16, 2000) * 50.0).astype(np.float16)
            pkl_path = root / "abn10001_20140101_100000_001.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump({"signal": sig, "label": 1}, f)

            patient = _DummyPatient(
                patient_id="abn10001",
                events=[
                    _DummyEvent(
                        record_id="abn10001_20140101_100000_001",
                        signal_file=str(pkl_path),
                        label=1,
                        split="train",
                    )
                ],
            )

            # 1) Default 200 Hz with STFT
            task_200 = EEGSeizureIIIC(resample_rate=200, compute_stft=True)
            samples_200 = task_200(patient)
            self.assertEqual(len(samples_200), 1)
            self.assertEqual(samples_200[0]["signal"].shape, (16, 2000))
            self.assertEqual(samples_200[0]["stft"].shape, (16, 100, 19))
            self.assertEqual(samples_200[0]["label"], 1)
            self.assertEqual(samples_200[0]["split"], "train")

            # 2) Resampled 256 Hz (matching Biosignal_Token 256Hz mode)
            task_256 = EEGSeizureIIIC(resample_rate=256, compute_stft=True)
            samples_256 = task_256(patient)
            self.assertEqual(len(samples_256), 1)
            self.assertEqual(samples_256[0]["signal"].shape, (16, 2560))
            self.assertEqual(samples_256[0]["stft"].shape, (16, 128, 19))

            # 3) compute_stft=False (for SparcNet)
            task_no_stft = EEGSeizureIIIC(
                resample_rate=200,
                normalization="div_by_100",
                compute_stft=False,
            )
            self.assertEqual(task_no_stft.input_schema, {"signal": "tensor"})
            samples_no_stft = task_no_stft(patient)
            self.assertEqual(len(samples_no_stft), 1)
            self.assertNotIn("stft", samples_no_stft[0])
            self.assertEqual(samples_no_stft[0]["signal"].shape, (16, 2000))

    def test_alias_and_invalid_normalization(self):
        self.assertIs(IIICSeizureClassification, EEGSeizureIIIC)
        with self.assertRaises(ValueError):
            EEGSeizureIIIC(normalization="bad_norm")


if __name__ == "__main__":
    unittest.main()
