# Author: Jathurshan Pradeepkumar
# NetID: jp65
# Paper Title: Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
# Paper Link: https://arxiv.org/pdf/2502.16060
# Description: Unit tests for IIICDataset and EEGSeizureIIIC task

import os
import pickle
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.datasets import IIICDataset
from pyhealth.tasks import EEGSeizureIIIC, IIICSeizureClassification


@dataclass
class _DummyEvent:
    record_id: str
    signal_file: str
    label: int | None = None
    split: str = "train"


class _DummyPatient:
    def __init__(self, patient_id: str, events: list[_DummyEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> list[_DummyEvent]:
        if event_type in (None, "signal"):
            return self._events
        return []


class TestIIICDataset(unittest.TestCase):
    """Tests for IIICDataset metadata preparation and loading."""

    def _create_raw_s3_npy_layout(self, root: Path) -> None:
        """Create synthetic BDSP S3 sparcnet_data .npy files."""
        rng = np.random.RandomState(42)
        # 8 segments across 5 patients + 1 duplicate + 1 low-vote + 1 zero-std
        keys = np.array(
            [
                "abn10001_20140101_100000_001",
                "abn10001_20140101_100000_002",
                "abn10002_20140102_110000_001",
                "abn10003_20140103_120000_001",
                "abn10004_20140104_130000_001",
                "abn10005_20140105_140000_001",
                "abn10001_20140101_100000_001",  # duplicate -> deduplicated
                "abn10006_20140106_150000_001",  # vote sum <= 3 -> filtered
                "abn10007_20140107_160000_001",  # flat signal -> filtered
            ],
            dtype="<U30",
        )
        X = rng.randn(len(keys), 16, 2000).astype(np.float16)
        # Give first (idx 0) and last (idx 6) duplicate rows distinct signals
        X[0] = (np.ones((16, 2000)) + rng.randn(16, 2000) * 0.1).astype(np.float16)
        X[6] = (np.full((16, 2000), 5.0) + rng.randn(16, 2000) * 0.1).astype(
            np.float16
        )
        # Make the last segment completely flat (std == 0)
        X[-1] = 0.0

        Y = np.array(
            [
                [0, 4, 0, 0, 0, 0],   # first dup (idx 0, class 1, superseded)
                [0, 12, 1, 0, 0, 0],  # class 1 (SZ)
                [0, 0, 8, 2, 0, 0],   # class 2 (LPD)
                [0, 0, 0, 9, 0, 0],   # class 3 (GPD)
                [0, 0, 0, 0, 7, 1],   # class 4 (LRDA)
                [0, 0, 0, 0, 0, 11],  # class 5 (GRDA)
                [10, 0, 0, 0, 0, 0],  # last dup (idx 6, class 0, kept)
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
            self.assertEqual(
                set(df["patient_id"]),
                {"abn10001", "abn10002", "abn10003", "abn10004", "abn10005"},
            )

            # Check patient-level split disjointness (no patient in multiple splits)
            for _pid, group in df.groupby("patient_id"):
                self.assertEqual(len(group["split"].unique()), 1)

            # Verify train_df.csv, val_df.csv, test_df.csv and .pkl files were created
            self.assertTrue((root / "train_df.csv").exists())
            self.assertTrue((root / "val_df.csv").exists())
            self.assertTrue((root / "test_df.csv").exists())
            for sig_file in df["signal_file"]:
                self.assertTrue(Path(sig_file).is_file())

            # Verify duplicate key kept row 6's votes, label, and signal consistently
            dup_row = df[df["record_id"] == "abn10001_20140101_100000_001"].iloc[0]
            self.assertEqual(int(dup_row["label"]), 0)
            self.assertEqual(dup_row["votes"], "[10, 0, 0, 0, 0, 0]")
            with open(dup_row["signal_file"], "rb") as f:
                dup_pkl = pickle.load(f)
            raw_x = np.load(root / "all_train_X.npy")
            np.testing.assert_array_equal(dup_pkl["signal"], raw_x[6])
            self.assertEqual(int(dup_pkl["label"]), 0)

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

    def test_save_pkl_false_and_10_train_subset_and_param_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_raw_s3_npy_layout(root)
            cache_dir = root / "cache"

            # 1) save_pkl=False references slices via path.npy::idx
            ds_no_pkl = IIICDataset(
                root=str(root),
                cache_dir=str(cache_dir),
                save_pkl=False,
                vote_threshold=3,
            )
            df_no_pkl = pd.read_csv(root / "iiic-processed.csv")
            self.assertEqual(len(df_no_pkl), 6)
            for sig_ref in df_no_pkl["signal_file"]:
                self.assertIn("::", sig_ref)
            self.assertFalse((root / "train").exists())

            # Verify EEGSeizureIIIC loads path.npy::idx slices end-to-end
            samples_no_pkl = ds_no_pkl.set_task(
                EEGSeizureIIIC(compute_stft=False)
            )
            self.assertEqual(len(samples_no_pkl), 6)
            self.assertEqual(samples_no_pkl[0]["signal"].shape, (16, 2000))

            # 2) subset="10_train" rebuilds metadata and keeps >=10 votes (4 of 6)
            ds_10 = IIICDataset(
                root=str(root),
                cache_dir=str(cache_dir),
                subset="10_train",
                save_pkl=True,
                vote_threshold=3,
            )
            self.assertIsNotNone(ds_10)
            df_10 = pd.read_csv(root / "iiic-processed.csv")
            self.assertEqual(len(df_10), 4)

            # 3) Changing vote_threshold rebuilds metadata (no stale CSV reuse)
            ds_strict = IIICDataset(
                root=str(root),
                cache_dir=str(cache_dir),
                subset="all",
                save_pkl=True,
                vote_threshold=10,
            )
            self.assertIsNotNone(ds_strict)
            df_strict = pd.read_csv(root / "iiic-processed.csv")
            self.assertEqual(len(df_strict), 5)

    def test_pkl_dirs_fallback_and_10_train_subdir_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.RandomState(11)
            # Case 3 fallback: train/ val/ test/ with .pkl files and no *_df.csv or .npy
            for split, key, label in [
                ("train", "abn30001_20150101_000000_01", 1),
                ("val", "abn30002_20150102_000000_01", 2),
                ("test", "abn30003_20150103_000000_01", 3),
            ]:
                sdir = root / split
                sdir.mkdir(parents=True, exist_ok=True)
                sig = rng.randn(16, 2000).astype(np.float16)
                with open(sdir / f"{key}.pkl", "wb") as f:
                    pickle.dump({"signal": sig, "label": label}, f)

            ds = IIICDataset(root=str(root), cache_dir=str(root / "cache"))
            self.assertIsNotNone(ds)
            df = pd.read_csv(root / "iiic-processed.csv")
            self.assertEqual(len(df), 3)
            self.assertEqual(set(df["label"]), {1, 2, 3})

        # Also test 10_train_{X,Y,key}.npy in data_1109/ subdirectory with 1D Y
        with tempfile.TemporaryDirectory() as tmp2:
            root2 = Path(tmp2)
            sub = root2 / "data_1109"
            sub.mkdir(parents=True, exist_ok=True)
            rng = np.random.RandomState(21)
            keys = np.array(
                [
                    "abn40001_20160101_001",
                    "abn40002_20160102_001",
                    "abn40003_20160103_001",
                ],
                dtype="<U30",
            )
            X = rng.randn(3, 16, 2000).astype(np.float16)
            Y_1d = np.array([0, 1, 4], dtype=np.int64)
            np.save(sub / "10_train_X.npy", X)
            np.save(sub / "10_train_Y.npy", Y_1d)
            np.save(sub / "10_train_key.npy", keys)

            ds2 = IIICDataset(
                root=str(root2),
                cache_dir=str(root2 / "cache"),
                subset="10_train",
                save_pkl=False,
            )
            self.assertIsNotNone(ds2)
            df2 = pd.read_csv(root2 / "iiic-processed.csv")
            self.assertEqual(len(df2), 3)
            self.assertEqual(set(df2["label"]), {0, 1, 4})

        # Test cross-source duplicate key (appearing in both train and test .npy)
        with tempfile.TemporaryDirectory() as tmp3:
            root3 = Path(tmp3)
            rng = np.random.RandomState(31)
            shared_key = "abn50001_20160101_001"
            train_keys = np.array(
                [shared_key, "abn50002_20160102_001", "abn50003_20160103_001"],
                dtype="<U30",
            )
            train_x = rng.randn(3, 16, 2000).astype(np.float16)
            train_y = np.array(
                [
                    [10, 0, 0, 0, 0, 0],  # class 0 in train source
                    [0, 10, 0, 0, 0, 0],
                    [0, 0, 10, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
            test_keys = np.array([shared_key], dtype="<U30")
            test_x = (
                np.full((1, 16, 2000), 9.0) + rng.randn(1, 16, 2000) * 0.1
            ).astype(np.float16)
            test_y = np.array([[0, 0, 0, 0, 0, 12]], dtype=np.uint8)  # class 5

            np.save(root3 / "all_train_X.npy", train_x)
            np.save(root3 / "all_train_Y.npy", train_y)
            np.save(root3 / "all_train_key.npy", train_keys)
            np.save(root3 / "test_X.npy", test_x)
            np.save(root3 / "test_Y.npy", test_y)
            np.save(root3 / "test_key.npy", test_keys)

            ds3 = IIICDataset(root=str(root3), cache_dir=str(root3 / "cache"))
            self.assertIsNotNone(ds3)
            df3 = pd.read_csv(root3 / "iiic-processed.csv")
            # shared_key must appear only once across the combined dataframe
            self.assertEqual(len(df3), 3)
            shared_rows = df3[df3["record_id"] == shared_key]
            self.assertEqual(len(shared_rows), 1)
            self.assertEqual(int(shared_rows.iloc[0]["label"]), 5)
            with open(shared_rows.iloc[0]["signal_file"], "rb") as f:
                saved_pkl = pickle.load(f)
            np.testing.assert_array_equal(saved_pkl["signal"], test_x[0])
            self.assertEqual(int(saved_pkl["label"]), 5)

    def test_readonly_root_uses_cache_dir_and_keys_by_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ro_root = base / "ro_root"
            cache_base = base / "custom_cache"
            ro_root.mkdir()
            self._create_raw_s3_npy_layout(ro_root)

            os.chmod(ro_root, 0o555)
            try:
                ds1 = IIICDataset(
                    root=str(ro_root),
                    cache_dir=str(cache_base),
                    vote_threshold=3,
                    seed=5,
                )
                self.assertFalse((ro_root / "iiic-processed.csv").exists())
                df1 = pd.read_csv(Path(ds1.root) / "iiic-processed.csv")
                self.assertEqual(len(df1), 6)

                ds2 = IIICDataset(
                    root=str(ro_root),
                    cache_dir=str(cache_base),
                    vote_threshold=10,
                    seed=5,
                )
                self.assertNotEqual(ds1.root, ds2.root)
                df2 = pd.read_csv(Path(ds2.root) / "iiic-processed.csv")
                self.assertEqual(len(df2), 5)
            finally:
                os.chmod(ro_root, 0o755)

            with (
                patch.dict(os.environ, {"PYHEALTH_REQUIRE_CACHE_DIR": "1"}),
                self.assertRaises(ValueError),
            ):
                IIICDataset(root=str(ro_root))

    def test_relative_root_and_unlabeled_skip_behavior(self):
        # 1) Relative root resolves to absolute paths and works after chdir
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            data_dir = base / "rel_data"
            other_cwd = base / "other_cwd"
            data_dir.mkdir()
            other_cwd.mkdir()
            self._create_preextracted_pkl_layout(data_dir)

            orig_cwd = Path.cwd()
            try:
                os.chdir(base)
                ds = IIICDataset(
                    root="rel_data", cache_dir=str(base / "cache")
                )
                self.assertTrue(Path(ds.root).is_absolute())
                df = pd.read_csv(data_dir / "iiic-processed.csv")
                self.assertTrue(
                    all(Path(p).is_absolute() for p in df["signal_file"])
                )

                # Change working directory and run task against cached CSV
                os.chdir(other_cwd)
                samples = ds.set_task(EEGSeizureIIIC(compute_stft=False))
                self.assertEqual(len(samples), 4)
            finally:
                os.chdir(orig_cwd)

        # 2) _build_metadata_from_pkl_dirs skips pickles missing 'label' key
        with tempfile.TemporaryDirectory() as tmp2:
            root2 = Path(tmp2)
            tdir = root2 / "train"
            tdir.mkdir(parents=True)
            rng = np.random.RandomState(99)
            sig = rng.randn(16, 2000).astype(np.float16)

            with open(tdir / "abn60001_20170101_001.pkl", "wb") as f:
                pickle.dump({"signal": sig, "label": 3}, f)
            with open(tdir / "abn60002_20170102_001.pkl", "wb") as f:
                pickle.dump({"signal": sig}, f)  # no label key -> must skip

            with self.assertLogs("pyhealth.datasets.iiic", level="WARNING") as cm:
                ds2 = IIICDataset(
                    root=str(root2), cache_dir=str(root2 / "cache")
                )
            self.assertIsNotNone(ds2)
            df2 = pd.read_csv(root2 / "iiic-processed.csv")
            self.assertEqual(len(df2), 1)
            self.assertEqual(int(df2.iloc[0]["label"]), 3)
            self.assertTrue(
                any("missing 'label' key" in m for m in cm.output)
            )

        # 3) _build_metadata_from_split_csvs skips CSVs/rows without labels
        with tempfile.TemporaryDirectory() as tmp3:
            root3 = Path(tmp3)
            for sp in ("train", "val"):
                (root3 / sp).mkdir(parents=True)
            # train_df.csv has 'label' column for one row and NaN for the second
            pd.DataFrame(
                [
                    {"key": "abn70001_20180101_001", "label": 4},
                    {"key": "abn70002_20180102_001", "label": np.nan},
                ]
            ).to_csv(root3 / "train_df.csv", index=False)
            # val_df.csv has no majority, label, or vote columns at all
            pd.DataFrame([{"key": "abn70003_20180103_001"}]).to_csv(
                root3 / "val_df.csv", index=False
            )

            with self.assertLogs("pyhealth.datasets.iiic", level="WARNING") as cm3:
                ds3 = IIICDataset(
                    root=str(root3), cache_dir=str(root3 / "cache")
                )
            self.assertIsNotNone(ds3)
            df3 = pd.read_csv(root3 / "iiic-processed.csv")
            self.assertEqual(len(df3), 1)
            self.assertEqual(int(df3.iloc[0]["label"]), 4)
            self.assertTrue(
                any(
                    "has no 'majority', 'label', or vote columns" in m
                    for m in cm3.output
                )
            )


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

    def test_corrupted_pkl_is_logged_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.RandomState(1)
            sig = (rng.randn(16, 2000) * 50.0).astype(np.float16)

            valid_pkl = root / "abn10001_20140101_100000_001.pkl"
            with open(valid_pkl, "wb") as f:
                pickle.dump({"signal": sig, "label": 1}, f)

            corrupt_pkl = root / "abn10001_20140101_100000_002.pkl"
            corrupt_pkl.write_bytes(b"corrupted_non_pickle_bytes")

            patient = _DummyPatient(
                patient_id="abn10001",
                events=[
                    _DummyEvent(
                        record_id="abn10001_20140101_100000_001",
                        signal_file=str(valid_pkl),
                        label=1,
                        split="train",
                    ),
                    _DummyEvent(
                        record_id="abn10001_20140101_100000_002",
                        signal_file=str(corrupt_pkl),
                        label=2,
                        split="train",
                    ),
                ],
            )

            task = EEGSeizureIIIC(resample_rate=200, compute_stft=False)
            with self.assertLogs(
                "pyhealth.tasks.iiic_seizure_classification", level="WARNING"
            ) as cm:
                samples = task(patient)

            self.assertEqual(len(samples), 1)
            self.assertEqual(
                samples[0]["record_id"], "abn10001_20140101_100000_001"
            )
            self.assertTrue(
                any(
                    "Skipping unreadable IIIC segment" in msg
                    for msg in cm.output
                )
            )


if __name__ == "__main__":
    unittest.main()
