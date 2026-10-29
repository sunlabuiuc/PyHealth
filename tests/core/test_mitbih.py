"""Offline synthetic MIT-BIH dataset, split and task tests; no real records.

Contributors: Thi Van Trinh Tran and Zexi Li.
Five tiny synthetic WFDB records (3 s each) are written at test time. Covers
the abnormal-beat ratio, the paper split's exclusions, the YAML metadata
table, step- and window-level labels, and the end-to-end ``set_task`` path.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from distributed import LocalCluster

from pyhealth.datasets import MITBIHAnomalyRatioSplit, MITBIHDataset
from pyhealth.datasets.mitbih import MITBIH_METADATA_FILE, beat_anomaly_ratio
from pyhealth.tasks import MITBIHAnomalyDetection
from pyhealth.tasks.mitbih import anomaly_mask

FS = 360
N_SAMPLES = 3 * FS
BEATS = np.arange(90, N_SAMPLES, 180)  # 6 beats, every 0.5 s

# record -> (leads, beat symbols, extra non-beat annotations)
RECORDS = {
    "100": (("MLII", "V1"), ["N"] * 6, [(10, "+"), (500, "~")]),
    "101": (("MLII", "V1"), ["N", "N", "V", "N", "N", "N"], [(10, "+")]),
    "102": (("MLII", "V1"), ["N", "V", "N", "V", "A", "N"], []),
    "103": (("MLII", "V1"), ["V"] * 6, []),  # 100% abnormal: always excluded
    "104": (("V5", "V2"), ["N", "V", "N", "N", "N", "N"], []),  # wrong leads
}


def _write_records(root: Path) -> None:
    import wfdb

    t = np.arange(N_SAMPLES) / FS
    for rid, (leads, beats, extra) in RECORDS.items():
        signal = np.stack([np.sin(2 * np.pi * t), 0.5 * np.cos(2 * np.pi * t)], 1)
        wfdb.wrsamp(
            rid,
            fs=FS,
            units=["mV", "mV"],
            sig_name=list(leads),
            p_signal=signal,
            fmt=["212", "212"],
            comments=["69 M", f"synthetic record {rid}"],
            write_dir=str(root),
        )
        ann = sorted([(int(s), b) for s, b in zip(BEATS, beats)] + extra)
        wfdb.wrann(
            rid,
            "atr",
            np.array([s for s, _ in ann]),
            [b for _, b in ann],
            write_dir=str(root),
        )


class TestMITBIHHelpers(unittest.TestCase):
    def test_beat_anomaly_ratio_ignores_non_beat_annotations(self):
        self.assertEqual(beat_anomaly_ratio(["N", "+", "V", "~", "|", "N"]), 1 / 3)
        self.assertEqual(beat_anomaly_ratio(["+", "~"]), 0.0)

    def test_anomaly_mask_expands_abnormal_beats_only(self):
        mask = anomaly_mask(
            np.array([100, 300]), ["V", "N"], 400, fs=1000, half_window_ms=10
        )
        self.assertEqual(mask.sum(), 21)
        self.assertTrue(mask[90:111].all())
        self.assertFalse(mask[280:321].any())

    def test_anomaly_mask_clips_at_edges(self):
        mask = anomaly_mask(np.array([2]), ["V"], 10, fs=1000, half_window_ms=5)
        self.assertEqual(mask.tolist(), [1] * 8 + [0] * 2)

    def test_task_rejects_bad_config(self):
        for kwargs in (
            {"window_len": 0},
            {"step_size": -1},
            {"target_fs": 0},
            {"half_window_ms": -1},
            {"label_level": "beat"},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MITBIHAnomalyDetection(**kwargs)

    def test_output_schema_follows_label_level(self):
        self.assertEqual(MITBIHAnomalyDetection().output_schema, {"label": "tensor"})
        self.assertEqual(
            MITBIHAnomalyDetection(label_level="window").output_schema,
            {"label": "binary"},
        )


SPLIT_RATIOS = {"a": 0.0, "b": 0.2, "c": 0.5, "d": 1.0, "e": 0.1}
SPLIT_LEADS = {k: ("MLII", "V1") for k in "abcd"} | {"e": ("V5", "V2")}


class TestMITBIHAnomalyRatioSplit(unittest.TestCase):
    def test_paper_mode_filters_leads_and_all_abnormal(self):
        split = MITBIHAnomalyRatioSplit(SPLIT_RATIOS, SPLIT_LEADS, n_train=1, n_test=1)
        self.assertEqual(split.select_records(), (["a"], ["c"]))

    def test_fraction_mode_uses_all_leads(self):
        split = MITBIHAnomalyRatioSplit(
            SPLIT_RATIOS, SPLIT_LEADS, mode="fraction", train_frac=0.5
        )
        self.assertEqual(split.select_records(), (["a", "e"], ["b", "c"]))

    def test_ties_broken_by_record_id(self):
        split = MITBIHAnomalyRatioSplit(
            {"z": 0.0, "y": 0.0, "x": 0.3}, mode="fraction", train_frac=0.67
        )
        self.assertEqual(split.select_records(), (["y", "z"], ["x"]))

    def test_split_indices_and_to_dict(self):
        split = MITBIHAnomalyRatioSplit(SPLIT_RATIOS, SPLIT_LEADS, n_train=1, n_test=1)
        parts = split.split_indices({"a": [0, 1], "c": [4], "d": [2], "e": [3]})
        self.assertEqual([p.tolist() for p in parts], [[0, 1], [4]])
        self.assertEqual(parts[0].dtype, np.int64)
        info = split.to_dict()
        self.assertEqual((info["train_records"], info["test_records"]), (["a"], ["c"]))

    def test_too_few_records_raises(self):
        split = MITBIHAnomalyRatioSplit(SPLIT_RATIOS, SPLIT_LEADS)  # needs 6 + 4
        with self.assertRaises(ValueError):
            split.select_records()

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            MITBIHAnomalyRatioSplit(SPLIT_RATIOS, mode="random").select_records()


def _threaded_cluster(**kwargs):
    # Same Dask work in worker threads instead of child processes, which take
    # seconds to start; as in tests/core/survival_test_utils.py.
    return LocalCluster(**{**kwargs, "processes": False})


class TestMITBIHDatasetAndTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cluster_patch = patch(
            "pyhealth.datasets.base_dataset.DaskCluster", _threaded_cluster
        )
        cls.cluster_patch.start()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "mitdb"
        cls.meta_dir = cls.tmp / "meta"
        cls.root.mkdir()
        _write_records(cls.root)
        cls.dataset = MITBIHDataset(
            root=cls.root, metadata_cache_dir=cls.meta_dir, cache_dir=cls.tmp / "cache"
        )
        # One-time costs (importing scipy.signal, indexing the patients) are
        # paid here rather than by whichever task test happens to run first.
        MITBIHAnomalyDetection()(cls.dataset.get_patient("100"))

    @classmethod
    def tearDownClass(cls):
        cls.cluster_patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            MITBIHDataset(root=self.root / "missing", metadata_cache_dir=self.meta_dir)

    def test_missing_annotations_raises(self):
        partial = self.tmp / "partial"
        partial.mkdir()
        shutil.copy(self.root / "100.hea", partial)
        shutil.copy(self.root / "100.dat", partial)
        with self.assertRaises(FileNotFoundError):
            MITBIHDataset(root=partial, metadata_cache_dir=self.tmp / "meta2")

    def test_metadata_table_matches_yaml_config(self):
        csv_path = self.meta_dir / MITBIH_METADATA_FILE
        self.assertTrue(csv_path.is_file())
        self.assertFalse((self.root / MITBIH_METADATA_FILE).exists())
        config_path = (
            Path(__file__).parents[2] / "pyhealth/datasets/configs/mitbih.yaml"
        )
        table = yaml.safe_load(config_path.read_text())["tables"]["records"]
        self.assertEqual(table["file_path"], MITBIH_METADATA_FILE)
        df = pd.read_csv(csv_path, dtype=str)
        self.assertEqual(list(df.columns), [table["patient_id"]] + table["attributes"])
        self.assertEqual(sorted(df["record_id"]), sorted(RECORDS))

    def test_record_metadata(self):
        meta = self.dataset.record_metadata
        self.assertEqual(sorted(meta), sorted(RECORDS))
        self.assertEqual(meta["104"]["leads"], ("V5", "V2"))
        self.assertEqual(meta["100"]["anomaly_ratio"], 0.0)
        self.assertAlmostEqual(meta["102"]["anomaly_ratio"], 0.5)
        self.assertEqual(meta["100"]["n_samples"], N_SAMPLES)
        self.assertIn("synthetic record 100", meta["100"]["comments"])

    def test_patients_have_one_record_event(self):
        self.assertEqual(sorted(self.dataset.unique_patient_ids), sorted(RECORDS))
        patient = self.dataset.get_patient("101")
        events = patient.get_events(event_type="records")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["leads"], "MLII,V1")
        self.assertEqual(float(events[0]["fs"]), FS)
        self.assertIn("synthetic record 101", events[0]["comments"])

    def test_anomaly_ratio_split_from_dataset(self):
        split = self.dataset.anomaly_ratio_split(n_train=2, n_test=1)
        self.assertEqual(split.select_records(), (["100", "101"], ["102"]))

    def test_task_windows_and_labels(self):
        task = MITBIHAnomalyDetection()
        samples = task(self.dataset.get_patient("101"))
        n_resampled = N_SAMPLES * 125 // 360  # 375
        self.assertEqual(len(samples), n_resampled // 128)
        first = samples[0]
        self.assertEqual(first["signal"].shape, (128, 2))
        self.assertEqual(first["signal"].dtype, np.float32)
        self.assertEqual(first["label"].dtype, np.int64)
        self.assertEqual(first["record_id"], "101")
        self.assertEqual([s["window_start"] for s in samples], [0, 128])

        # The only abnormal beat (V at 450 / 360 Hz = 1.25 s) maps to step ~156
        # at 125 Hz, with +/-150 ms (~19 steps) labelled on each side.
        label = np.concatenate([s["label"] for s in samples])
        self.assertTrue(label[140:172].all())
        self.assertEqual(label[:130].sum() + label[180:].sum(), 0)
        self.assertEqual(first["description"], "69 M; synthetic record 101")

    def test_label_width_follows_half_window(self):
        patient = self.dataset.get_patient("101")
        widths = {}
        for half_ms in (0.0, 50.0, 150.0):
            samples = MITBIHAnomalyDetection(half_window_ms=half_ms)(patient)
            widths[half_ms] = int(sum(s["label"].sum() for s in samples))
        # 2 * round(125 Hz * ms / 1000) + 1 steps, up to resampling rounding.
        self.assertLessEqual(widths[0.0], 2)
        self.assertAlmostEqual(widths[50.0], 13, delta=2)
        self.assertAlmostEqual(widths[150.0], 39, delta=2)

    def test_window_level_labels(self):
        task = MITBIHAnomalyDetection(label_level="window")
        samples = task(self.dataset.get_patient("101"))
        # The abnormal beat (step ~156) lies in the second window only.
        self.assertEqual([s["label"] for s in samples], [0, 1])
        overlapping = MITBIHAnomalyDetection(label_level="window", step_size=64)
        labels = [s["label"] for s in overlapping(self.dataset.get_patient("101"))]
        # Windows start at 0, 64, 128, 192; steps ~137-175 are labelled.
        self.assertEqual(labels, [0, 1, 1, 0])

    def test_task_skips_patient_without_events(self):
        class _Empty:
            def get_events(self, event_type=None):
                return []

        self.assertEqual(MITBIHAnomalyDetection()(_Empty()), [])

    def test_set_task_with_paper_split(self):
        split = self.dataset.anomaly_ratio_split(n_train=2, n_test=1)
        train, test = self.dataset.set_task(split=split)
        self.assertEqual(
            {train[i]["record_id"] for i in range(len(train))}, {"100", "101"}
        )
        self.assertEqual({test[i]["record_id"] for i in range(len(test))}, {"102"})
        self.assertEqual(tuple(train[0]["signal"].shape), (128, 2))
        self.assertEqual(tuple(test[0]["label"].shape), (128,))

    def test_set_task_with_window_labels(self):
        split = self.dataset.anomaly_ratio_split(n_train=2, n_test=1)
        _, test = self.dataset.set_task(
            MITBIHAnomalyDetection(label_level="window"), split=split
        )
        labels = [int(test[i]["label"]) for i in range(len(test))]
        self.assertEqual(len(labels), 2)
        self.assertEqual(max(labels), 1)


if __name__ == "__main__":
    unittest.main()
