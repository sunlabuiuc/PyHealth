"""Offline synthetic LUDB manifest tests; no real records or downloads.

Contributors: Thi Van Trinh Tran and Zexi Li.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets.ludb import LUDBDataset, LUDBManifestSplit


def _artifacts(root):
    records, prepared = root / "records", root / "prepared"
    records.mkdir()
    (prepared / "masks").mkdir(parents=True)
    names = ["i", "ii", "iii", "avr", "avl", "avf",
             "v1", "v2", "v3", "v4", "v5", "v6"]
    rows = []
    # Reverse source-patient order exercises numeric patient ordering.
    for pid, part in ((10, "test"), (2, "train")):
        signal = np.tile(np.arange(5000, dtype=np.int16)[:, None], (1, 12))
        (records / f"{pid}.dat").write_bytes(signal.astype("<i2").tobytes())
        header = [f"{pid} 12 500 5000"] + [
            f"{pid}.dat 16 1000(0)/mV 16 0 0 0 0 {name}" for name in names
        ]
        (records / f"{pid}.hea").write_text("\n".join(header) + "\n")
        labels = np.tile((np.arange(5000) // 100 % 4)[None, :], (12, 1))
        ignored = np.zeros_like(labels, dtype=bool)
        ignored[0, 20:30] = True
        ignored[0, 512:1024] = True
        np.savez(prepared / "masks" / f"{pid}.npz", labels=labels,
                 ignore_mask=ignored, lead_names=np.asarray(names))
        for index, start in enumerate((0, 512)):
            rows.append({
                "patient_id": pid, "split": part, "lead": "i", "lead_index": 0,
                "window_index": index, "start_sample": start,
                "end_sample_exclusive": start + 512, "window_length": 512,
                "has_supervised_points": index == 0,
            })
    manifest = root / "custom_windows.csv"
    split = root / "patient_split.csv"
    metadata = root / "metadata_prepared.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.DataFrame({"patient_id": [10, 2], "split": ["test", "train"]}).to_csv(
        split, index=False,
    )
    pd.DataFrame({"ID": [2, 10], "age": [30, 60], "qc": ["keep", "keep"]}).to_csv(
        metadata, index=False,
    )
    return {
        "manifest_path": manifest, "records_root": records, "mask_root": prepared,
        "split_manifest_path": split, "metadata_path": metadata,
        "cache_dir": root / "cache",
    }


class TestLUDB(unittest.TestCase):
    def test_split_preserves_membership_and_numeric_order(self):
        split = LUDBManifestSplit((10, 2), (11,))
        parts = split.split_indices({"10": [0, 1], "11": [2], "2": [3, 4]})
        self.assertEqual(parts[0].tolist(), [3, 4, 0, 1])
        self.assertEqual(parts[1].tolist(), [2])
        with self.assertRaises(ValueError):
            split.split_indices({"2": [0]})
        with self.assertRaises(ValueError):
            LUDBManifestSplit((1,), (1,))

    def test_dataset_resolves_manifest_before_native_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            before = hashlib.sha256(args["manifest_path"].read_bytes()).hexdigest()
            raw = LUDBDataset(**args)
            self.assertEqual(
                raw.config.tables["windows"].file_path, "custom_windows.csv",
            )
            self.assertIsNone(raw.config.tables["windows"].timestamp)
            self.assertEqual(raw.manifest_split.train_patients, (2,))
            self.assertEqual(raw.manifest_split.test_patients, (10,))
            self.assertEqual(before, hashlib.sha256(
                args["manifest_path"].read_bytes(),
            ).hexdigest())
            self.assertIn(str(args["cache_dir"]), str(raw.cache_dir))

    def test_disagreement_is_not_silently_resplit(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            frame = pd.read_csv(args["manifest_path"])
            frame.loc[0, "split"] = "train"
            frame.to_csv(args["manifest_path"], index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                LUDBDataset(**args)

    def test_task_cache_identity_covers_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            raw = LUDBDataset(**args)
            first = raw.default_task.artifact_hashes
            path = args["mask_root"] / "masks" / "2.npz"
            with np.load(path) as stored:
                arrays = {name: stored[name].copy() for name in stored.files}
            arrays["ignore_mask"][0, 0] = True
            np.savez(path, **arrays)
            self.assertNotEqual(first, raw.default_task.artifact_hashes)
