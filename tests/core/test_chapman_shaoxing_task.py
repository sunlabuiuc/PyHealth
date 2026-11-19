"""Tests for the Chapman-Shaoxing ECG classification tasks."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyhealth.datasets import ChapmanShaoxingDataset
from pyhealth.tasks import (
    ChapmanShaoxingSNOMEDClassification,
    ChapmanShaoxingSuperclassClassification,
)
from pyhealth.tasks.chapman_shaoxing_classification import (
    CHAPMAN_SNOMED_CODES,
    CHAPMAN_SUPERCLASSES,
    count_unmapped_codes,
    map_dx_to_superclasses,
    split_dx_codes,
)

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def write_record(folder: Path, name: str, dx: str, n_samples: int = 40) -> None:
    """Write a WFDB header and ``.mat`` (format 16, 24-byte offset)."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"{name} 12 500 {n_samples}"]
    lines += [f"{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in LEADS]
    lines += ["#Age: 50", "#Sex: Male", f"#Dx: {dx}"]
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")
    data = np.arange(n_samples * 12, dtype="<i2").reshape(n_samples, 12)
    (folder / f"{name}.mat").write_bytes(b"\0" * 24 + data.tobytes())


class TestLabelMapping(unittest.TestCase):
    def test_rbbb_goes_to_cd_by_default(self):
        self.assertEqual(map_dx_to_superclasses("59118001"), ["cd"])
        self.assertEqual(
            map_dx_to_superclasses("59118001", include_rbbb_in_cd=False), []
        )

    def test_unknown_codes_ignored_and_order_fixed(self):
        self.assertEqual(
            map_dx_to_superclasses("164889003;713427006;426783006"),
            ["normal", "afib"],
        )
        self.assertEqual(count_unmapped_codes(["713427006", "1;2"])["713427006"], 1)

    def test_split_handles_numbers_and_missing(self):
        self.assertEqual(split_dx_codes(426783006), ["426783006"])
        self.assertEqual(split_dx_codes(float("nan")), [])
        self.assertEqual(split_dx_codes("1, 2"), ["1", "2"])

    def test_pinned_vocabularies(self):
        task = ChapmanShaoxingSuperclassClassification()
        self.assertEqual(
            task.output_schema["labels"], ("multilabel", {"labels": list(CHAPMAN_SUPERCLASSES)})
        )
        snomed = ChapmanShaoxingSNOMEDClassification()
        self.assertEqual(len(snomed.output_schema["labels"][1]["labels"]), 55)
        self.assertEqual(len(CHAPMAN_SNOMED_CODES), 55)
        with self.assertRaises(ValueError):
            ChapmanShaoxingSNOMEDClassification(codes=["1", "1"])


@unittest.skipUnless(importlib.util.find_spec("wfdb"), "wfdb is not installed")
class TestChapmanShaoxingTaskEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "ecg"
        sub = root / "WFDBRecords" / "01" / "010"
        write_record(sub, "JS00001", "164889003,59118001")
        write_record(sub, "JS00002", "426783006")
        write_record(sub, "JS00003", "713427006")  # no mapped superclass
        cls.dataset = ChapmanShaoxingDataset(
            root=str(root),
            metadata_cache_dir=str(cls.tmp / "meta"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_superclass_samples(self):
        samples = self.dataset.set_task(
            ChapmanShaoxingSuperclassClassification(), num_workers=1
        )
        self.assertEqual(len(samples), 2)
        by_id = {samples[i]["record_id"]: samples[i] for i in range(len(samples))}
        signal = by_id["JS00001"]["signal"]
        self.assertEqual(tuple(signal.shape), (12, 40))
        self.assertAlmostEqual(float(signal[0].mean()), 0.0, places=4)
        # normal, cd, mi, sttc, other, afib, hyp
        self.assertEqual(by_id["JS00001"]["labels"].tolist(), [0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(by_id["JS00002"]["labels"].tolist(), [1, 0, 0, 0, 0, 0, 0])

    def test_snomed_samples_keep_unlisted_records_out(self):
        task = ChapmanShaoxingSNOMEDClassification(
            codes=["426783006", "59118001"], standardize=False
        )
        samples = self.dataset.set_task(task, num_workers=1)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples.output_processors["labels"].size(), 2)


if __name__ == "__main__":
    unittest.main()
