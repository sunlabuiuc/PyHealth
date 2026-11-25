"""
Unit tests for MDACEDataset and MDACEEvidenceCoding on a synthetic fixture.

Contributors:
    Juri Moon (jmoon326)

Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958
"""
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MDACEDataset
from pyhealth.datasets.mdace import trim_span
from pyhealth.tasks import MDACEEvidenceCoding

ROOT = Path(__file__).parent.parent.parent / "test-resources" / "core" / "mdace"


class TestMDACE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.dataset = MDACEDataset(
            root=str(ROOT / "MDace"),
            mimic3_root=str(ROOT / "mimic3"),
            splits_dir=str(ROOT / "splits"),
            cache_dir=cls.cache_dir.name,
        )
        cls.all_samples = cls.dataset.set_task(
            MDACEEvidenceCoding(discharge_summaries_only=False)
        )
        cls.ds_samples = cls.dataset.set_task()
        cls.by_note = {s["note_id"]: s for s in cls.all_samples}

    @classmethod
    def tearDownClass(cls):
        cls.all_samples.close()
        cls.ds_samples.close()
        cls.cache_dir.cleanup()

    def _events(self, pid):
        return self.dataset.get_patient(pid).get_events(event_type="mdace")

    def test_patients_and_notes(self):
        # Distractor note 99 (patient 503) is not annotated and must be dropped.
        self.assertEqual(sorted(self.dataset.unique_patient_ids), ["501", "502"])
        self.assertEqual(len(self._events("501")), 3)
        self.assertEqual(len(self._events("502")), 1)

    def test_trim_span(self):
        text = "Dx: hypertension, ok"
        self.assertEqual(trim_span((3, 16), text), (4, 15))
        self.assertEqual(trim_span((4, 8), text), (4, 8))
        self.assertEqual(trim_span((17, 20), text), (18, 20))  # end == len

    def test_trimmed_spans_and_alignment(self):
        # Raw task output (before the multilabel processor turns codes
        # into a multi-hot tensor).
        raw = MDACEEvidenceCoding()(self.dataset.get_patient("501"))
        s = next(r for r in raw if r["note_id"] == "11")
        self.assertEqual(s["icd_codes"], ["401.9", "250.00"])
        self.assertEqual(len(s["evidence_spans"]), len(s["icd_codes"]))
        self.assertEqual(s["evidence_spans"][0], [[20, 31]])  # both ends trimmed
        self.assertEqual(s["evidence_spans"][1], [[34, 48], [51, 58]])

    def test_split_column(self):
        splits = {nid: s["split"] for nid, s in self.by_note.items()}
        self.assertEqual(
            splits, {"11": "train", "12": "train", "21": "val", "31": "test"}
        )

    def test_no_splits_dir(self):
        # Indexing step only: building a second full dataset takes over a second.
        with tempfile.TemporaryDirectory() as out:
            df = MDACEDataset._index_data(
                ROOT / "MDace" / "Inpatient" / "ICD-9" / "1.0",
                ROOT / "mimic3" / "NOTEEVENTS.csv.gz",
                None,
                out,
            )
        self.assertEqual(len(df), 4)
        self.assertTrue(df["split"].isna().all())

    def test_discharge_filter(self):
        self.assertEqual(len(self.all_samples), 4)
        self.assertEqual(len(self.ds_samples), 3)
        self.assertEqual(
            {s["category"] for s in self.ds_samples}, {"Discharge summary"}
        )

    def test_sample_schema(self):
        s = self.by_note["21"]
        for key in ("text", "icd_codes", "note_id", "hadm_id", "split",
                    "category", "evidence_spans", "patient_id"):
            self.assertIn(key, s)
        self.assertEqual(s["hadm_id"], "900002")
        self.assertIsInstance(s["text"], str)
        self.assertEqual(self.all_samples.output_processors["icd_codes"].size(), 4)


if __name__ == "__main__":
    unittest.main()
