"""Real ``dataset.set_task(task)`` integration tests on MIMIC-IV demo data.

Regression tests for the W5 cache-A/B blocker: ``set_task`` is the only
code path that instantiates every ``input_schema``/``output_schema``
processor spec, so only a real ``set_task`` exercises the
``labs_mask`` / ``forward_fill`` combination (and the note tokenizer,
tables, and mask numerics) end to end.

``LabsMIMIC4`` runs on upstream's ``test-resources/core/mimic4demo``
fixture, the same demo data other MIMIC-IV task tests use.
``NotesLabsMIMIC4`` additionally needs discharge/radiology notes, which the
demo does not ship, so the test stages a minimal note root (same
``note/*.csv.gz`` layout ``mimic4_note.yaml`` expects) with notes for the
demo patient who has two stays and dies in the second one.
"""

import gzip
import os
import tempfile
import unittest
from pathlib import Path

import torch

DEMO_ROOT = (
    Path(__file__).resolve().parents[2]
    / "test-resources"
    / "core"
    / "mimic4demo"
)

# Demo patient 10008: stay 20012 (2151-10-05 -> 2151-10-25, survived),
# stay 20013 (2151-12-15 -> 2151-12-20, died) -> mortality label 1.
NOTES_PATIENT = 10008


def _write_note_csv(path, rows):
    import csv

    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject_id",
                "charttime",
                "note_id",
                "hadm_id",
                "note_type",
                "note_seq",
                "storetime",
                "text",
            ]
        )
        writer.writerows(rows)


def _stage_note_root(root):
    note_dir = Path(root) / "note"
    note_dir.mkdir(parents=True, exist_ok=True)
    _write_note_csv(
        note_dir / "discharge.csv.gz",
        [
            (
                NOTES_PATIENT,
                "2151-10-05",
                1,
                20012,
                "discharge",
                1,
                "2151-10-25 12:00:00",
                "Chief Complaint:\n\nshortness of breath\n\n",
            ),
            (
                NOTES_PATIENT,
                "2151-12-15",
                2,
                20013,
                "discharge",
                1,
                "2151-12-20 12:00:00",
                "Chief Complaint:\n\nchest pain\n\n",
            ),
        ],
    )
    _write_note_csv(
        note_dir / "radiology.csv.gz",
        [
            (
                NOTES_PATIENT,
                "2151-10-06 09:00:00",
                3,
                20012,
                "radiology",
                1,
                "2151-10-06 10:00:00",
                "Indication:\n\nearly film\n\nImpression:\n\nno acute findings\n\n",
            ),
        ],
    )


def _processed_tensors(value):
    """Unwrap one processed field into a flat list of tensors."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        out = []
        for v in value:
            out.extend(_processed_tensors(v))
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


class TestLabsMIMIC4SetTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import MIMIC4Dataset
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        cls._tmp = tempfile.TemporaryDirectory()
        dataset = MIMIC4Dataset(
            ehr_root=str(DEMO_ROOT),
            # Same table list the e2e runner uses; patients/admissions
            # are auto-added by MIMIC4EHRDataset.
            ehr_tables=["labevents"],
            cache_dir=os.path.join(cls._tmp.name, "cache"),
        )
        cls.sample_dataset = dataset.set_task(LabsMIMIC4())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_set_task_returns_samples(self):
        self.assertGreater(len(self.sample_dataset), 0)

    def test_labels_cover_both_classes(self):
        labels = set()
        for i in range(len(self.sample_dataset)):
            labels.add(int(self.sample_dataset[i]["mortality"]))
        # Demo patient 10008 dies in the second stay; most survive.
        self.assertEqual(labels, {0, 1})

    def test_processors_fit_feature_width(self):
        # The labs stream starts with patients who have no lab events; fit
        # must scan past them. value_dim() sizes the unified numeric
        # encoder, so an unfitted processor (width 1) would break the first
        # forward with a matmul shape error.
        self.assertEqual(self.sample_dataset.input_processors["labs"].value_dim(), 10)
        self.assertEqual(
            self.sample_dataset.input_processors["labs_mask"].value_dim(), 10
        )

    def test_labs_mask_is_parallel_and_unimputed(self):
        saw_unobserved = False
        for i in range(len(self.sample_dataset)):
            sample = self.sample_dataset[i]
            # Coupled (time, value) fields process to (time, value) tensors.
            _labs_time, values = _processed_tensors(sample["labs"])
            _mask_time, mask = _processed_tensors(sample["labs_mask"])
            self.assertEqual(values.shape, mask.shape)
            self.assertTrue(torch.isfinite(values).all())
            unique = set(mask.reshape(-1).tolist())
            self.assertLessEqual(unique, {0.0, 1.0})
            if 0.0 in unique:
                saw_unobserved = True
        # The demo labs only cover a subset of the 10 categories, so some
        # mask entry must be unobserved (0.0, never forward-filled to 1.0).
        self.assertTrue(saw_unobserved)


class TestNotesLabsMIMIC4SetTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import MIMIC4Dataset
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        cls._tmp = tempfile.TemporaryDirectory()
        _stage_note_root(os.path.join(cls._tmp.name, "notes"))
        cls.dataset = MIMIC4Dataset(
            ehr_root=str(DEMO_ROOT),
            ehr_tables=["labevents"],
            note_root=os.path.join(cls._tmp.name, "notes"),
            note_tables=["discharge", "radiology"],
            cache_dir=os.path.join(cls._tmp.name, "cache"),
        )
        cls.sample_dataset = cls.dataset.set_task(NotesLabsMIMIC4())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_set_task_returns_samples(self):
        self.assertGreater(len(self.sample_dataset), 0)

    def test_note_text_survives_processing(self):
        # Raw task records (pre-processor) carry the extracted sections.
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4()
        patient = self.dataset.get_patient(str(NOTES_PATIENT))
        record = task(patient)[0]
        texts, _times = record["admission_note_times"]
        joined = " [SEP] ".join(texts)
        self.assertIn("shortness of breath", joined)
        self.assertIn("chest pain", joined)
        self.assertIn("early film", joined)

    def test_processed_sample_has_notes_labs_and_mask(self):
        found_notes = False
        for i in range(len(self.sample_dataset)):
            sample = self.sample_dataset[i]
            self.assertIn("labs", sample)
            self.assertIn("labs_mask", sample)
            note_tensors = [
                t
                for t in _processed_tensors(sample["admission_note_times"])
                if t.numel() > 0
            ]
            if note_tensors:
                found_notes = True
        self.assertTrue(found_notes)


if __name__ == "__main__":
    unittest.main()
