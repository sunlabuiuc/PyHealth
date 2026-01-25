import csv
import os
import tempfile

from base import BaseTestCase
from pyhealth.datasets import MIMIC3NoteDataset
from pyhealth.tasks.sdoh_utils import TARGET_CODES, codes_to_multihot


class TestSdohMimic3Notes(BaseTestCase):
    def setUp(self):
        self.set_random_seed()

    def test_mimic3_note_dataset(self):
        """Test MIMIC3NoteDataset with label filtering and categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            noteevents_path = os.path.join(tmpdir, "NOTEEVENTS.csv")
            with open(noteevents_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "SUBJECT_ID",
                        "HADM_ID",
                        "CHARTDATE",
                        "CHARTTIME",
                        "CATEGORY",
                        "TEXT",
                    ]
                )
                writer.writerow(
                    [
                        "1",
                        "10",
                        "2020-01-01",
                        "2020-01-01 12:00:00",
                        "Physician",
                        "Pt is homeless",
                    ]
                )
                writer.writerow(
                    [
                        "1",
                        "10",
                        "2020-01-01",
                        "2020-01-01 13:00:00",
                        "Radiology",
                        "XR chest normal",
                    ]
                )
                writer.writerow(
                    [
                        "2",
                        "20",
                        "2020-02-01",
                        "2020-02-01 09:00:00",
                        "Physician",
                        "No issues reported",
                    ]
                )

            label_map = {
                "10": {"manual": {"V600"}, "true": set()},
                "20": {"manual": set(), "true": set()},
            }

            dataset = MIMIC3NoteDataset(
                noteevents_path=noteevents_path,
                hadm_ids=["10"],
                include_categories=["Physician"],
            )
            sample_dataset = dataset.set_task(
                label_source="manual",
                label_map=label_map,
            )

            self.assertEqual(1, len(sample_dataset))
            sample = next(iter(sample_dataset))
            self.assertEqual("10", sample["visit_id"])
            self.assertEqual(["Pt is homeless"], sample["notes"])
            self.assertEqual(["Physician"], sample["note_categories"])

            expected_label = codes_to_multihot({"V600"}, TARGET_CODES).tolist()
            self.assertEqual(expected_label, sample["label"].tolist())

    def test_note_sorting_and_chartdate_fallback(self):
        """Test ordering with missing charttime and chartdate fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            noteevents_path = os.path.join(tmpdir, "NOTEEVENTS.csv")
            with open(noteevents_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "SUBJECT_ID",
                        "HADM_ID",
                        "CHARTDATE",
                        "CHARTTIME",
                        "CATEGORY",
                        "TEXT",
                    ]
                )
                writer.writerow(
                    [
                        "1",
                        "10",
                        "2020-01-02",
                        "",
                        "Physician",
                        "Later note",
                    ]
                )
                writer.writerow(
                    [
                        "1",
                        "10",
                        "2020-01-01",
                        "2020-01-01 08:00:00",
                        "Physician",
                        "Earlier note",
                    ]
                )

            label_map = {"10": {"manual": {"V600"}, "true": set()}}
            dataset = MIMIC3NoteDataset(
                noteevents_path=noteevents_path,
                include_categories=["Physician"],
            )
            sample_dataset = dataset.set_task(
                label_source="manual",
                label_map=label_map,
            )

            sample = next(iter(sample_dataset))
            self.assertEqual(["Earlier note", "Later note"], sample["notes"])
