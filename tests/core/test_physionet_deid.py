"""
Unit tests for the PhysioNetDeIDDataset and DeIDNERTask classes.

Author:
    Matt McKenna (mtm16@illinois.edu)
"""
import logging
import os
from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import PhysioNetDeIDDataset
from pyhealth.datasets.physionet_deid import (
    bio_tag,
    classify_phi,
    phi_spans_in_original,
)
from pyhealth.tasks import DeIDNERTask


class TestPhysioNetDeIDDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "physionet_deid"
        )
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.dataset = PhysioNetDeIDDataset(
            root=str(cls.root), cache_dir=cls.cache_dir.name
        )
        cls.task = DeIDNERTask()
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.cache_dir.cleanup()

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 10)

    def test_patient_ids(self):
        ids = set(self.dataset.unique_patient_ids)
        self.assertEqual(
            ids,
            {"10", "20", "60", "70", "80", "90", "100", "110", "120", "130"},
        )

    def test_patient_10_has_two_notes(self):
        events = self.dataset.get_patient("10").get_events()
        self.assertEqual(len(events), 2)

    def test_patient_20_has_one_note(self):
        events = self.dataset.get_patient("20").get_events()
        self.assertEqual(len(events), 1)

    def test_patient_60_has_one_note(self):
        events = self.dataset.get_patient("60").get_events()
        self.assertEqual(len(events), 1)

    def test_patient_10_note1_has_tokens_and_labels(self):
        events = self.dataset.get_patient("10").get_events()
        note1 = events[0]
        self.assertIn("text", note1)
        self.assertIn("labels", note1)

    def test_patient_10_note1_token_count(self):
        """Note 1 for patient 10 should have the right number of tokens."""
        events = self.dataset.get_patient("10").get_events()
        note1 = events[0]
        tokens = note1["text"].split(" ")
        self.assertEqual(len(tokens), 21)

    def test_patient_20_no_phi(self):
        """Patient 20's note has no PHI, all labels should be O."""
        events = self.dataset.get_patient("20").get_events()
        labels = events[0]["labels"].split(" ")
        self.assertTrue(all(lbl == "O" for lbl in labels))

    def test_patient_60_has_name_labels(self):
        """Patient 60's note starts with NAME."""
        events = self.dataset.get_patient("60").get_events()
        labels = events[0]["labels"].split(" ")
        self.assertEqual(labels[0], "B-NAME")
        self.assertEqual(labels[1], "I-NAME")

    def test_patient_60_has_location_label(self):
        """Patient 60's note has LOCATION."""
        events = self.dataset.get_patient("60").get_events()
        tokens = events[0]["text"].split(" ")
        labels = events[0]["labels"].split(" ")
        lakewood_idx = tokens.index("Lakewood")
        self.assertEqual(labels[lakewood_idx], "B-LOCATION")
        self.assertEqual(labels[lakewood_idx + 1], "I-LOCATION")

    def test_patient_60_has_date_label(self):
        """Patient 60's note has DATE."""
        events = self.dataset.get_patient("60").get_events()
        tokens = events[0]["text"].split(" ")
        labels = events[0]["labels"].split(" ")
        date_idx = tokens.index("11/05/2097")
        self.assertEqual(labels[date_idx], "B-DATE")

    def test_patient_60_has_profession_label(self):
        """Patient 60's note has PROFESSION."""
        events = self.dataset.get_patient("60").get_events()
        tokens = events[0]["text"].split(" ")
        labels = events[0]["labels"].split(" ")
        prof_idx = tokens.index("plumber.")
        self.assertEqual(labels[prof_idx], "B-PROFESSION")

    def test_stats(self):
        self.dataset.stats()

    def test_tmp_dir_cleaned_up_on_del(self):
        """Temp directory should be removed when dataset is deleted."""
        cache_dir = tempfile.TemporaryDirectory()
        dataset = PhysioNetDeIDDataset(
            root=str(self.root), cache_dir=cache_dir.name
        )
        tmp_dir = dataset._tmp_dir
        self.assertTrue(os.path.isdir(tmp_dir))
        del dataset
        self.assertFalse(os.path.exists(tmp_dir))
        cache_dir.cleanup()

    # -- Task tests --

    def test_task_sample_count(self):
        """11 notes total across 10 patients."""
        self.assertEqual(len(self.samples), 11)

    def test_task_sample_has_text_and_labels(self):
        sample = self.samples[0]
        self.assertIn("text", sample)
        self.assertIn("labels", sample)

    def test_task_text_and_labels_same_length(self):
        for sample in self.samples:
            tokens = sample["text"].split(" ")
            labels = sample["labels"].split(" ")
            self.assertEqual(len(tokens), len(labels))

    def test_task_labels_are_valid_bio(self):
        valid = {"O"}
        for cat in ("AGE", "CONTACT", "DATE", "ID", "LOCATION", "NAME", "PROFESSION"):
            valid.add(f"B-{cat}")
            valid.add(f"I-{cat}")
        for sample in self.samples:
            for label in sample["labels"].split(" "):
                self.assertIn(label, valid)

    def test_task_sample_has_patient_id(self):
        self.assertIn("patient_id", self.samples[0])


class TestPhiSpanAlignment(unittest.TestCase):
    """Tests for phi_spans_in_original with repeated non-PHI text."""

    def test_repeated_word_across_phi_boundary(self):
        """Non-PHI word 'at' appears before and after PHI tag."""
        orig = "seen at Mercy Hospital at noon"
        deid = "seen at [**Hospital**] at noon"
        spans = phi_spans_in_original(orig, deid)
        tagged = bio_tag(orig, spans)
        words = [w for w, _ in tagged]
        labels = [l for _, l in tagged]
        self.assertEqual(words, ["seen", "at", "Mercy", "Hospital", "at", "noon"])
        self.assertEqual(labels[0], "O")       # seen
        self.assertEqual(labels[1], "O")       # at (before PHI)
        self.assertIn("LOCATION", labels[2])   # Mercy
        self.assertIn("LOCATION", labels[3])   # Hospital
        self.assertEqual(labels[4], "O")       # at (after PHI)
        self.assertEqual(labels[5], "O")       # noon

    def test_repeated_chunk_between_two_phi_tags(self):
        """Same non-PHI text separates two different PHI spans."""
        orig = "Dr. Smith and Dr. Jones"
        deid = "[**Doctor Name**] and [**Doctor Name**]"
        spans = phi_spans_in_original(orig, deid)
        tagged = bio_tag(orig, spans)
        words = [w for w, _ in tagged]
        labels = [l for _, l in tagged]
        self.assertEqual(words, ["Dr.", "Smith", "and", "Dr.", "Jones"])
        self.assertNotEqual(labels[0], "O")  # Dr.
        self.assertNotEqual(labels[1], "O")  # Smith
        self.assertEqual(labels[2], "O")     # and
        self.assertNotEqual(labels[3], "O")  # Dr.
        self.assertNotEqual(labels[4], "O")  # Jones


class TestClassifyPhiFallback(unittest.TestCase):
    """Test that classify_phi logs a warning on unknown tags."""

    def test_unknown_tag_logs_warning(self):
        with self.assertLogs("pyhealth.datasets.physionet_deid", level=logging.WARNING) as cm:
            result = classify_phi("xyzzy gibberish tag")
        self.assertEqual(result, "NAME")
        self.assertTrue(any("no keyword match" in msg for msg in cm.output))


class TestDeIDNERTaskWindowing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "physionet_deid"
        )
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.dataset = PhysioNetDeIDDataset(
            root=str(cls.root), cache_dir=cls.cache_dir.name
        )
        cls.task = DeIDNERTask(window_size=10, window_overlap=5)
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.cache_dir.cleanup()

    def test_windowing_produces_more_samples(self):
        """Windowing should produce more samples than the 11 notes."""
        self.assertGreater(len(self.samples), 11)

    def test_window_size_respected(self):
        """Each window should have at most window_size tokens."""
        for sample in self.samples:
            tokens = sample["text"].split(" ")
            self.assertLessEqual(len(tokens), 10)

    def test_window_text_and_labels_same_length(self):
        for sample in self.samples:
            tokens = sample["text"].split(" ")
            labels = sample["labels"].split(" ")
            self.assertEqual(len(tokens), len(labels))

    def test_window_has_patient_id(self):
        self.assertIn("patient_id", self.samples[0])


if __name__ == "__main__":
    unittest.main()
