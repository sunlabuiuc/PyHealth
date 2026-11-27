import csv
import pickle
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC4CXRDataset
from pyhealth.models import TransformersModel
from pyhealth.tasks.prior_exam_preference import PriorExamPreferenceTask

ANNOTATIONS = [
    {
        "study_id": "50000001",
        "sentence_id": "1",
        "new_sentence": "The lungs are clear.",
        "orig_sentence": "The lungs are clear.",
        "section": "findings",
    },
    {
        "study_id": "50000001",
        "sentence_id": "2",
        "new_sentence": "No acute change.",
        "orig_sentence": "No interval change from prior.",
        "section": "findings",
    },
    {
        "study_id": "50000001",
        "sentence_id": "1",
        "new_sentence": "",
        "orig_sentence": "Stable exam.",
        "section": "impression",
    },
]


def _write_annotations(path, rows=ANNOTATIONS):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ANNOTATIONS[0]))
        writer.writeheader()
        writer.writerows(rows)


def _event(study_id, dicom_id, image_path, timestamp=None):
    return {
        "patient_id": "10000001",
        "event_type": "metadata",
        "timestamp": timestamp or datetime(2026, 1, 1, 12, 0),
        "metadata/study_id": study_id,
        "metadata/dicom_id": dicom_id,
        "metadata/image_path": image_path,
    }


def _patient(*events):
    frame = pl.from_dicts(events).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms"))
    )
    return Patient(patient_id="10000001", data_source=frame)


def _metadata_row(dicom_id, view):
    return {
        "dicom_id": dicom_id,
        "subject_id": "10000001",
        "study_id": "50000001",
        "PerformedProcedureStepDescription": "CHEST",
        "ViewPosition": view,
        "Rows": "1024",
        "Columns": "1024",
        "StudyDate": "20260101",
        "StudyTime": "120000",
        "ProcedureCodeSequence_CodeMeaning": "CHEST",
        "ViewCodeSequence_CodeMeaning": view,
        "PatientOrientationCodeSequence_CodeMeaning": "Erect",
    }


def _write_tiny_bert(path):
    """Writes a random local encoder so the model smoke test needs no network."""
    torch.manual_seed(0)
    vocab = path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tokenizer.save_pretrained(path)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    BertModel(config).save_pretrained(path)


class TestPriorExamPreferenceTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.annotations = self.root / "train.csv"
        _write_annotations(self.annotations)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _task(self, mode="sentence_only"):
        return PriorExamPreferenceTask({"train": self.annotations}, context_mode=mode)

    def _samples(self, mode="sentence_only"):
        patient = _patient(
            _event("50000001", "image-b", "/synthetic/b.jpg"),
            _event("50000001", "image-a", "/synthetic/a.jpg"),
        )
        return self._task(mode)(patient)

    def test_reconstructs_reports_and_edit_provenance(self):
        samples = self._samples()
        self.assertEqual(len(samples), 3)
        self.assertEqual(
            [sample["edit_type"] for sample in samples],
            ["unchanged", "rewritten", "deleted"],
        )
        self.assertEqual([sample["needs_edit"] for sample in samples], [0, 1, 1])
        self.assertEqual(
            samples[0]["rejected_report"],
            "FINDINGS:\nThe lungs are clear.\n"
            "No interval change from prior.\nIMPRESSION:\nStable exam.",
        )
        self.assertEqual(
            samples[0]["chosen_report"],
            "FINDINGS:\nThe lungs are clear.\nNo acute change.\nIMPRESSION:",
        )
        self.assertEqual(
            samples[0]["edit_counts"],
            {"unchanged": 1, "rewritten": 1, "deleted": 1},
        )

    def test_preserves_multiple_images_in_stable_order(self):
        sample = self._samples()[0]
        self.assertEqual(
            sample["image_paths"], ["/synthetic/a.jpg", "/synthetic/b.jpg"]
        )
        self.assertEqual(sample["dicom_ids"], ["image-a", "image-b"])

    def test_context_modes_only_change_text(self):
        by_mode = {
            mode: self._samples(mode)
            for mode in (
                "sentence_only",
                "neighbor_context",
                "section_context",
            )
        }
        target = 1
        self.assertEqual(
            by_mode["sentence_only"][target]["text"],
            "Target sentence:\nNo interval change from prior.",
        )
        self.assertIn(
            "Previous sentence:\nThe lungs are clear.",
            by_mode["neighbor_context"][target]["text"],
        )
        self.assertIn(
            "[TARGET] No interval change from prior. [/TARGET]",
            by_mode["section_context"][target]["text"],
        )
        for mode, samples in by_mode.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    [sample["record_id"] for sample in samples],
                    [
                        "50000001:findings:1",
                        "50000001:findings:2",
                        "50000001:impression:1",
                    ],
                )
                self.assertEqual(
                    [sample["needs_edit"] for sample in samples], [0, 1, 1]
                )

    def test_task_is_picklable_for_set_task_workers(self):
        restored = pickle.loads(pickle.dumps(self._task()))
        self.assertEqual(restored.context_mode, "sentence_only")
        self.assertEqual(
            restored.annotation_files,
            {"train": str(self.annotations.resolve())},
        )

    def test_rejects_duplicate_annotation_key(self):
        _write_annotations(self.annotations, [ANNOTATIONS[0], ANNOTATIONS[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._task()

    def test_rejects_unexpected_section(self):
        row = {**ANNOTATIONS[0], "section": "comparison"}
        _write_annotations(self.annotations, [row])
        with self.assertRaisesRegex(ValueError, "section must be"):
            self._task()

    def test_rejects_missing_column_and_bad_position(self):
        with self.annotations.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[name for name in ANNOTATIONS[0] if name != "section"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    name: value
                    for name, value in ANNOTATIONS[0].items()
                    if name != "section"
                }
            )
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self._task()

        row = {**ANNOTATIONS[0], "sentence_id": "not-an-integer"}
        _write_annotations(self.annotations, [row])
        with self.assertRaisesRegex(ValueError, "Invalid ReXPref-Prior row"):
            self._task()

    def test_rejects_study_in_multiple_splits(self):
        validation = self.root / "val.csv"
        _write_annotations(validation, [ANNOTATIONS[0]])
        with self.assertRaisesRegex(ValueError, "more than one annotation split"):
            PriorExamPreferenceTask({"train": self.annotations, "val": validation})

    def test_rejects_patient_crossing_release_splits(self):
        validation = self.root / "val.csv"
        validation_row = {**ANNOTATIONS[0], "study_id": "50000002"}
        _write_annotations(validation, [validation_row])
        task = PriorExamPreferenceTask({"train": self.annotations, "val": validation})
        patient = _patient(
            _event("50000001", "image-a", "/synthetic/a.jpg"),
            _event("50000002", "image-b", "/synthetic/b.jpg"),
        )
        with self.assertRaisesRegex(ValueError, "more than one release split"):
            task(patient)

    def test_requires_mimic_cxr_metadata_table(self):
        frame = pl.DataFrame(
            {
                "patient_id": ["1"],
                "event_type": ["other"],
                "timestamp": [None],
            }
        ).lazy()
        with self.assertRaisesRegex(ValueError, "requires the MIMIC-CXR"):
            self._task().pre_filter(frame)

    def test_ignores_unannotated_study(self):
        patient = _patient(_event("59999999", "image-a", "/synthetic/a.jpg"))
        self.assertEqual(self._task()(patient), [])


class TestPriorExamPreferenceIntegration(unittest.TestCase):
    def test_mimic_cxr_set_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source"
            source.mkdir()
            annotations = temp_path / "train.csv"
            _write_annotations(annotations)
            pandas_rows = [
                _metadata_row("synthetic-image-a", "PA"),
                _metadata_row("synthetic-image-b", "LATERAL"),
            ]
            pd.DataFrame(pandas_rows).to_csv(
                source / "mimic-cxr-2.0.0-metadata.csv.gz",
                index=False,
                compression="gzip",
            )

            dataset = MIMIC4CXRDataset(
                root=str(source),
                tables=["metadata"],
                cache_dir=str(temp_path / "cache"),
                num_workers=1,
            )
            samples = dataset.set_task(
                PriorExamPreferenceTask({"train": annotations}), num_workers=1
            )

            self.assertEqual(len(samples), 3)
            self.assertEqual(samples.input_schema, {"text": "text"})
            self.assertEqual(samples.output_schema, {"needs_edit": "binary"})
            self.assertEqual(samples[0]["patient_id"], "10000001")
            self.assertEqual(samples[0]["study_id"], "50000001")
            self.assertEqual(len(samples[0]["image_paths"]), 2)

            model_path = temp_path / "tiny-bert"
            model_path.mkdir()
            _write_tiny_bert(model_path)
            model = TransformersModel(
                samples,
                model_name=str(model_path),
                max_length=16,
            )
            output = model(
                text=[samples[index]["text"] for index in range(2)],
                needs_edit=torch.tensor([[0.0], [1.0]]),
            )
            self.assertEqual(output["y_prob"].shape, (2, 1))
            output["loss"].backward()
            self.assertIsNotNone(model.fc.weight.grad)


if __name__ == "__main__":
    unittest.main()
