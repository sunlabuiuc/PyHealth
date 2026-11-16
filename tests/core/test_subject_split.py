"""SubjectSplit, split_by_subject, and set_task fitting on training subjects."""

import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import (
    BaseDataset,
    SubjectSplit,
    create_sample_dataset,
    split_by_subject,
)
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.tasks import BaseTask

CONFIG = """version: "1.0"
tables:
  events:
    file_path: "events.csv"
    patient_id: "patient_id"
    timestamp: null
    attributes:
      - "value"
"""


class MeanCenter(FeatureProcessor):
    """Subtracts the mean it learned while fitting."""

    learns_statistics = True

    def __init__(self):
        self.mean = None

    def fit(self, samples, field):
        values = [float(s[field][0]) for s in samples]
        self.mean = sum(values) / len(values)

    def process(self, value):
        return torch.tensor([float(value[0]) - self.mean])


class _ValueTask(BaseTask):
    task_name = "subject_split_task"
    input_schema = {"value": MeanCenter}
    output_schema = {"label": "binary"}

    def __call__(self, patient):
        # Both classes occur in the training subjects, so the binary
        # label processor can fit without seeing the held-out subjects.
        return [
            {
                "patient_id": patient.patient_id,
                "value": [float(event.value)],
                "label": int(patient.patient_id.endswith("2")),
            }
            for event in patient.get_events("events")
        ]


class TestSubjectSplit(unittest.TestCase):
    def test_split_indices_sorts_and_drops_unlisted_subjects(self):
        split = SubjectSplit((["chb02", "chb01"], ["chb04"]))
        parts = split.split_indices(
            {"chb01": [2, 0], "chb02": [1], "chb03": [5], "chb04": [4, 3]}
        )
        self.assertEqual([part.tolist() for part in parts], [[0, 1, 2], [3, 4]])
        self.assertEqual(
            split.to_dict(),
            {"kind": "subject", "subjects": [["chb01", "chb02"], ["chb04"]]},
        )

    def test_overlapping_empty_and_string_parts(self):
        with self.assertRaises(ValueError):
            SubjectSplit((["chb01", "chb02"], ["chb02"]))
        with self.assertRaises(ValueError):
            SubjectSplit((["chb01"], []))
        with self.assertRaises(ValueError):
            SubjectSplit((["chb01"],))
        with self.assertRaises(TypeError):
            SubjectSplit(("chb01", "chb02"))

    def test_missing_subject_warns_and_is_skipped(self):
        split = SubjectSplit((["chb01"], ["chb02", "chb99"]))
        with self.assertLogs("pyhealth.datasets.splitter", level="WARNING") as logs:
            parts = split.split_indices({"chb01": [0], "chb02": [1]})
        self.assertEqual([part.tolist() for part in parts], [[0], [1]])
        self.assertIn("chb99", logs.output[0])

    def test_split_by_subject(self):
        samples = [
            {
                "patient_id": f"chb0{i}",
                "record_id": f"r{i}",
                "feature": i,
                "label": i % 2,
            }
            for i in range(1, 5)
        ]
        dataset = create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})
        train, test = split_by_subject(dataset, (["chb01", "chb02"], ["chb04"]))
        self.assertEqual(
            sorted(train[i]["patient_id"] for i in range(len(train))),
            ["chb01", "chb02"],
        )
        self.assertEqual([test[i]["patient_id"] for i in range(len(test))], ["chb04"])


class TestSubjectSplitSetTask(unittest.TestCase):
    def test_processors_fit_on_training_subjects_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = ["patient_id,value"]
            # Train subjects sit at 10. Held-out subjects sit at 100, which
            # would move the mean if they leaked into fitting.
            for subject, value in (
                ("chb01", 10),
                ("chb02", 10),
                ("chb03", 100),
                ("chb04", 100),
            ):
                rows.append(f"{subject},{value}")
            (root / "events.csv").write_text("\n".join(rows) + "\n")
            (root / "config.yaml").write_text(CONFIG)
            dataset = BaseDataset(
                root=str(root),
                tables=["events"],
                dataset_name="SubjectSplitDataset",
                config_path=str(root / "config.yaml"),
                cache_dir=str(root / "cache"),
                num_workers=1,
            )
            split = SubjectSplit((["chb01", "chb02"], ["chb03"]))
            train, val = dataset.set_task(_ValueTask(), split=split)
            self.assertEqual(
                {train[i]["patient_id"] for i in range(len(train))}, {"chb01", "chb02"}
            )
            self.assertEqual({val[i]["patient_id"] for i in range(len(val))}, {"chb03"})
            self.assertAlmostEqual(train.input_processors["value"].mean, 10.0)
            self.assertAlmostEqual(val.input_processors["value"].mean, 10.0)
            self.assertTrue(torch.allclose(train[0]["value"], torch.tensor([0.0])))
            # chb04 was in neither list, so it is in neither part.
            held_out = {train[i]["patient_id"] for i in range(len(train))}
            held_out.update(val[i]["patient_id"] for i in range(len(val)))
            self.assertNotIn("chb04", held_out)
            self.assertEqual(train.fit_split["kind"], "subject")


if __name__ == "__main__":
    unittest.main()
