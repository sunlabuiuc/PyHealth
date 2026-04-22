"""Tests for Hallmarks of Cancer dataset and task (synthetic CSV only)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from pyhealth.datasets import HallmarksOfCancerDataset
from pyhealth.tasks.hallmarks_of_cancer_classification import (
    HallmarksOfCancerSentenceClassification,
    _parse_labels_raw,
)


class TestParseLabels(unittest.TestCase):
    def test_pipe_style(self):
        self.assertEqual(_parse_labels_raw("none"), ["none"])
        self.assertEqual(
            _parse_labels_raw("a##b"),
            ["a", "b"],
        )

    def test_json_style(self):
        self.assertEqual(_parse_labels_raw('["none"]'), ["none"])
        self.assertEqual(
            _parse_labels_raw('["activating invasion and metastasis"]'),
            ["activating invasion and metastasis"],
        )


class TestHallmarksOfCancerTask(unittest.TestCase):
    def test_pre_filter_column(self):
        task = HallmarksOfCancerSentenceClassification(split="train")
        lf = pl.DataFrame(
            {
                "patient_id": ["1", "2"],
                "event_type": ["hoc", "hoc"],
                "timestamp": [None, None],
                "hoc/split": ["train", "validation"],
                "hoc/text": ["hello", "world"],
                "hoc/labels": ["none", "none"],
                "hoc/document_id": ["d1", "d2"],
            }
        ).lazy()
        out = task.pre_filter(lf).collect()
        self.assertEqual(len(out), 1)
        self.assertEqual(out["patient_id"].to_list()[0], "1")


class TestHallmarksOfCancerDataset(unittest.TestCase):
    def _write_minimal_csv(self, directory: Path) -> None:
        lines = [
            "sentence_id,document_id,text,labels,split",
            's1,d1,"hello",none,train',
            's2,d2,"world","a##b",train',
            's3,d3,"val row",none,validation',
        ]
        (directory / "hallmarks_of_cancer.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def test_load_and_task_produces_samples(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=tmp,
        ):
            root = Path(tmp) / "hoc_root"
            root.mkdir()
            self._write_minimal_csv(root)
            ds = HallmarksOfCancerDataset(
                root=str(root),
                cache_dir=Path(tmp) / "hoc_cache",
                num_workers=1,
                dev=True,
            )
            task = HallmarksOfCancerSentenceClassification(split="train")
            samples = ds.set_task(task)
            self.assertGreaterEqual(len(samples), 1)
            row0 = samples[0]
            self.assertIn("text", row0)
            self.assertIn("labels", row0)
            self.assertIn("source_text", row0)
            self.assertIn("target_text", row0)
            self.assertEqual(
                row0["target_text"],
                HallmarksOfCancerSentenceClassification.labels_to_target_text(
                    row0["labels"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
