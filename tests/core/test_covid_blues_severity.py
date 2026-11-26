"""Tests for the COVID-BLUES severity classification task."""

import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from pyhealth.datasets import COVIDBLUESDataset, get_dataloader
from pyhealth.tasks import COVIDBLUESSeverityClassification

try:
    from tests.core.test_covid_blues_dataset import build_covid_blues_root
except ImportError:  # pragma: no cover - run from tests/core
    from test_covid_blues_dataset import build_covid_blues_root


class TestCOVIDBLUESSeverityClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root, frames = build_covid_blues_root(cls.temp_dir)
        # Scores for three videos; the others have none and are skipped.
        pd.DataFrame(
            {
                "video_file": ["patient_1_L1", "patient_2_L1", "patient_3_R3"],
                "Severity Score": [2.0, 0.0, 3.0],
                "A-lines": ["yes", "yes", "no"],
                "B-lines": ["yes", "no", "yes"],
                "comments": ["", "", ""],
            }
        ).to_csv(root / "severity.csv", index=False)
        cls.dataset = COVIDBLUESDataset(
            root=str(root), frames_dir=str(frames), cache_dir=str(cls.temp_dir / "c")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_frames_get_their_video_score(self):
        samples = self.dataset.set_task(
            COVIDBLUESSeverityClassification(), num_workers=1
        )
        try:
            # patient_1_L1 (4 frames), patient_2_L1 (4), patient_3_R3 (2).
            self.assertEqual(len(samples), 10)
            self.assertEqual(samples.output_processors["severity"].size(), 4)
            by_video = {s["video_id"]: int(s["severity"]) for s in samples}
            self.assertEqual(
                by_video,
                {"patient_1_L1": 2, "patient_2_L1": 0, "patient_3_R3": 3},
            )
            self.assertEqual(
                Counter(int(s["severity"]) for s in samples),
                Counter({2: 4, 0: 4, 3: 2}),
            )
            batch = next(iter(get_dataloader(samples, batch_size=4)))
            self.assertEqual(tuple(batch["image"].shape), (4, 1, 224, 224))
        finally:
            samples.close()


if __name__ == "__main__":
    unittest.main()
