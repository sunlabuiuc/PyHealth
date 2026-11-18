"""Tests for the COVID-BLUES lung ultrasound dataset loader.

All frames are synthetic noise images: the CC BY-NC-ND licence rules out
committing frames derived from the real videos.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

from pyhealth.datasets import COVIDBLUESDataset
from pyhealth.datasets import covid_blues as covid_blues_module

# patient id -> PCR result
PATIENTS = {1: 1, 2: 0, 3: 1, 4: 0}
# video stem -> number of frames
VIDEOS = {
    "patient_1_L1": 4,
    "patient_1_R2_2": 2,
    "patient_2_L1": 4,
    "patient_3_R3": 2,
    "patient_4_L2": 2,
    "patient_99_L1": 1,  # no clinical row: skipped
}


def write_frame(path: Path, seed: int) -> None:
    """Write an 8x8 synthetic grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (8, 8), dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def build_covid_blues_root(base: Path) -> tuple[Path, Path]:
    """Write a synthetic COVID-BLUES clone and a frames folder under ``base``."""
    root = base / "COVID-BLUES"
    (root / "lus_videos").mkdir(parents=True)
    pd.DataFrame(
        {"patient_id": list(PATIENTS), "cov_test": list(PATIENTS.values())}
    ).to_csv(root / "clinical_variables.csv", index=False)
    pd.DataFrame(
        {"video_file": ["patient_1_L1"], "Severity Score": [2.0],
         "A-lines": ["yes"], "B-lines": ["yes"], "comments": [""]}
    ).to_csv(root / "severity.csv", index=False)
    frames = base / "frames"
    for seed, (stem, count) in enumerate(VIDEOS.items()):
        (root / "lus_videos" / f"{stem}.mp4").write_bytes(b"")
        for index in range(1, count + 1):
            write_frame(frames / stem / f"{index:06d}.png", seed * 10 + index)
    return root, frames


class TestCOVIDBLUESDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root, self.frames = build_covid_blues_root(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dataset(self, **kwargs):
        kwargs.setdefault("frames_dir", str(self.frames))
        return COVIDBLUESDataset(
            root=str(self.root), cache_dir=str(self.temp_dir / "cache"), **kwargs
        )

    def test_frames_metadata(self):
        dataset = self._dataset()
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(metadata), 14)  # patient_99 skipped
        self.assertEqual(sorted(dataset.unique_patient_ids), ["1", "2", "3", "4"])
        repeat = metadata[metadata["video_id"] == "patient_1_R2_2"].iloc[0]
        self.assertEqual((repeat["blue_point"], repeat["repeat"]), ("R2", 2))
        first = metadata[metadata["video_id"] == "patient_1_L1"]
        self.assertEqual(set(first["severity_score"]), {2.0})
        self.assertEqual(first["frame_index"].tolist(), [1, 2, 3, 4])
        self.assertEqual(set(metadata[metadata["patient_id"] == 2]["covid"]), {0})
        self.assertFalse((self.root / "covid-blues-metadata-pyhealth.csv").exists())

    def test_frame_step(self):
        dataset = self._dataset(frame_step=2)
        metadata = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(metadata), 7)
        self.assertEqual(
            metadata[metadata["video_id"] == "patient_1_L1"]["frame_index"].tolist(),
            [1, 3],
        )
        with self.assertRaises(ValueError):
            self._dataset(frame_step=0)

    def test_needs_frames_or_decoder(self):
        with self.assertRaises(ValueError):
            COVIDBLUESDataset(root=str(self.root), frames_dir=None,
                              cache_dir=str(self.temp_dir / "cache"))

    def test_decode_videos_uses_decoder_and_cache(self):
        def fake_decode(video_path, out_dir):
            write_frame(out_dir / "000000.png", 0)
            return 1

        with mock.patch.object(
            covid_blues_module, "_decode_video", side_effect=fake_decode
        ) as decoder:
            dataset = self._dataset(frames_dir=None, decode_videos=True)
        self.assertEqual(decoder.call_count, len(VIDEOS))
        self.assertTrue(str(dataset.frames_dir).startswith(str(self.temp_dir / "cache")))
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 5)

    def test_decoder_missing_raises_clear_error(self):
        with mock.patch.dict("sys.modules", {"av": None, "cv2": None}), \
                self.assertRaisesRegex(ImportError, "av"):
            covid_blues_module._decode_video(
                self.root / "lus_videos" / "patient_1_L1.mp4", self.temp_dir / "out"
            )

    def test_default_task(self):
        self.assertEqual(
            self._dataset().default_task.task_name, "COVIDBLUESFrameClassification"
        )


if __name__ == "__main__":
    unittest.main()
