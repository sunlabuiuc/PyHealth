"""Tests for the COVID-BLUES lung ultrasound dataset loader.

All frames are synthetic noise images: the CC BY-NC-ND licence rules out
committing frames derived from the real videos.
"""

import importlib.util
import shutil
import tempfile
import types
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

    def test_video_without_frames_is_not_cached(self):
        def fake_decode(video_path, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            if video_path.stem == "patient_2_L1":
                return 0  # e.g. cv2 could not open the file
            write_frame(out_dir / "000000.png", 0)
            return 1

        with mock.patch.object(
            covid_blues_module, "_decode_video", side_effect=fake_decode
        ), self.assertLogs(covid_blues_module.logger, level="WARNING") as logs:
            dataset = self._dataset(frames_dir=None, decode_videos=True)
        self.assertIn("patient_2_L1", "\n".join(logs.output))
        self.assertFalse((dataset.frames_dir / "patient_2_L1").exists())
        self.assertEqual(list(dataset.frames_dir.glob(".*.partial")), [])
        self.assertNotIn("2", dataset.unique_patient_ids)

        # The failed video is retried on the next run.
        with mock.patch.object(
            covid_blues_module, "_decode_video", return_value=0
        ) as decoder:
            dataset.extract_frames()
        self.assertEqual(decoder.call_count, 1)

    def test_video_that_raises_is_skipped_and_retried(self):
        def fake_decode(video_path, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            write_frame(out_dir / "000000.png", 0)
            if video_path.stem == "patient_2_L1":
                # e.g. PyAV raising InvalidDataError on a corrupt .mp4
                raise ValueError("Invalid data found when processing input")
            return 1

        with mock.patch.object(
            covid_blues_module, "_decode_video", side_effect=fake_decode
        ) as decoder, self.assertLogs(
            covid_blues_module.logger, level="WARNING"
        ) as logs:
            dataset = self._dataset(frames_dir=None, decode_videos=True)
        self.assertEqual(decoder.call_count, len(VIDEOS))
        self.assertIn("patient_2_L1", "\n".join(logs.output))
        self.assertFalse((dataset.frames_dir / "patient_2_L1").exists())
        self.assertEqual(list(dataset.frames_dir.glob(".*.partial")), [])
        for stem in VIDEOS:
            if stem != "patient_2_L1":
                self.assertTrue((dataset.frames_dir / stem).is_dir(), stem)
        self.assertNotIn("2", dataset.unique_patient_ids)
        self.assertIn("1", dataset.unique_patient_ids)

        # Only the failed video is retried on the next run.
        with mock.patch.object(
            covid_blues_module, "_decode_video", side_effect=fake_decode
        ) as decoder, self.assertLogs(covid_blues_module.logger, level="WARNING"):
            dataset.extract_frames()
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(decoder.call_args.args[0].stem, "patient_2_L1")

    def test_missing_decoder_is_not_swallowed(self):
        with mock.patch.object(
            covid_blues_module, "_decode_video", side_effect=ImportError("av")
        ), self.assertRaises(ImportError):
            self._dataset(frames_dir=None, decode_videos=True)

    def test_severity_tolerates_mp4_suffix_and_warns_on_no_match(self):
        table = pd.read_csv(self.root / "severity.csv")
        table["video_file"] = ["patient_1_L1.mp4"]
        table.to_csv(self.root / "severity.csv", index=False)
        metadata = pd.read_csv(self._dataset().metadata_path)
        first = metadata[metadata["video_id"] == "patient_1_L1"]
        self.assertEqual(set(first["severity_score"]), {2.0})

        table["video_file"] = ["video_1"]
        table.to_csv(self.root / "severity.csv", index=False)
        with self.assertLogs(covid_blues_module.logger, level="WARNING"):
            self._dataset(refresh_metadata=True)

    def test_decode_video_with_pyav_module(self):
        class FakeFrame:
            def __init__(self, seed):
                self.seed = seed

            def to_image(self):
                pixels = np.full((4, 4, 3), self.seed, dtype=np.uint8)
                return Image.fromarray(pixels)

        class FakeContainer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def decode(self, video=0):
                return iter([FakeFrame(1), FakeFrame(2), FakeFrame(3)])

        fake_av = types.SimpleNamespace(open=lambda path: FakeContainer())
        out = self.temp_dir / "av_out"
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            count = covid_blues_module._decode_video(Path("v.mp4"), out)
        self.assertEqual(count, 3)
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            ["000000.png", "000001.png", "000002.png"],
        )

    def test_decode_video_with_opencv_module(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
        frames[0][..., 0] = 255  # BGR blue

        class FakeCapture:
            def __init__(self, path):
                self.frames = list(frames)
                self.released = False

            def read(self):
                if self.frames:
                    return True, self.frames.pop(0)
                return False, None

            def release(self):
                self.released = True

        fake_cv2 = types.SimpleNamespace(
            VideoCapture=FakeCapture,
            COLOR_BGR2RGB=4,
            cvtColor=lambda frame, code: frame[..., ::-1],
        )
        out = self.temp_dir / "cv2_out"
        with mock.patch.dict("sys.modules", {"av": None, "cv2": fake_cv2}):
            count = covid_blues_module._decode_video(Path("v.mp4"), out)
        self.assertEqual(count, 2)
        first = np.asarray(Image.open(out / "000000.png"))
        self.assertEqual(first[0, 0].tolist(), [0, 0, 255])  # now RGB blue

        # A video OpenCV cannot read yields 0 frames.
        frames.clear()
        with mock.patch.dict("sys.modules", {"av": None, "cv2": fake_cv2}):
            self.assertEqual(
                covid_blues_module._decode_video(Path("v.mp4"), self.temp_dir / "e"),
                0,
            )

    # PyAV is optional and not in the CI test environment, so CI may skip
    # this test; the mocked-PyAV test above still covers the code path.
    @unittest.skipUnless(importlib.util.find_spec("av"), "needs PyAV")
    def test_decode_real_video_with_pyav(self):
        import av

        video = self.temp_dir / "tiny.mp4"
        with av.open(str(video), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=5)
            stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
            for i in range(3):
                pixels = np.full((16, 16, 3), i * 40, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        out = self.temp_dir / "real_out"
        self.assertEqual(covid_blues_module._decode_video(video, out), 3)
        self.assertEqual(len(list(out.glob("*.png"))), 3)

    def test_default_task(self):
        self.assertEqual(
            self._dataset().default_task.task_name, "COVIDBLUESFrameClassification"
        )


if __name__ == "__main__":
    unittest.main()
