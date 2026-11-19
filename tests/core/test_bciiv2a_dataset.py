"""Tests for BCICompetitionIV2aDataset on tiny synthetic BNCI files."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import savemat

from pyhealth.datasets import BCICompetitionIV2aDataset
from pyhealth.datasets.bciiv2a import bciiv2a_filename

FS = 50  # low rate keeps the fixtures small; the reader uses the file's fs


def write_bnci_session(
    path: Path, n_trials: int = 4, artifact_first: bool = False
) -> None:
    """Write a BNCI-style ``data`` cell: one EOG run and two trial runs."""
    runs = np.empty((1, 3), dtype=object)
    empty = np.zeros((0, 1))
    runs[0, 0] = {
        "X": np.zeros((100, 25)),
        "trial": empty,
        "y": empty,
        "fs": FS,
        "artifacts": empty,
    }
    rng = np.random.default_rng(0)
    for k in (1, 2):
        n = n_trials * 8 * FS + FS
        onsets = 1 + np.arange(n_trials) * 8 * FS
        artifacts = np.zeros(n_trials)
        if artifact_first and k == 1:
            artifacts[0] = 1
        runs[0, k] = {
            "X": rng.normal(size=(n, 25)),
            "trial": onsets.reshape(-1, 1).astype(float),
            "y": (np.arange(n_trials) % 4 + 1).reshape(-1, 1).astype(float),
            "fs": FS,
            "artifacts": artifacts.reshape(-1, 1),
        }
    savemat(str(path), {"data": runs})


class TestBCICompetitionIV2aDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "bnci"
        self.root.mkdir()
        for subject in (1, 2):
            for session in ("T", "E"):
                write_bnci_session(self.root / bciiv2a_filename(subject, session))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        return BCICompetitionIV2aDataset(
            root=str(self.root),
            metadata_cache_dir=str(self.tmp / "meta"),
            cache_dir=str(self.tmp / "cache"),
            **kwargs,
        )

    def test_rows_carry_official_split(self):
        ds = self._dataset(subjects=[1, 2])
        self.assertEqual(sorted(ds.unique_patient_ids), ["A01", "A02"])
        events = ds.get_patient("A01").get_events(event_type="sessions")
        by_session = {str(e.session): e for e in events}
        self.assertEqual(str(by_session["T"].split), "train")
        self.assertEqual(str(by_session["E"].split), "test")
        self.assertEqual(str(by_session["E"].record_id), "A01E")
        self.assertEqual(list(self.root.glob("*.csv")), [])

    def test_session_selection(self):
        ds = self._dataset(subjects=[2], sessions=["e"])
        events = ds.get_patient("A02").get_events(event_type="sessions")
        self.assertEqual([str(e.session) for e in events], ["E"])

    def test_missing_file_and_bad_arguments(self):
        with self.assertRaises(FileNotFoundError):
            self._dataset(subjects=[3])
        with self.assertRaises(ValueError):
            self._dataset(subjects=[10])
        with self.assertRaises(ValueError):
            self._dataset(sessions=["X"])

    def test_download_fetches_missing_files(self):
        target = self.tmp / "empty"
        calls = []

        def fake_download(url, dest):
            calls.append((url, dest))
            write_bnci_session(Path(dest))

        with mock.patch("pyhealth.utils.download_file", side_effect=fake_download):
            ds = BCICompetitionIV2aDataset(
                root=str(target),
                subjects=[3],
                sessions=["E"],
                download=True,
                metadata_cache_dir=str(self.tmp / "meta"),
                cache_dir=str(self.tmp / "cache"),
            )
        self.assertEqual(
            calls,
            [
                (
                    "https://bnci-horizon-2020.eu/database/data-sets/001-2014/A03E.mat",
                    str(target / "A03E.mat"),
                )
            ],
        )
        events = ds.get_patient("A03").get_events(event_type="sessions")
        self.assertEqual([str(e.record_id) for e in events], ["A03E"])
        samples = ds.set_task(num_workers=1)
        self.assertEqual(len(samples), 8)  # 2 runs x 4 trials, no artifacts
        samples.close()

        # Files already present are not downloaded again.
        with mock.patch("pyhealth.utils.download_file") as again:
            self._dataset(subjects=[1], download=True)
        again.assert_not_called()

    def test_default_task(self):
        ds = self._dataset(subjects=[1])
        self.assertEqual(ds.default_task.tmin, 2.0)
        self.assertEqual(ds.default_task.tmax, 6.0)


if __name__ == "__main__":
    unittest.main()
