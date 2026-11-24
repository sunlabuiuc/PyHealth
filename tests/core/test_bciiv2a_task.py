"""Tests for the BCI Competition IV 2a motor imagery task."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from pyhealth.datasets import BCICompetitionIV2aDataset, split_by_strat_fold
from pyhealth.tasks import BCIIV2aMotorImageryClassification
from pyhealth.tasks.bciiv2a_motor_imagery import load_bciiv2a_trials

FS = 50


def write_bnci_session(path: Path, n_trials: int = 4) -> None:
    """One EOG run (no trials) and one run whose signal encodes time."""
    runs = np.empty((1, 2), dtype=object)
    empty = np.zeros((0, 1))
    runs[0, 0] = {
        "X": np.zeros((100, 25)),
        "trial": empty,
        "y": empty,
        "fs": FS,
        "artifacts": empty,
    }
    n = n_trials * 8 * FS + FS
    x = np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 25))
    x[:, 22:] = -1.0  # EOG channels
    artifacts = np.zeros(n_trials)
    artifacts[1] = 1
    runs[0, 1] = {
        "X": x,
        "trial": (1 + np.arange(n_trials) * 8 * FS).reshape(-1, 1).astype(float),
        "y": (np.arange(n_trials) % 4 + 1).reshape(-1, 1).astype(float),
        "fs": FS,
        "artifacts": artifacts.reshape(-1, 1),
    }
    savemat(str(path), {"data": runs})


class TestTrialReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path = cls.tmp / "A01T.mat"
        write_bnci_session(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_default_window_starts_at_cue(self):
        trials = load_bciiv2a_trials(str(self.path))
        self.assertEqual(len(trials), 3)  # artifact trial dropped
        first = trials[0]
        self.assertEqual(first["signal"].shape, (22, 4 * FS))
        # trial starts at sample index 0; the cue is 2 s later
        self.assertEqual(first["signal"][0, 0], 2 * FS)
        self.assertEqual([t["label"] for t in trials], [0, 2, 3])

    def test_options(self):
        trials = load_bciiv2a_trials(
            str(self.path), tmin=0, tmax=4, drop_artifacts=False, include_eog=True
        )
        self.assertEqual(len(trials), 4)
        self.assertEqual(trials[0]["signal"].shape, (25, 4 * FS))
        self.assertEqual(trials[0]["signal"][0, 0], 0)
        self.assertTrue(trials[1]["artifact"])
        with self.assertRaises(ValueError):
            load_bciiv2a_trials(str(self.path), tmin=3, tmax=2)

    def test_bandpass(self):
        raw = load_bciiv2a_trials(str(self.path))
        trials = load_bciiv2a_trials(str(self.path), l_freq=4.0, h_freq=20.0)
        self.assertEqual(trials[0]["signal"].shape, (22, 4 * FS))
        # The fixture signal is a slow ramp (DC + trend); a 4 Hz high-pass
        # should remove almost all of it inside the trial windows.
        raw_level = np.abs(raw[0]["signal"]).mean()
        filtered_level = np.abs(trials[0]["signal"]).mean()
        self.assertLess(filtered_level, 0.01 * raw_level)

    def test_non_finite_trial_skipped_with_warning(self):
        path = self.tmp / "A02T.mat"
        write_bnci_session(path)
        from scipy.io import loadmat

        mat = loadmat(str(path))
        run = mat["data"][0, 1]
        x = run["X"][0, 0]
        x[2 * 8 * FS + 2 * FS + 5, 0] = np.nan  # inside trial 2's window
        savemat(str(path), {"data": mat["data"]})
        with self.assertLogs(
            "pyhealth.tasks.bciiv2a_motor_imagery", level="WARNING"
        ) as logs:
            trials = load_bciiv2a_trials(str(path), drop_artifacts=False)
        self.assertEqual([t["trial"] for t in trials], [0, 1, 3])
        self.assertIn("non-finite", logs.output[0])


class TestBCIIV2aTaskEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "bnci"
        root.mkdir()
        for session in ("T", "E"):
            write_bnci_session(root / f"A01{session}.mat")
        cls.dataset = BCICompetitionIV2aDataset(
            root=str(root),
            subjects=[1],
            metadata_cache_dir=str(cls.tmp / "meta"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_samples_and_official_split(self):
        samples = self.dataset.set_task(
            BCIIV2aMotorImageryClassification(), num_workers=1
        )
        self.assertEqual(len(samples), 6)
        self.assertEqual(samples.output_processors["label"].size(), 4)
        sample = samples[0]
        self.assertEqual(tuple(sample["signal"].shape), (22, 4 * FS))
        self.assertIn(sample["split"], {"train", "test"})
        train, _, test = split_by_strat_fold(
            samples,
            train_folds=(0,),
            val_folds=(),
            test_folds=(1,),
            fold_field="session_fold",
        )
        self.assertEqual((len(train), len(test)), (3, 3))
        self.assertEqual({train[i]["session"] for i in range(len(train))}, {"T"})


if __name__ == "__main__":
    unittest.main()
