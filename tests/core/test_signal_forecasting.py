"""Tests for PTBXLForecasting and SleepEDFForecasting (synthetic signals)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import APN
from pyhealth.tasks import (
    PTBXL_LEADS,
    SLEEPEDF_CHANNELS,
    PTBXLForecasting,
    SleepEDFForecasting,
)


class _Patient:
    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self.events = events

    def get_events(self, event_type=None):
        return self.events


def _ecg(seconds=10, rate=100, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * rate) / rate
    leads = [np.sin(2 * np.pi * (1 + i / 10) * t) * (i + 1) + 5 * i for i in range(12)]
    return (np.stack(leads) + 0.01 * rng.normal(size=(12, len(t)))).astype(np.float32)


def _ptbxl_patient(signal):
    event = SimpleNamespace(
        signal_file="records100/00000/00001_lr", sampling_rate="100",
        record_id="1", strat_fold="10",
    )
    return _Patient("p1", [event])


class TestPTBXLForecasting(unittest.TestCase):
    def run_task(self, task, signal):
        with mock.patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record", return_value=signal
        ):
            return task(_ptbxl_patient(signal))

    def test_shapes_times_and_fields(self):
        samples = self.run_task(PTBXLForecasting(context_length=96, horizon=48), _ecg())
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s["x"].shape, (96, 12))
        self.assertEqual(s["y"].shape, (48, 12))
        np.testing.assert_allclose(s["x_time"], np.arange(96) / 100, rtol=1e-6)
        np.testing.assert_allclose(s["y_time"], np.arange(96, 144) / 100, rtol=1e-6)
        self.assertEqual((s["record_id"], s["strat_fold"], s["start"]), ("1", 10, 0))

    def test_values_are_record_z_scores_and_targets_follow_context(self):
        signal = _ecg()
        s = self.run_task(PTBXLForecasting(context_length=48, horizon=24), signal)[0]
        z = (signal - signal.mean(1, keepdims=True)) / signal.std(1, keepdims=True)
        np.testing.assert_allclose(s["x"], z[:, :48].T, atol=1e-5)
        np.testing.assert_allclose(s["y"], z[:, 48:72].T, atol=1e-5)

    def test_lead_selection_and_windows(self):
        signal = _ecg()
        task = PTBXLForecasting(
            context_length=48, horizon=24, leads=["II", "V5"], num_windows=3
        )
        samples = self.run_task(task, signal)
        self.assertEqual([s["start"] for s in samples], [0, 464, 928])
        self.assertEqual(samples[0]["x"].shape, (48, 2))
        z = (signal - signal.mean(1, keepdims=True)) / signal.std(1, keepdims=True)
        lead_ii = PTBXL_LEADS.index("II")
        np.testing.assert_allclose(samples[2]["y"][:, 0], z[lead_ii, 976:1000], atol=1e-5)

    def test_drop_rate_masks_context_only_and_is_reproducible(self):
        task = PTBXLForecasting(drop_rate=0.3, seed=1)
        a = self.run_task(task, _ecg())[0]
        b = self.run_task(PTBXLForecasting(drop_rate=0.3, seed=1), _ecg())[0]
        share = np.isnan(a["x"]).mean()
        self.assertTrue(0.2 < share < 0.4)
        self.assertTrue(np.isfinite(a["y"]).all())
        np.testing.assert_array_equal(np.isnan(a["x"]), np.isnan(b["x"]))

    def test_short_record_and_flat_lead(self):
        signal = _ecg(seconds=1)
        self.assertEqual(self.run_task(PTBXLForecasting(96, 48), signal), [])
        signal = _ecg()
        signal[0] = 3.0
        s = self.run_task(PTBXLForecasting(48, 24), signal)[0]
        np.testing.assert_array_equal(s["x"][:, 0], 0.0)

    def test_invalid_arguments(self):
        for kwargs in (
            {"context_length": 1}, {"horizon": 0}, {"num_windows": 0},
            {"drop_rate": 1.0}, {"leads": ["V7"]}, {"leads": []},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                PTBXLForecasting(**kwargs)

    def test_apn_trains_on_the_samples(self):
        with mock.patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record", return_value=_ecg()
        ):
            task = PTBXLForecasting(context_length=48, horizon=24, num_windows=4)
            samples = task(_ptbxl_patient(None))
        dataset = create_sample_dataset(
            [{k: s[k] for k in ("patient_id", "x", "x_time", "y_time", "y")}
             for s in samples],
            PTBXLForecasting.input_schema, PTBXLForecasting.output_schema,
        )
        model = APN(dataset, hidden_dim=8, num_patches=4, time_dim=4)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        out = model(**batch)
        self.assertEqual(out["forecast"].shape, (4, 24, 12))
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]))


class _FakeRaw:
    def __init__(self, data, sfreq=100.0, ch_names=(*SLEEPEDF_CHANNELS, "EOG horizontal")):
        self.data = data
        self.info = {"sfreq": sfreq}
        self.ch_names = list(ch_names)
        self.n_times = data.shape[1]
        self.first_time = 0.0
        self.annotations = None

    def set_annotations(self, annotations, emit_warning=True):
        self.annotations = annotations

    def get_data(self, picks, start=0, stop=None):
        rows = [self.ch_names.index(p) for p in picks]
        return self.data[rows, start:stop]


def _night(label_file="hyp.edf"):
    return SimpleNamespace(signal_file="psg.edf", label_file=label_file, night="1")


class TestSleepEDFForecasting(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.data = rng.normal(size=(3, 100 * 600)) * np.array([[1e-5], [2e-5], [1e-4]])
        # Wake for 100 s, sleep from 100 s to 400 s, wake again.
        self.ann = SimpleNamespace(
            onset=[0.0, 100.0, 250.0, 400.0],
            duration=[100.0, 150.0, 150.0, 200.0],
            description=["Sleep stage W", "Sleep stage 2", "Sleep stage R",
                         "Sleep stage W"],
        )

    def run_task(self, task, events, raw=None):
        raw = raw or _FakeRaw(self.data)
        with mock.patch("mne.io.read_raw_edf", return_value=raw), mock.patch(
            "mne.read_annotations", return_value=self.ann
        ):
            return task(_Patient("7", events))

    def test_windows_lie_in_the_sleep_period(self):
        task = SleepEDFForecasting(context_length=96, horizon=48, num_windows=5)
        samples = self.run_task(task, [_night()])
        starts = [s["start"] for s in samples]
        self.assertEqual(starts[0], 10_000)
        self.assertEqual(starts[-1], 40_000 - 144)
        self.assertEqual(len(starts), 5)
        s = samples[0]
        self.assertEqual(s["x"].shape, (96, 2))
        self.assertEqual((s["record_id"], s["night"]), ("7-1", "1"))
        part = self.data[:2, 10_000:40_000]
        z = (part - part.mean(1, keepdims=True)) / part.std(1, keepdims=True)
        np.testing.assert_allclose(s["x"], z[:, :96].T, atol=1e-5)

    def test_without_hypnogram_uses_whole_night(self):
        samples = self.run_task(
            SleepEDFForecasting(num_windows=2), [_night(label_file=None)]
        )
        self.assertEqual([s["start"] for s in samples], [0, 60_000 - 144])

    def test_night_without_scored_sleep_is_skipped(self):
        self.ann.description = ["Sleep stage W", "Sleep stage ?", "Movement time",
                                "Sleep stage W"]
        self.assertEqual(self.run_task(SleepEDFForecasting(), [_night()]), [])

    def test_missing_channel_raises(self):
        raw = _FakeRaw(self.data, ch_names=("EEG Fpz-Cz", "EOG horizontal", "Resp"))
        with self.assertRaisesRegex(ValueError, "EEG Pz-Oz"):
            self.run_task(SleepEDFForecasting(), [_night()], raw=raw)
        with self.assertRaises(ValueError):
            SleepEDFForecasting(channels=[])


if __name__ == "__main__":
    unittest.main()
