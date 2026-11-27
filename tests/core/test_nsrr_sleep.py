"""Tests for NSRRSleepDataset and SleepStagingCardioRespiratory.

All data are synthetic: tiny EDF files and profusion XML files written to a
temporary directory. No real NSRR data is used.
"""

import os
import tempfile
import unittest

import numpy as np

from pyhealth.datasets import NSRRSleepDataset
from pyhealth.tasks import (
    SleepStagingCardioRespiratory,
    SleepStagingIBIMESA,
    SleepStagingIBISHHS,
)
from pyhealth.tasks.sleep_staging_nsrr import (
    POINTS_PER_EPOCH,
    _read_edf_channels,
    _read_stage_codes,
    _to_epochs,
)

# Stage codes of the synthetic night; 9 marks an unscored epoch.
STAGES = [0, 0, 1, 2, 2, 3, 4, 5, 5, 9]
N_EPOCHS = len(STAGES)


def write_edf(path, channels, n_seconds):
    """Writes a minimal EDF file. ``channels`` is [(label, rate_hz), ...]."""
    rng = np.random.default_rng(0)
    ns = len(channels)

    def field(value, width):
        return str(value).ljust(width)[:width].encode("ascii")

    header = (
        field("0", 8)
        + field("synthetic", 80)
        + field("synthetic", 80)
        + field("01.01.00", 8)
        + field("00.00.00", 8)
        + field(256 * (ns + 1), 8)
        + field("", 44)
        + field(n_seconds, 8)
        + field(1, 8)
        + field(ns, 4)
    )
    signal_header = b"".join(
        [field(label, 16) for label, _ in channels]
        + [field("", 80) for _ in channels]
        + [field("uV", 8) for _ in channels]
        + [field(-100, 8) for _ in channels]
        + [field(100, 8) for _ in channels]
        + [field(-32768, 8) for _ in channels]
        + [field(32767, 8) for _ in channels]
        + [field("", 80) for _ in channels]
        + [field(rate, 8) for _, rate in channels]
        + [field("", 32) for _ in channels]
    )
    columns = [
        (rng.normal(0, 20, (n_seconds, rate)) * 327.67).astype("<i2")
        for _, rate in channels
    ]
    with open(path, "wb") as f:
        f.write(header + signal_header)
        f.write(np.concatenate(columns, axis=1).astype("<i2").tobytes())


def write_xml(path, stages):
    body = "".join(f"<SleepStage>{s}</SleepStage>" for s in stages)
    with open(path, "w") as f:
        f.write(
            "<?xml version='1.0'?><CMPStudyConfig><EpochLength>30</EpochLength>"
            f"<SleepStages>{body}</SleepStages></CMPStudyConfig>"
        )


def make_recording(folder, stem, channels, stages=STAGES):
    write_edf(os.path.join(folder, f"{stem}.edf"), channels, 30 * len(stages))
    write_xml(os.path.join(folder, f"{stem}-profusion.xml"), stages)


def periodic_signal(n_seconds, rate, period_samples):
    """Deterministic pulsatile waveform with a beat every ``period_samples``."""
    t = np.arange(n_seconds * rate)
    wave = np.sin(2 * np.pi * t / period_samples)
    wave += 0.5 * np.sin(4 * np.pi * t / period_samples)
    wave[::period_samples] += 2.0
    return (np.clip(wave, -3.0, 3.0) * 10000).astype("<i2").reshape(
        n_seconds, rate
    )


def write_edf_array(path, channels, signals):
    """Writes an EDF from precomputed ``{label: (n_seconds, rate)}`` arrays."""
    ns = len(channels)
    n_seconds = next(iter(signals.values())).shape[0]

    def field(value, width):
        return str(value).ljust(width)[:width].encode("ascii")

    header = (
        field("0", 8)
        + field("synthetic", 80)
        + field("synthetic", 80)
        + field("01.01.00", 8)
        + field("00.00.00", 8)
        + field(256 * (ns + 1), 8)
        + field("", 44)
        + field(n_seconds, 8)
        + field(1, 8)
        + field(ns, 4)
    )
    signal_header = b"".join(
        [field(label, 16) for label, _ in channels]
        + [field("", 80) for _ in channels]
        + [field("uV", 8) for _ in channels]
        + [field(-100, 8) for _ in channels]
        + [field(100, 8) for _ in channels]
        + [field(-32768, 8) for _ in channels]
        + [field(32767, 8) for _ in channels]
        + [field("", 80) for _ in channels]
        + [field(rate, 8) for _, rate in channels]
        + [field("", 32) for _ in channels]
    )
    columns = [np.asarray(signals[label], dtype="<i2") for label, _ in channels]
    with open(path, "wb") as f:
        f.write(header + signal_header)
        f.write(np.concatenate(columns, axis=1).astype("<i2").tobytes())


SHHS_CHANNELS = [("EEG", 125), ("ECG", 125), ("THOR RES", 10), ("ABDO RES", 10)]
MESA_CHANNELS = [("EKG", 256), ("Thor", 32), ("Abdo", 32), ("Pleth", 256)]


_FIXTURE = None
_TMP = None


def setUpModule():
    """Builds the synthetic cohorts and datasets once for the whole module."""
    global _FIXTURE, _TMP
    _TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    root = _TMP.name
    shhs, mesa = os.path.join(root, "shhs"), os.path.join(root, "mesa")
    edfs = os.path.join(shhs, "edfs")
    os.makedirs(edfs)
    os.makedirs(mesa)
    make_recording(edfs, "shhs1-200001", SHHS_CHANNELS)
    make_recording(edfs, "shhs2-200001", SHHS_CHANNELS)
    make_recording(edfs, "shhs1-200002", SHHS_CHANNELS)
    make_recording(mesa, "mesa-sleep-0001", MESA_CHANNELS)
    cache = os.path.join(root, "cache")
    shhs_ds = NSRRSleepDataset(shhs, cohort="shhs", cache_dir=cache)
    mesa_ds = NSRRSleepDataset(mesa, cohort="mesa", cache_dir=cache)
    # get_patient runs a query on every call, so fetch the patients once.
    _FIXTURE = dict(
        shhs=shhs,
        mesa=mesa,
        shhs_ds=shhs_ds,
        mesa_ds=mesa_ds,
        shhs_ids=sorted(shhs_ds.unique_patient_ids),
        mesa_ids=list(mesa_ds.unique_patient_ids),
        two_visit_patient=shhs_ds.get_patient("200001"),
        shhs_patient=shhs_ds.get_patient("200002"),
        mesa_patient=mesa_ds.get_patient("0001"),
    )


def tearDownModule():
    _TMP.cleanup()


class NSRRTestCase(unittest.TestCase):
    """Exposes the module-level synthetic fixtures as class attributes."""

    @classmethod
    def setUpClass(cls):
        for key, value in _FIXTURE.items():
            setattr(cls, key, value)


class TestNSRRSleepDataset(NSRRTestCase):
    def test_indexes_patients_and_visits(self):
        self.assertEqual(self.shhs_ids, ["200001", "200002"])
        events = self.two_visit_patient.get_events()
        self.assertEqual(sorted(e.visit for e in events), ["shhs1", "shhs2"])
        self.assertTrue(all(os.path.exists(e.signal_file) for e in events))
        self.assertTrue(all(e.label_file.endswith("-profusion.xml") for e in events))

    def test_metadata_written_and_default_task(self):
        dataset = self.shhs_ds
        self.assertTrue(
            os.path.exists(os.path.join(self.shhs, "nsrr-shhs-pyhealth.csv"))
        )
        self.assertIsInstance(dataset.default_task, SleepStagingCardioRespiratory)

    def test_mesa_cohort(self):
        self.assertEqual(self.mesa_ids, ["0001"])

    def test_recording_without_annotation_is_skipped(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            make_recording(tmp, "shhs1-200001", SHHS_CHANNELS)
            make_recording(tmp, "shhs1-200002", SHHS_CHANNELS)
            os.remove(os.path.join(tmp, "shhs1-200002-profusion.xml"))
            NSRRSleepDataset.prepare_metadata(tmp, "shhs")
            with open(os.path.join(tmp, "nsrr-shhs-pyhealth.csv")) as f:
                rows = f.read().strip().splitlines()
        self.assertEqual(len(rows), 2)  # header + the annotated recording only

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            NSRRSleepDataset(self.shhs, cohort="unknown")
        with self.assertRaises(ValueError):  # no MESA files under the SHHS root
            NSRRSleepDataset(self.shhs, cohort="mesa")


class TestReaders(NSRRTestCase):
    def test_edf_native_rates_and_labels(self):
        path = os.path.join(self.shhs, "edfs", "shhs1-200001.edf")
        out = _read_edf_channels(path, ["ecg", "thor res", "missing"])
        self.assertEqual(set(out), {"ecg", "thor res"})
        ecg, rate = out["ecg"]
        self.assertEqual(rate, 125.0)
        self.assertEqual(len(ecg), 125 * 30 * N_EPOCHS)
        self.assertEqual(out["thor res"][1], 10.0)
        self.assertLessEqual(float(np.abs(ecg).max()), 100.0)  # physical range

    def test_invalid_edf_raises(self):
        bad = os.path.join(self.shhs, "bad.edf")
        with open(bad, "wb") as f:
            f.write(b"not an edf")
        with self.assertRaises(ValueError):
            _read_edf_channels(bad, ["ecg"])

    def test_stage_codes(self):
        path = os.path.join(self.shhs, "edfs", "shhs1-200001-profusion.xml")
        self.assertEqual(_read_stage_codes(path).tolist(), STAGES)


class TestSetTaskAndModel(NSRRTestCase):
    """Checks that the task output works with ``set_task`` and ``MLP``."""

    def test_set_task_and_mlp_forward(self):
        import torch

        from pyhealth.datasets import get_dataloader
        from pyhealth.models import MLP

        samples = self.shhs_ds.set_task(SleepStagingCardioRespiratory())
        self.assertEqual(len(samples), 3 * (N_EPOCHS - 1))  # three nights
        model = MLP(dataset=samples, embedding_dim=8, hidden_dim=8)
        batch = next(iter(get_dataloader(samples, batch_size=4, shuffle=False)))
        output = model(**batch)
        self.assertEqual(output["y_prob"].shape[0], 4)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()


class TestSleepStagingCardioRespiratory(NSRRTestCase):
    def test_samples_shapes_and_labels(self):
        samples = SleepStagingCardioRespiratory()(self.shhs_patient)
        self.assertEqual(len(samples), N_EPOCHS - 1)  # unscored epoch dropped
        self.assertEqual([s["label"] for s in samples], [0, 0, 1, 1, 1, 2, 2, 3, 3])
        self.assertEqual(samples[0]["ecg"].shape, (POINTS_PER_EPOCH["ecg"],))
        self.assertEqual(samples[0]["thx"].shape, (POINTS_PER_EPOCH["thx"],))
        self.assertEqual(samples[0]["abd"].shape, (POINTS_PER_EPOCH["abd"],))
        self.assertEqual(samples[0]["ecg"].dtype, np.float32)
        self.assertEqual(samples[0]["visit"], "shhs1")

    def test_normalization_is_per_night(self):
        t = np.arange(125 * 30 * 4) / 125.0  # four epochs of a 1 Hz sine
        signal = 3.0 + 5.0 * np.sin(2 * np.pi * t)
        out = _to_epochs(signal, 125.0, 4, POINTS_PER_EPOCH["ecg"], normalize=True)
        self.assertEqual(out.shape, (4, POINTS_PER_EPOCH["ecg"]))
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(out.std()), 1.0, delta=0.05)
        raw = _to_epochs(signal, 125.0, 4, POINTS_PER_EPOCH["ecg"], normalize=False)
        self.assertGreater(float(raw.mean()), 2.0)

    def test_label_schemes(self):
        patient = self.shhs_patient
        five = SleepStagingCardioRespiratory(label_scheme="five_class")(patient)
        self.assertEqual([s["label"] for s in five], [0, 0, 1, 2, 2, 3, 3, 4, 4])
        binary = SleepStagingCardioRespiratory(label_scheme="sleep_wake")(patient)
        self.assertEqual([s["label"] for s in binary], [0, 0, 1, 1, 1, 1, 1, 1, 1])

    def test_modality_subset_sets_schema(self):
        task = SleepStagingCardioRespiratory(modalities=("ecg", "thx"))
        self.assertEqual(set(task.input_schema), {"ecg", "thx"})
        sample = task(self.shhs_patient)[0]
        self.assertNotIn("abd", sample)

    def test_mesa_ppg_and_channel_aliases(self):
        task = SleepStagingCardioRespiratory(modalities=("ecg", "ppg", "thx", "abd"))
        samples = task(self.mesa_patient)
        self.assertEqual(len(samples), N_EPOCHS - 1)
        self.assertEqual(samples[0]["ppg"].shape, (POINTS_PER_EPOCH["ppg"],))

    def test_missing_channel_skips_recording(self):
        patient = self.shhs_patient  # SHHS has no PPG
        task = SleepStagingCardioRespiratory(modalities=("ecg", "ppg"))
        with self.assertLogs("pyhealth.tasks.sleep_staging_nsrr", level="WARNING"):
            self.assertEqual(task(patient), [])

    def test_invalid_arguments(self):
        for kwargs in (
            {"modalities": ("eeg",)},
            {"modalities": ()},
            {"modalities": ("ecg", "ecg")},
            {"label_scheme": "nope"},
        ):
            with self.assertRaises(ValueError):
                SleepStagingCardioRespiratory(**kwargs)


class TestSleepStagingIBI(unittest.TestCase):
    """Checks SHHS/MESA IBI tasks on deterministic synthetic EDF/XML files."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = cls.tmp.name
        cls.shhs_root = os.path.join(root, "shhs_ibi")
        cls.mesa_root = os.path.join(root, "mesa_ibi")
        shhs_edfs = os.path.join(cls.shhs_root, "edfs")
        mesa_edfs = os.path.join(cls.mesa_root, "edfs")
        os.makedirs(shhs_edfs)
        os.makedirs(mesa_edfs)

        n_seconds = 30 * len(STAGES)
        write_edf_array(
            os.path.join(shhs_edfs, "shhs1-300001.edf"),
            [("ECG", 125)],
            {"ECG": periodic_signal(n_seconds, 125, 125)},
        )
        write_xml(
            os.path.join(shhs_edfs, "shhs1-300001-profusion.xml"), STAGES
        )
        write_edf_array(
            os.path.join(mesa_edfs, "mesa-sleep-0002.edf"),
            [("Pleth", 256)],
            {"Pleth": periodic_signal(n_seconds, 256, 256)},
        )
        write_xml(
            os.path.join(mesa_edfs, "mesa-sleep-0002-profusion.xml"), STAGES
        )

        cache = os.path.join(root, "ibi_cache")
        cls.shhs_ds = NSRRSleepDataset(
            cls.shhs_root, cohort="shhs", cache_dir=cache
        )
        cls.mesa_ds = NSRRSleepDataset(
            cls.mesa_root, cohort="mesa", cache_dir=cache
        )
        cls.shhs_patient = cls.shhs_ds.get_patient("300001")
        cls.mesa_patient = cls.mesa_ds.get_patient("0002")
        cls.expected_labels = [0, 0, 1, 1, 1, 1, 1, 2, 2, -1]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shhs_samples(self):
        samples = SleepStagingIBISHHS(peak_backend="scipy")(self.shhs_patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["signal"].shape, (len(STAGES), 750))
        self.assertEqual(sample["signal"].dtype, np.float32)
        self.assertEqual(sample["label"].tolist(), self.expected_labels)
        self.assertEqual(sample["lengths"], len(STAGES))
        self.assertEqual(sample["record_id"], "shhs1-300001")

    def test_mesa_samples(self):
        samples = SleepStagingIBIMESA(peak_backend="scipy")(self.mesa_patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["signal"].shape, (len(STAGES), 750))
        self.assertEqual(sample["signal"].dtype, np.float32)
        self.assertEqual(sample["label"].tolist(), self.expected_labels)
        self.assertEqual(sample["lengths"], len(STAGES))

    def test_pooled_schemas_match(self):
        shhs = SleepStagingIBISHHS(peak_backend="scipy")
        mesa = SleepStagingIBIMESA(peak_backend="scipy")
        self.assertEqual(shhs.input_schema, mesa.input_schema)
        self.assertEqual(shhs.output_schema, mesa.output_schema)
        shhs_sample = shhs(self.shhs_patient)[0]
        mesa_sample = mesa(self.mesa_patient)[0]
        self.assertEqual(shhs_sample["signal"].shape, mesa_sample["signal"].shape)
        self.assertEqual(shhs_sample["label"].dtype, mesa_sample["label"].dtype)

    def test_max_epochs_truncates(self):
        task = SleepStagingIBISHHS(peak_backend="scipy", max_epochs=4)
        sample = task(self.shhs_patient)[0]
        self.assertEqual(sample["signal"].shape, (4, 750))
        self.assertEqual(sample["label"].tolist(), self.expected_labels[:4])
        self.assertEqual(sample["lengths"], 4)

    def test_set_task_integration(self):
        import torch

        task = SleepStagingIBISHHS(peak_backend="scipy", max_epochs=4)
        samples = self.shhs_ds.set_task(task, num_workers=1)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(tuple(sample["signal"].shape), (4, 750))
        self.assertEqual(sample["signal"].dtype, torch.float32)
        self.assertEqual(tuple(sample["label"].shape), (4,))
        self.assertEqual(sample["label"].dtype, torch.long)
        self.assertEqual(sample["lengths"], 4)


if __name__ == "__main__":
    unittest.main()
