"""AHEPADementiaDataset and DementiaClassificationAHEPA, on small .set files."""

import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.io

from pyhealth.datasets import AHEPADementiaDataset
from pyhealth.datasets.metadata_csv import resolve_metadata_root
from pyhealth.tasks import DementiaClassificationAHEPA
from pyhealth.tasks.dementia_classification_ahepa import (
    AHEPA_CHANNELS,
    AHEPA_GROUPS,
    AHEPA_PROBLEMS,
)

SFREQ = 20
PARTICIPANTS = (
    # participant_id, Gender, Age, Group, MMSE
    ("sub-001", "F", "57", "A", "16"),
    ("sub-002", "M", "70", "A", "14"),
    ("sub-037", "M", "57", "C", "30"),
    ("sub-038", "F", "62", "C", "30"),
    ("sub-066", "M", "73", "F", "20"),
)


def _write_set(path: Path, data: np.ndarray, names=AHEPA_CHANNELS) -> None:
    """Write ``data`` (microvolts, one row per channel) as an EEGLAB file.

    The variables are the ones MNE reads from the real files, which store
    the signal inside the ``.set`` file.
    """
    chanlocs = np.zeros(len(names), dtype=[("labels", "O")])
    for i, name in enumerate(names):
        chanlocs[i]["labels"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        path,
        {
            "data": data.astype(np.float32),
            "nbchan": len(names),
            "pnts": data.shape[1],
            "srate": float(SFREQ),
            "trials": 1,
            "xmin": 0.0,
            "xmax": (data.shape[1] - 1) / SFREQ,
            "chanlocs": chanlocs,
            "event": np.array([]),
            "epoch": np.array([]),
            "icawinv": np.array([]),
            "icasphere": np.array([]),
            "icaweights": np.array([]),
            "icaact": np.array([]),
            "setname": "synthetic",
            "ref": "common",
        },
        appendmat=False,
    )


def _signal(seconds: float, offset: float = 0.0, n_channels: int = 19) -> np.ndarray:
    """Channel ``i`` holds ``100 * i + offset`` plus the sample index."""
    samples = np.arange(round(seconds * SFREQ))
    return np.stack([100 * i + offset + samples for i in range(n_channels)])


def _set_path(root: Path, subject: str, derivatives: bool) -> Path:
    folder = root / "derivatives" if derivatives else root
    return folder / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"


def _write_dataset(root: Path, seconds: float = 10.0, participants=PARTICIPANTS):
    """Write participants.tsv and both recordings of every participant."""
    lines = ["participant_id\tGender\tAge\tGroup\tMMSE"]
    lines += ["\t".join(row) for row in participants]
    root.mkdir(parents=True, exist_ok=True)
    (root / "participants.tsv").write_text("\n".join(lines) + "\n")
    for subject, *_ in participants:
        _write_set(_set_path(root, subject, True), _signal(seconds))
        _write_set(_set_path(root, subject, False), _signal(seconds, offset=5000))


def _patient(root: Path, subject: str, group: str, **overrides) -> SimpleNamespace:
    """A patient with one ``recordings`` event, like a row of the CSV."""
    fields = {
        "signal_file": str(_set_path(root, subject, True)),
        "raw_signal_file": str(_set_path(root, subject, False)),
        "group": group,
    }
    fields.update(overrides)
    event = SimpleNamespace(**fields)
    return SimpleNamespace(
        patient_id=subject, get_events=lambda event_type=None: [event]
    )


def _task(**kwargs) -> DementiaClassificationAHEPA:
    """A task with 4 second windows shifted by 2."""
    options = {"window_seconds": 4, "stride_seconds": 2}
    options.update(kwargs)
    return DementiaClassificationAHEPA(**options)


def _prepare(root: Path) -> pd.DataFrame:
    ds = AHEPADementiaDataset.__new__(AHEPADementiaDataset)
    ds.root = str(root)
    ds.prepare_metadata()
    return pd.read_csv(
        root / "ahepa_dementia-pyhealth.csv", dtype=str, keep_default_na=False
    )


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "ds004504"


class TestAHEPAMetadata(_TempDir):
    def test_one_row_per_participant_with_both_recordings(self):
        _write_dataset(self.root)

        df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), [row[0] for row in PARTICIPANTS])
        first = df.iloc[0]
        self.assertEqual(
            Path(first["signal_file"]), _set_path(self.root, "sub-001", True)
        )
        self.assertEqual(
            Path(first["raw_signal_file"]), _set_path(self.root, "sub-001", False)
        )
        self.assertEqual(
            (first["group"], first["gender"], first["age"], first["mmse"]),
            ("A", "F", "57", "16"),
        )
        self.assertEqual(list(df["group"]), ["A", "A", "C", "C", "F"])

    def test_partial_download(self):
        _write_dataset(self.root)
        _set_path(self.root, "sub-002", True).unlink()
        _set_path(self.root, "sub-037", False).unlink()
        for derivatives in (True, False):
            _set_path(self.root, "sub-066", derivatives).unlink()

        with self.assertLogs(
            "pyhealth.datasets.ahepa_dementia", level="WARNING"
        ) as logs:
            df = _prepare(self.root).set_index("patient_id")

        self.assertEqual(list(df.index), ["sub-001", "sub-002", "sub-037", "sub-038"])
        self.assertEqual(df.loc["sub-002", "signal_file"], "")
        self.assertNotEqual(df.loc["sub-002", "raw_signal_file"], "")
        self.assertEqual(df.loc["sub-037", "raw_signal_file"], "")
        self.assertIn("1 of 5 subjects", logs.output[0])

    def test_no_participants_file_or_no_recording_writes_nothing(self):
        self.root.mkdir(parents=True)
        for with_participants in (False, True):
            if with_participants:
                (self.root / "participants.tsv").write_text(
                    "participant_id\tGender\tAge\tGroup\tMMSE\nsub-001\tF\t57\tA\t16\n"
                )
            with self.assertLogs("pyhealth.datasets.ahepa_dementia", level="WARNING"):
                ds = AHEPADementiaDataset.__new__(AHEPADementiaDataset)
                ds.root = str(self.root)
                ds.prepare_metadata()
            self.assertFalse((self.root / "ahepa_dementia-pyhealth.csv").exists())

    def test_existing_csv_is_left_untouched(self):
        _write_dataset(self.root, participants=PARTICIPANTS[:2])
        _set_path(self.root, "sub-002", True).unlink()
        _set_path(self.root, "sub-002", False).unlink()
        with self.assertLogs(
            "pyhealth.datasets.ahepa_dementia", level="WARNING"
        ) as logs:
            _prepare(self.root)
        csv_path = self.root / "ahepa_dementia-pyhealth.csv"
        first = csv_path.read_bytes()
        # The warning says how to index a recording downloaded later.
        self.assertIn("Delete ahepa_dementia-pyhealth.csv", logs.output[0])
        _write_set(_set_path(self.root, "sub-002", True), _signal(10))

        df = _prepare(self.root)

        self.assertEqual(csv_path.read_bytes(), first)
        self.assertEqual(list(df["patient_id"]), ["sub-001"])

    def test_metadata_goes_to_the_home_cache_when_root_is_read_only(self):
        _write_dataset(self.root, participants=PARTICIPANTS[:2])
        home = self.root.parent / "home"
        real_to_csv = pd.DataFrame.to_csv

        def to_csv(df, path_or_buf=None, *args, **kwargs):
            # Stands in for a read-only root: chmod has no effect on Windows.
            if str(path_or_buf).startswith(str(self.root)):
                raise PermissionError(f"read-only: {path_or_buf}")
            return real_to_csv(df, path_or_buf, *args, **kwargs)

        with (
            patch.object(pd.DataFrame, "to_csv", to_csv),
            patch("pyhealth.datasets.metadata_csv.Path.home", return_value=home),
        ):
            ds = AHEPADementiaDataset.__new__(AHEPADementiaDataset)
            ds.root = str(self.root)
            ds.prepare_metadata()
            metadata_root = Path(
                resolve_metadata_root(
                    self.root, "ahepa_dementia", "ahepa_dementia-pyhealth.csv"
                )
            )

        self.assertFalse((self.root / "ahepa_dementia-pyhealth.csv").exists())
        self.assertTrue(metadata_root.is_relative_to(home))
        cached = pd.read_csv(metadata_root / "ahepa_dementia-pyhealth.csv", dtype=str)
        self.assertEqual(list(cached["patient_id"]), ["sub-001", "sub-002"])

    def test_participants_rows_are_cleaned(self):
        _write_dataset(self.root, participants=PARTICIPANTS[:3])
        header = "participant_id\tGender\tAge\tGroup\tMMSE\n"
        (self.root / "participants.tsv").write_text(
            header
            + " sub-001 \tF\t57\t a \t16\n"  # spaces, lower-case group
            + "\tM\t70\tA\t14\n"  # no id
            + "sub-037\tM\t57\tC\t30\n"
            + "sub-037\tF\t62\tF\t20\n"  # repeated id
        )

        with self.assertLogs(
            "pyhealth.datasets.ahepa_dementia", level="WARNING"
        ) as logs:
            df = _prepare(self.root)

        self.assertEqual(list(df["patient_id"]), ["sub-001", "sub-037"])
        self.assertEqual(list(df["group"]), ["A", "C"])
        self.assertEqual(len(logs.output), 2)
        self.assertIn("1 row(s) without a participant_id", logs.output[0])
        self.assertIn("sub-037", logs.output[1])

    def test_malformed_participants_file_raises(self):
        self.root.mkdir(parents=True)
        contents = {
            "empty": "",
            "comma separated": "participant_id,Gender,Group\nsub-001,F,A\n",
            "no Group column": "participant_id\tGender\nsub-001\tF\n",
        }
        for name, content in contents.items():
            (self.root / "participants.tsv").write_text(content)
            ds = AHEPADementiaDataset.__new__(AHEPADementiaDataset)
            ds.root = str(self.root)
            with self.subTest(name), self.assertRaises(ValueError):
                ds.prepare_metadata()

    def test_default_task(self):
        ds = AHEPADementiaDataset.__new__(AHEPADementiaDataset)
        self.assertIsInstance(ds.default_task, DementiaClassificationAHEPA)


class TestDementiaClassificationAHEPA(_TempDir):
    def setUp(self):
        super().setUp()
        _write_dataset(self.root)

    def test_windows_signal_and_label(self):
        samples = _task()(_patient(self.root, "sub-001", "A"))

        # 10 s of signal, 4 s windows every 2 s: starts 0, 2, 4 and 6.
        self.assertEqual([s["start_second"] for s in samples], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual({s["label"] for s in samples}, {1})
        self.assertEqual({s["group"] for s in samples}, {"A"})
        second = samples[1]
        self.assertEqual(second["patient_id"], "sub-001")
        self.assertEqual(second["signal"].shape, (19, 4 * SFREQ))
        self.assertEqual(second["signal"].dtype, np.float32)
        # Channel i holds 100 * i plus the sample index; the window starts at
        # sample 2 * SFREQ.
        expected = _signal(10)[:, 2 * SFREQ : 6 * SFREQ]
        np.testing.assert_allclose(second["signal"], expected)

    def test_labels_of_every_problem(self):
        expected = {
            # problem: label of groups A, F and C (None: left out)
            "ad_vs_cn": {"A": 1, "F": None, "C": 0},
            "ftd_vs_cn": {"A": None, "F": 1, "C": 0},
            "ad_vs_ftd": {"A": 1, "F": 0, "C": None},
            "dementia_vs_cn": {"A": 1, "F": 1, "C": 0},
            "three_class": {"A": 1, "F": 2, "C": 0},
        }
        self.assertEqual(set(expected), set(AHEPA_PROBLEMS))
        subject = {"A": "sub-001", "F": "sub-066", "C": "sub-037"}
        self.assertEqual(set(subject), set(AHEPA_GROUPS))
        for problem, labels in expected.items():
            task = _task(problem=problem)
            for group, label in labels.items():
                with self.subTest(problem=problem, group=group):
                    samples = task(_patient(self.root, subject[group], group))
                    if label is None:
                        self.assertEqual(samples, [])
                    else:
                        self.assertEqual({s["label"] for s in samples}, {label})

    def test_output_schema_follows_the_problem(self):
        for problem in AHEPA_PROBLEMS:
            # The three classes are fixed, so that a label keeps its meaning
            # when a group is absent.
            kind = (
                ("multiclass", {"labels": 3}) if problem == "three_class" else "binary"
            )
            with self.subTest(problem=problem):
                task = DementiaClassificationAHEPA(problem=problem)
                self.assertEqual(task.output_schema, {"label": kind})
        # The class attribute is not changed by the three-class instance.
        self.assertEqual(DementiaClassificationAHEPA.output_schema, {"label": "binary"})

    def test_raw_variant_reads_the_unprocessed_recording(self):
        cleaned = _task()(_patient(self.root, "sub-001", "A"))
        raw = _task(variant="raw")(_patient(self.root, "sub-001", "A"))

        np.testing.assert_allclose(raw[0]["signal"] - cleaned[0]["signal"], 5000.0)

    def test_average_reference_removes_the_shared_signal(self):
        # Every channel of the synthetic file holds the same ramp (the sample
        # index) plus its own constant 100 * i: the ramp is shared.
        stored = _task()(_patient(self.root, "sub-001", "A"))
        average = _task(reference="average")(_patient(self.root, "sub-001", "A"))

        self.assertGreater(float(np.ptp(stored[1]["signal"][0])), 70.0)
        constants = 100.0 * np.arange(19)
        expected = np.repeat((constants - constants.mean())[:, None], 4 * SFREQ, axis=1)
        np.testing.assert_allclose(average[1]["signal"], expected, atol=1e-3)
        np.testing.assert_allclose(average[1]["signal"].mean(axis=0), 0.0, atol=1e-3)

    def test_missing_recording_of_the_variant_is_skipped(self):
        patient = _patient(self.root, "sub-001", "A", signal_file="")

        with self.assertLogs(
            "pyhealth.tasks.dementia_classification_ahepa", level="WARNING"
        ):
            self.assertEqual(_task()(patient), [])
        self.assertEqual(len(_task(variant="raw")(patient)), 4)

    def test_missing_channel_skips_the_recording(self):
        path = _set_path(self.root, "sub-001", True)
        _write_set(path, _signal(10, n_channels=18), names=AHEPA_CHANNELS[:-1])

        with self.assertLogs(
            "pyhealth.tasks.dementia_classification_ahepa", level="WARNING"
        ) as logs:
            samples = _task()(_patient(self.root, "sub-001", "A"))

        self.assertEqual(samples, [])
        self.assertIn("Pz", logs.output[0])

    def test_channels_are_returned_in_the_reference_order(self):
        path = _set_path(self.root, "sub-001", True)
        _write_set(path, _signal(10)[::-1], names=AHEPA_CHANNELS[::-1])

        samples = _task()(_patient(self.root, "sub-001", "A"))

        np.testing.assert_allclose(samples[0]["signal"], _signal(10)[:, : 4 * SFREQ])

    def test_unknown_group_is_skipped_with_a_warning(self):
        with self.assertLogs(
            "pyhealth.tasks.dementia_classification_ahepa", level="WARNING"
        ):
            self.assertEqual(_task()(_patient(self.root, "sub-001", "X")), [])

    def test_resampling_halves_the_samples_and_keeps_the_values(self):
        samples = _task(resample_rate=SFREQ / 2)(_patient(self.root, "sub-001", "A"))

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["signal"].shape, (19, 4 * SFREQ // 2))
        self.assertEqual([s["start_second"] for s in samples], [0.0, 2.0, 4.0, 6.0])
        # The file holds a ramp on every channel; at half the rate each window
        # is every other sample of it, including the first and last windows,
        # where zero-padding in the resampler would bend the ramp.
        every_other = _signal(10)[:, ::2]
        for index, sample in enumerate(samples):
            start = index * 2 * SFREQ // 2
            np.testing.assert_allclose(
                sample["signal"],
                every_other[:, start : start + 4 * SFREQ // 2],
                atol=1e-3,
            )

    def test_resample_rate_that_cannot_be_reached_raises(self):
        with self.assertRaises(ValueError):
            _task(resample_rate=math.pi)(_patient(self.root, "sub-001", "A"))

    def test_recording_shorter_than_a_window_is_skipped(self):
        _write_set(_set_path(self.root, "sub-001", True), _signal(3))

        with self.assertLogs(
            "pyhealth.tasks.dementia_classification_ahepa", level="WARNING"
        ) as logs:
            samples = _task()(_patient(self.root, "sub-001", "A"))

        self.assertEqual(samples, [])
        self.assertIn("shorter than one window", logs.output[0])

    def test_tail_that_does_not_fill_a_window_is_dropped(self):
        _write_set(_set_path(self.root, "sub-001", True), _signal(11))

        samples = _task()(_patient(self.root, "sub-001", "A"))

        # 11 s: the window starting at 8 s would end at 12 s.
        self.assertEqual([s["start_second"] for s in samples], [0.0, 2.0, 4.0, 6.0])

    def test_window_or_stride_below_one_sample_raises(self):
        for kwargs in ({"window_seconds": 0.01}, {"stride_seconds": 0.01}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                _task(**kwargs)(_patient(self.root, "sub-001", "A"))

    def test_extra_channels_of_the_file_are_left_out(self):
        data = np.concatenate([_signal(10), np.full((1, 10 * SFREQ), 9999.0)])
        _write_set(
            _set_path(self.root, "sub-001", True), data, names=(*AHEPA_CHANNELS, "ECG")
        )

        stored = _task()(_patient(self.root, "sub-001", "A"))
        average = _task(reference="average")(_patient(self.root, "sub-001", "A"))

        np.testing.assert_allclose(stored[0]["signal"], _signal(10)[:, : 4 * SFREQ])
        # The mean is taken over the 19 EEG channels only.
        constants = 100.0 * np.arange(19)
        np.testing.assert_allclose(
            average[0]["signal"][:, 0], constants - constants.mean(), atol=1e-3
        )

    def test_group_code_is_read_without_case_or_spaces(self):
        samples = _task()(_patient(self.root, "sub-001", " a "))
        self.assertEqual({s["label"] for s in samples}, {1})
        self.assertEqual({s["group"] for s in samples}, {"A"})

    def test_default_window_and_stride(self):
        task = DementiaClassificationAHEPA()
        self.assertEqual(
            (task.problem, task.window_seconds, task.stride_seconds, task.variant),
            ("ad_vs_cn", 4.0, 2.0, "derivatives"),
        )
        self.assertIsNone(task.resample_rate)
        self.assertEqual(task.reference, "stored")
        self.assertEqual(len(AHEPA_CHANNELS), 19)

    def test_invalid_arguments(self):
        cases = (
            {"problem": "ad"},
            {"variant": "cleaned"},
            {"reference": "cz"},
            {"window_seconds": 0},
            {"window_seconds": float("nan")},
            {"stride_seconds": -1},
            {"stride_seconds": float("inf")},
            {"resample_rate": 0},
        )
        for kwargs in cases:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                DementiaClassificationAHEPA(**kwargs)


class TestAHEPAEndToEnd(_TempDir):
    def test_set_task_keeps_the_groups_of_the_problem(self):
        _write_dataset(self.root)

        dataset = AHEPADementiaDataset(
            root=str(self.root), cache_dir=str(self.root.parent / "cache")
        )
        samples = dataset.set_task(_task())

        self.assertEqual(len(dataset.unique_patient_ids), 5)
        # Two Alzheimer subjects and two controls, 4 windows each; the
        # frontotemporal subject is left out of ad_vs_cn.
        self.assertEqual(len(samples), 16)
        self.assertEqual(sum(int(s["label"]) for s in samples), 8)
        self.assertEqual(tuple(samples[0]["signal"].shape), (19, 4 * SFREQ))

    def test_three_class_labels_keep_their_meaning_without_controls(self):
        patients = tuple(p for p in PARTICIPANTS if p[3] != "C")
        _write_dataset(self.root, participants=patients)

        dataset = AHEPADementiaDataset(
            root=str(self.root), cache_dir=str(self.root.parent / "cache")
        )
        samples = dataset.set_task(_task(problem="three_class"))

        # Alzheimer's disease stays 1 and frontotemporal dementia 2, although
        # no sample has the control label 0.
        by_group = {s["group"]: int(s["label"]) for s in samples}
        self.assertEqual(by_group, {"A": 1, "F": 2})


def _load_example():
    """Load the example script as a module, without running ``main``."""
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "eeg"
        / "ahepa_dementia"
        / "ahepa_dementia_classification.py"
    )
    spec = importlib.util.spec_from_file_location("ahepa_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAHEPAExample(_TempDir):
    """Smoke test of the example script on the synthetic recordings."""

    def test_relative_band_powers_sum_to_one(self):
        example = _load_example()
        rate = example.RESAMPLE_RATE
        t = np.arange(4 * rate) / rate
        signal = np.stack([np.sin(2 * np.pi * f * t) for f in (2, 6, 10, 20, 30)])
        power = example.relative_band_power(signal, rate)
        self.assertEqual(power.shape, (5, 5))
        np.testing.assert_allclose(power.sum(axis=1), 1.0, rtol=1e-6)
        # Each sine falls in its own band.
        np.testing.assert_array_equal(power.argmax(axis=1), np.arange(5))

    def test_main_runs_on_synthetic_recordings(self):
        example = _load_example()
        _write_dataset(self.root)
        cache_dir = str(self.root.parent / "cache")

        def dataset_with_temp_cache(root):
            return AHEPADementiaDataset(root=root, cache_dir=cache_dir)

        out = io.StringIO()
        argv = ["ahepa_dementia_classification.py", "--root", str(self.root)]
        with (
            patch.object(sys, "argv", argv),
            patch.object(example, "AHEPADementiaDataset", dataset_with_temp_cache),
            contextlib.redirect_stdout(out),
        ):
            example.main()

        printed = out.getvalue()
        self.assertIn("windows of 4 subjects, 5 features", printed)
        self.assertIn("kNN (k=3)", printed)
        self.assertIn("random forest", printed)


if __name__ == "__main__":
    unittest.main()
