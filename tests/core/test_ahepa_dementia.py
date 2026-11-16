"""AHEPADementiaDataset and DementiaClassificationAHEPA, on small .set files."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.io

from pyhealth.datasets import AHEPADementiaDataset
from pyhealth.tasks import DementiaClassificationAHEPA
from pyhealth.tasks.dementia_classification_ahepa import (
    AHEPA_CHANNELS,
    AHEPA_GROUPS,
    PROBLEMS,
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
        self.assertEqual(set(expected), set(PROBLEMS))
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
        for problem in PROBLEMS:
            kind = "multiclass" if problem == "three_class" else "binary"
            with self.subTest(problem=problem):
                task = DementiaClassificationAHEPA(problem=problem)
                self.assertEqual(task.output_schema, {"label": kind})
        # The class attribute is not changed by the three-class instance.
        self.assertEqual(DementiaClassificationAHEPA.output_schema, {"label": "binary"})

    def test_raw_variant_reads_the_unprocessed_recording(self):
        cleaned = _task()(_patient(self.root, "sub-001", "A"))
        raw = _task(variant="raw")(_patient(self.root, "sub-001", "A"))

        np.testing.assert_allclose(raw[0]["signal"] - cleaned[0]["signal"], 5000.0)

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

    def test_resampling_halves_the_samples(self):
        samples = _task(resample_rate=SFREQ / 2)(_patient(self.root, "sub-001", "A"))

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["signal"].shape, (19, 4 * SFREQ // 2))
        self.assertEqual([s["start_second"] for s in samples], [0.0, 2.0, 4.0, 6.0])

    def test_default_window_and_stride(self):
        task = DementiaClassificationAHEPA()
        self.assertEqual(
            (task.problem, task.window_seconds, task.stride_seconds, task.variant),
            ("ad_vs_cn", 30.0, 15.0, "derivatives"),
        )
        self.assertIsNone(task.resample_rate)
        self.assertEqual(len(AHEPA_CHANNELS), 19)

    def test_invalid_arguments(self):
        cases = (
            {"problem": "ad"},
            {"variant": "cleaned"},
            {"window_seconds": 0},
            {"stride_seconds": -1},
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


if __name__ == "__main__":
    unittest.main()
