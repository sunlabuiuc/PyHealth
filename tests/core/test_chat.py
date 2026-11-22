"""CHATDataset metadata, on small synthetic EDF and annotation files."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets import CHATDataset
from pyhealth.datasets.chat import read_chat_annotations, summarize_chat_annotations
from pyhealth.datasets.metadata_csv import resolve_metadata_root
from tests.core.psg_test_utils import write_edf, write_tsv

SFREQ = 8
SECONDS = 120
CHANNELS = ("C4", "M1", "Airflow", "SaO2")
CSV = "chat-pyhealth.csv"

# (visit, nsrrid) -> annotation rows as (onset, duration, description)
STUDIES = {
    ("baseline", "300001"): [
        (0, 30, "Wake"),
        (30, 30, "Stage 2 sleep"),
        (60, 30, "Stage 3 sleep"),
        (90, 30, "REM sleep"),
        (35.0, 12.0, "Obstructive apnea"),
        (65.0, 15.0, "Hypopnea"),
        (95.0, 10.0, "Central apnea"),
        (100.0, 3.0, "Arousal"),
    ],
    ("followup", "300001"): [
        (0, 30, "Wake"),
        (30, 60, "Stage 2 sleep"),
        (90, 30, "Wake"),
        (40.0, 11.0, "Hypopnea"),
    ],
    ("baseline", "300002"): [
        (0, 60, "Stage 1 sleep"),
        (60, 60, "Wake"),
    ],
}

SUBJECTS = {
    "baseline": pd.DataFrame(
        {
            "nsrrid": ["300001", "300002"],
            "ageyear_at_meas": ["6.0", "8.5"],
            "male": ["1", "0"],
            "race3": ["2", "1"],
            "ahi_a0h3": ["4.1", "2.2"],
        }
    ),
    "followup": pd.DataFrame(
        {
            "nsrrid": ["300001"],
            "ageyear_at_meas": ["6.6"],
            "male": ["1"],
            "race3": ["2"],
        }
    ),
}

_XML_CONCEPTS = {
    "Wake": ("Stages|Stages", "Wake|0"),
    "Stage 1 sleep": ("Stages|Stages", "Stage 1 sleep|1"),
    "Stage 2 sleep": ("Stages|Stages", "Stage 2 sleep|2"),
    "Stage 3 sleep": ("Stages|Stages", "Stage 3 sleep|3"),
    "Stage 4 sleep": ("Stages|Stages", "Stage 4 sleep|4"),
    "REM sleep": ("Stages|Stages", "REM sleep|5"),
    "Obstructive apnea": (
        "Respiratory|Respiratory",
        "Obstructive apnea|Obstructive Apnea",
    ),
    "Central apnea": ("Respiratory|Respiratory", "Central apnea|Central Apnea"),
    "Mixed apnea": ("Respiratory|Respiratory", "Mixed apnea|Mixed Apnea"),
    "Hypopnea": ("Respiratory|Respiratory", "Hypopnea|Hypopnea"),
    "Arousal": ("Arousals|Arousals", "Arousal|Arousal ()"),
}


def _write_xml(path: Path, rows) -> None:
    """Write rows in the NSRR XML layout, with a Recording Start Time entry."""
    events = [
        (
            "<ScoredEvent><EventType/><EventConcept>Recording Start Time"
            "</EventConcept><Start>0</Start><Duration>120.0</Duration>"
            "<ClockTime>00.00.00 21.30.00</ClockTime></ScoredEvent>"
        )
    ]
    for onset, duration, description in rows:
        etype, concept = _XML_CONCEPTS[description]
        events.append(
            f"<ScoredEvent><EventType>{etype}</EventType>"
            f"<EventConcept>{concept}</EventConcept>"
            f"<Start>{onset}</Start><Duration>{duration}</Duration></ScoredEvent>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<PSGAnnotation>'
        "<EpochLength>30</EpochLength><ScoredEvents>"
        + "".join(events)
        + "</ScoredEvents></PSGAnnotation>",
        encoding="utf-8",
    )


def _edf(root: Path, visit: str, nsrrid: str) -> Path:
    return root / "polysomnography" / "edfs" / visit / f"chat-{visit}-{nsrrid}.edf"


def _xml(root: Path, visit: str, nsrrid: str) -> Path:
    return (
        root
        / "polysomnography"
        / "annotations-events-nsrr"
        / visit
        / f"chat-{visit}-{nsrrid}-nsrr.xml"
    )


def _tsv(root: Path, visit: str, nsrrid: str) -> Path:
    return _edf(root, visit, nsrrid).with_name(f"chat-{visit}-{nsrrid}-nsrr.tsv")


def _write_dataset(root: Path, studies=STUDIES, subjects=SUBJECTS) -> None:
    for (visit, nsrrid), rows in studies.items():
        write_edf(_edf(root, visit, nsrrid), CHANNELS, sfreq=SFREQ, seconds=SECONDS)
        _write_xml(_xml(root, visit, nsrrid), rows)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    for visit, table in subjects.items():
        table.to_csv(
            root / "datasets" / f"chat-{visit}-dataset-0.14.0.csv", index=False
        )


def _prepare(root: Path) -> pd.DataFrame:
    ds = CHATDataset.__new__(CHATDataset)
    ds.root = str(root)
    ds.prepare_metadata()
    return pd.read_csv(root / CSV, dtype=str, keep_default_na=False)


class _TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "chat"


class TestCHATAnnotations(_TempDir):
    def test_xml_concepts_become_descriptions(self):
        path = self.root / "a-nsrr.xml"
        _write_xml(path, STUDIES[("baseline", "300001")])

        df = read_chat_annotations(path)

        # The Recording Start Time entry is left out.
        self.assertEqual(len(df), len(STUDIES[("baseline", "300001")]))
        self.assertEqual(list(df.columns), ["onset", "duration", "description"])
        self.assertEqual(df.iloc[4].tolist(), [35.0, 12.0, "Obstructive apnea"])
        self.assertEqual(df.iloc[7]["description"], "Arousal")

    def test_xml_and_tsv_give_the_same_table(self):
        rows = STUDIES[("baseline", "300001")]
        _write_xml(self.root / "a.xml", rows)
        write_tsv(self.root / "a.tsv", rows)

        xml = read_chat_annotations(self.root / "a.xml")
        tsv = read_chat_annotations(self.root / "a.tsv")

        pd.testing.assert_frame_equal(xml, tsv, check_dtype=False)

    def test_counts_events_and_sleep_time(self):
        path = self.root / "a.xml"
        _write_xml(path, STUDIES[("baseline", "300001")])

        summary = summarize_chat_annotations(path)

        # 90 s of sleep (stage 2, stage 3, REM), 3 events.
        self.assertEqual((summary["n_apnea"], summary["n_hypopnea"]), (2, 1))
        self.assertAlmostEqual(summary["sleep_hours"], 0.025)
        self.assertAlmostEqual(summary["ahi"], 120.0)

    def test_vocabulary_matches_the_original_code(self):
        path = self.root / "a.xml"
        _write_xml(
            path,
            [
                (0, 30, "Stage 4 sleep"),
                (5, 10, "Mixed apnea"),
                (15, 10, "Obstructive apnea"),
            ],
        )

        summary = summarize_chat_annotations(path)

        # Mixed apneas are not counted; stage 4 is sleep.
        self.assertEqual((summary["n_apnea"], summary["n_hypopnea"]), (1, 0))
        self.assertAlmostEqual(summary["sleep_hours"], 30 / 3600, places=4)


class TestCHATMetadata(_TempDir):
    def test_one_row_per_study_with_subject_tables(self):
        _write_dataset(self.root)

        df = _prepare(self.root)

        # By subject, then visit.
        self.assertEqual(
            list(df["record_id"]),
            ["chat-baseline-300001", "chat-followup-300001", "chat-baseline-300002"],
        )
        self.assertEqual(list(df["patient_id"]), ["300001", "300001", "300002"])
        self.assertEqual(list(df["visit"]), ["baseline", "followup", "baseline"])
        self.assertEqual(list(df["age_years"]), ["6.0", "6.6", "8.5"])
        self.assertEqual(list(df["male"]), ["1", "1", "0"])
        self.assertEqual(list(df["race"]), ["2", "2", "1"])
        row = df.iloc[0]
        self.assertEqual(
            Path(row["signal_file"]), _edf(self.root, "baseline", "300001")
        )
        self.assertEqual(
            Path(row["annotation_file"]), _xml(self.root, "baseline", "300001")
        )
        self.assertEqual(row["annotation_format"], "xml")
        self.assertEqual(float(row["sfreq"]), SFREQ)
        self.assertEqual(float(row["duration_sec"]), SECONDS)
        self.assertEqual(row["channels"].split("|"), list(CHANNELS))
        self.assertEqual((row["n_apnea"], row["n_hypopnea"]), ("2", "1"))
        self.assertEqual(float(row["ahi"]), 120.0)

    def test_tsv_next_to_the_edf_is_used_without_xml(self):
        _write_dataset(self.root)
        _xml(self.root, "baseline", "300002").unlink()
        write_tsv(
            _tsv(self.root, "baseline", "300002"), STUDIES[("baseline", "300002")]
        )

        df = _prepare(self.root).set_index("record_id")

        row = df.loc["chat-baseline-300002"]
        self.assertEqual(row["annotation_format"], "tsv")
        self.assertEqual(
            Path(row["annotation_file"]), _tsv(self.root, "baseline", "300002")
        )

    def test_xml_is_preferred_over_tsv(self):
        _write_dataset(self.root)
        write_tsv(_tsv(self.root, "baseline", "300001"), [(0, 30, "Wake")])

        df = _prepare(self.root).set_index("record_id")

        self.assertEqual(df.loc["chat-baseline-300001", "annotation_format"], "xml")
        self.assertEqual(df.loc["chat-baseline-300001", "n_apnea"], "2")

    def test_studies_without_annotations_or_valid_name_are_left_out(self):
        _write_dataset(self.root)
        _xml(self.root, "followup", "300001").unlink()
        write_edf(
            self.root / "polysomnography" / "edfs" / "baseline" / "other-300009.edf",
            CHANNELS,
        )
        # A file whose visit in the name does not match its folder.
        write_edf(
            _edf(self.root, "followup", "300002").with_name("chat-baseline-300002.edf"),
            CHANNELS,
        )

        with self.assertLogs("pyhealth.datasets.chat", level="WARNING") as logs:
            df = _prepare(self.root)

        self.assertEqual(
            list(df["record_id"]), ["chat-baseline-300001", "chat-baseline-300002"]
        )
        output = "\n".join(logs.output)
        self.assertIn("other-300009.edf", output)
        self.assertIn("chat-followup-300001.edf", output)
        self.assertIn(f"Delete {CSV}", output)

    def test_unreadable_annotation_is_left_out(self):
        _write_dataset(self.root)
        _xml(self.root, "baseline", "300002").write_text("<PSGAnnotation>")

        with self.assertLogs("pyhealth.datasets.chat", level="WARNING"):
            df = _prepare(self.root)

        self.assertNotIn("chat-baseline-300002", list(df["record_id"]))

    def test_missing_subject_table_or_subject_leaves_columns_empty(self):
        subjects = {"baseline": SUBJECTS["baseline"].iloc[[1]]}
        _write_dataset(self.root, subjects=subjects)

        with self.assertLogs("pyhealth.datasets.chat", level="WARNING") as logs:
            df = _prepare(self.root).set_index("record_id")

        self.assertEqual(df.loc["chat-baseline-300001", "age_years"], "")
        self.assertEqual(df.loc["chat-followup-300001", "age_years"], "")
        self.assertEqual(df.loc["chat-baseline-300002", "age_years"], "8.5")
        output = "\n".join(logs.output)
        self.assertIn("chat-followup-dataset-*.csv", output)
        self.assertIn("chat-baseline-300001", output)

    def test_no_studies_writes_nothing(self):
        self.root.mkdir(parents=True)
        ds = CHATDataset.__new__(CHATDataset)
        ds.root = str(self.root)

        with self.assertLogs("pyhealth.datasets.chat", level="WARNING"):
            ds.prepare_metadata()

        self.assertFalse((self.root / CSV).exists())

    def test_existing_csv_is_left_untouched(self):
        _write_dataset(self.root, studies={("baseline", "300002"): []})
        first = _prepare(self.root)
        _write_dataset(self.root)

        df = _prepare(self.root)

        self.assertEqual(list(df["record_id"]), list(first["record_id"]))

    def test_metadata_goes_to_the_home_cache_when_root_is_read_only(self):
        _write_dataset(self.root)
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
            ds = CHATDataset.__new__(CHATDataset)
            ds.root = str(self.root)
            ds.prepare_metadata()
            metadata_root = Path(resolve_metadata_root(self.root, "chat", CSV))

        self.assertFalse((self.root / CSV).exists())
        self.assertTrue(metadata_root.is_relative_to(home))

    def test_newest_subject_table_release_is_read(self):
        _write_dataset(self.root)
        old = SUBJECTS["baseline"].assign(ageyear_at_meas=["1.0", "1.0"])
        old.to_csv(
            self.root / "datasets" / "chat-baseline-dataset-0.9.0.csv", index=False
        )

        with self.assertLogs("pyhealth.datasets.chat", level="WARNING") as logs:
            df = _prepare(self.root).set_index("record_id")

        # 0.14.0 is newer than 0.9.0, although it sorts first as text.
        self.assertEqual(df.loc["chat-baseline-300001", "age_years"], "6.0")
        self.assertIn("chat-baseline-dataset-0.14.0.csv", logs.output[0])

    def test_tsv_without_required_columns_is_left_out(self):
        _write_dataset(self.root)
        _xml(self.root, "baseline", "300002").unlink()
        tsv = _tsv(self.root, "baseline", "300002")
        tsv.write_text("start\tlabel\n0\tWake\n")

        with self.assertLogs("pyhealth.datasets.chat", level="WARNING") as logs:
            df = _prepare(self.root)

        self.assertNotIn("chat-baseline-300002", list(df["record_id"]))
        self.assertIn("duration", logs.output[0])


class TestCHATEndToEnd(_TempDir):
    def test_patients_and_study_events(self):
        _write_dataset(self.root)

        dataset = CHATDataset(
            root=str(self.root), cache_dir=str(self.root.parent / "cache")
        )

        self.assertEqual(sorted(dataset.unique_patient_ids), ["300001", "300002"])
        studies = dataset.get_patient("300001").get_events("studies")
        self.assertEqual(sorted(s.visit for s in studies), ["baseline", "followup"])
        followup = next(s for s in studies if s.visit == "followup")
        self.assertEqual(followup.record_id, "chat-followup-300001")
        self.assertTrue(Path(followup.signal_file).is_file())
        self.assertAlmostEqual(float(followup.ahi), 60.0)


if __name__ == "__main__":
    unittest.main()
