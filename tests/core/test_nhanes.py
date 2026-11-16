"""Unit tests for NHANESDataset (NHANES 2003-2006 accelerometer + mortality).

All data is synthetic: tiny hand-built arrays and files, or the files written
by ``pyhealth.datasets.nhanes_synthetic``. No real NHANES data is used.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import nhanes
from pyhealth.datasets.nhanes import NHANESDataset
from pyhealth.datasets.nhanes_synthetic import (
    PLANTED_NONWEAR,
    PLANTED_SEDENTARY,
    generate_synthetic_nhanes,
)
from pyhealth.tasks.nhanes_mortality_prediction import NHANESMortalityPrediction


def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _minutes(counts, paxcal=1, paxstat=1, first_paxn=1, day=1, seqn=1):
    """Builds PAXRAW-style minute rows starting at midnight of ``day``."""
    counts = np.asarray(counts, dtype=float)
    paxn = np.arange(first_paxn, first_paxn + counts.size)
    clock = paxn - 1
    return pd.DataFrame(
        {
            "SEQN": seqn,
            "PAXSTAT": paxstat,
            "PAXCAL": paxcal,
            "PAXDAY": (day - 1 + clock // 1440) % 7 + 1,
            "PAXN": paxn,
            "PAXHOUR": (clock % 1440) // 60,
            "PAXMINUT": clock % 60,
            "PAXINTEN": counts,
        }
    )


def _all_worn(counts: np.ndarray) -> np.ndarray:
    """Stub non-wear detector: nothing is non-wear."""
    return np.zeros(counts.size, dtype=bool)


def _mortality_line(seqn, elig, mort=".", months_int=".", months_exm="."):
    """Formats one line of the NCHS fixed-width linked mortality file."""
    return f"{seqn:<14}{elig}{mort}" + " " * 26 + f"{months_int:>3}{months_exm:>3}"


class TestNonwearRules(unittest.TestCase):
    """The two non-wear detectors on hand-built minute counts."""

    def test_choi_needs_90_zero_minutes(self):
        """90 zero minutes are non-wear; 89 are not."""
        self.assertTrue(nhanes._choi_nonwear(np.zeros(90)).all())
        self.assertFalse(nhanes._choi_nonwear(np.zeros(89)).any())

    def test_choi_bridges_short_interruptions(self):
        """Up to 2 nonzero minutes with 30 zeros on both sides are bridged."""
        bridged = np.array(_zeros(40) + [5, 5] + _zeros(50))
        self.assertTrue(nhanes._choi_nonwear(bridged).all())
        too_long = np.array(_zeros(40) + [5, 5, 5] + _zeros(50))
        self.assertFalse(nhanes._choi_nonwear(too_long).any())
        short_window = np.array(_zeros(29) + [5] + _zeros(70))
        self.assertFalse(nhanes._choi_nonwear(short_window).any())

    def test_choi_has_no_count_cap(self):
        """Published Choi allows an interruption of any size."""
        counts = np.array(_zeros(45) + [5000] + _zeros(45))
        self.assertTrue(nhanes._choi_nonwear(counts).all())

    def test_authors_rule_spikes_capped_at_100(self):
        """The authors' rule splits at counts above 100."""
        small = np.array(_zeros(60) + [50] + _zeros(60))
        large = np.array(_zeros(60) + [150] + _zeros(60))
        self.assertTrue(nhanes._authors_nonwear(small).all())
        self.assertFalse(nhanes._authors_nonwear(large).any())

    def test_authors_rule_rejects_adjacent_spikes(self):
        """Two adjacent spikes break the period; only the far side is found."""
        counts = np.array(_zeros(100) + [40, 60] + _zeros(100))
        self.assertTrue(nhanes._choi_nonwear(counts).all())
        authors = nhanes._authors_nonwear(counts)
        self.assertFalse(authors[:102].any())
        self.assertTrue(authors[102:].all())

    def test_authors_rule_misses_zero_run_next_to_blips(self):
        """A clean zero run next to repeated blips stays worn (authors only)."""
        counts = np.array(
            _zeros(200) + [5] + _zeros(10) + [5] + _zeros(10) + [5] + _zeros(10) + [500]
        )
        self.assertTrue(nhanes._choi_nonwear(counts)[:200].all())
        self.assertFalse(nhanes._authors_nonwear(counts)[:200].any())

    def test_empty_input(self):
        """Both detectors accept an empty array."""
        for detector in (nhanes._choi_nonwear, nhanes._authors_nonwear):
            self.assertEqual(detector(np.zeros(0)).size, 0)


class TestEpochs(unittest.TestCase):
    """Minute rows to 5-minute epochs, with a stub non-wear detector."""

    def test_sas_zeros_are_cleaned(self):
        """pandas' 5.4e-79 for SAS zeros becomes 0; real values are kept."""
        frame = pd.DataFrame({"x": [5.397605e-79, -5.397605e-79, 27.3]})
        cleaned = nhanes._clean_sas_zeros(frame)
        self.assertEqual(cleaned["x"].tolist(), [0.0, 0.0, 27.3])

    def test_intensity_is_mean_over_worn_minutes(self):
        """Epoch intensity averages worn minutes only; 1 worn minute counts."""
        counts = [10, 20, 30, 40, 50, 1, 2, 3, 4, 5]

        def first_four_nonwear(c):
            flags = np.zeros(c.size, dtype=bool)
            flags[:4] = True
            return flags

        epochs = nhanes._participant_epochs(_minutes(counts), first_four_nonwear)
        self.assertEqual(epochs["intensity"].tolist(), [50.0, 3.0])
        self.assertEqual(epochs["wear_minutes"].tolist(), [1, 5])
        self.assertEqual(epochs["wear_mask"].tolist(), [1, 1])

    def test_unworn_epoch_has_zero_intensity(self):
        """An epoch with no worn minutes has intensity 0 and mask 0."""
        epochs = nhanes._participant_epochs(
            _minutes([7] * 10), lambda c: np.r_[np.ones(5, bool), np.zeros(5, bool)]
        )
        self.assertEqual(epochs["intensity"].tolist(), [0.0, 7.0])
        self.assertEqual(epochs["wear_mask"].tolist(), [0, 1])

    def test_invalid_monitor_is_dropped(self):
        """Uncalibrated or unreliable monitors give no epochs."""
        self.assertIsNone(
            nhanes._participant_epochs(_minutes([5] * 10, paxcal=2), _all_worn)
        )
        self.assertIsNone(
            nhanes._participant_epochs(_minutes([5] * 10, paxstat=2), _all_worn)
        )

    def test_missing_minutes_count_as_not_worn(self):
        """Gaps in PAXN stay gaps on the minute grid."""
        minutes = _minutes([5] * 15)
        minutes = minutes[~minutes["PAXN"].isin([6, 7, 8, 9, 10])]
        epochs = nhanes._participant_epochs(minutes, _all_worn)
        self.assertEqual(len(epochs), 3)
        self.assertEqual(epochs["wear_minutes"].tolist(), [5, 0, 5])

    def test_clock_wraps_to_next_day(self):
        """Day, hour and minute follow the first record across midnight."""
        minutes = _minutes([1] * 10)
        minutes["PAXDAY"], minutes["PAXHOUR"] = 7, 23
        minutes["PAXMINUT"] = [55, 56, 57, 58, 59, 0, 1, 2, 3, 4]
        epochs = nhanes._participant_epochs(minutes, _all_worn)
        self.assertEqual(epochs["day"].tolist(), [6, 0])  # Saturday, Sunday
        self.assertEqual(epochs["hour"].tolist(), [23, 0])
        self.assertEqual(epochs["minute"].tolist(), [55, 0])

    def test_duplicate_minutes_raise(self):
        """Duplicate PAXN values are rejected."""
        minutes = pd.concat([_minutes([1] * 5), _minutes([1] * 5)])
        with self.assertRaises(ValueError):
            nhanes._participant_epochs(minutes, _all_worn)


class TestParticipantTable(unittest.TestCase):
    """Covariate and mortality parsing on tiny hand-written files."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        pd.DataFrame(
            {
                "SEQN": [1, 2, 3, 4],
                "RIDAGEYR": [40, 17, 70, 50],
                "RIAGENDR": [1, 2, 2, 1],
                "INDHHINC": [77, 3, 5, 99],
            }
        ).to_csv(root / "DEMO_C.csv", index=False)
        pd.DataFrame({"SEQN": [1], "BMXBMI": [27.3], "BMXWAIST": [90.0]}).to_csv(
            root / "BMX_C.csv", index=False
        )
        pd.DataFrame(
            {
                "SEQN": [1, 3],
                "BPXSY1": [120, 140],
                "BPXSY2": [122, np.nan],
                "BPXSY3": [np.nan, np.nan],
                "BPXSY4": [124, np.nan],
                "BPXDI1": [0, 0],
                "BPXDI2": [80, np.nan],
                "BPXDI3": [82, np.nan],
                "BPXDI4": [np.nan, np.nan],
            }
        ).to_csv(root / "BPX_C.csv", index=False)
        lines = [
            _mortality_line(1, 1, 0, 180, 179),
            _mortality_line(2, 2),
            _mortality_line(3, 1, 1, 40, 39),
            _mortality_line(4, 3),
        ]
        (root / "NHANES_2003_2004_MORT_2019_PUBLIC.dat").write_text("\n".join(lines))
        pd.DataFrame(columns=nhanes._PAXRAW_COLUMNS).to_csv(
            root / "PAXRAW_C.csv", index=False
        )
        files = nhanes._find_cycle_files(root)["C"]
        cls.table = nhanes._read_participants("C", files).set_index("SEQN")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_keeps_eligible_adults_only(self):
        """Children (ELIGSTAT 2) and ineligible adults (3) are dropped."""
        self.assertEqual(sorted(self.table.index), [1, 3])

    def test_mortality_fields(self):
        """MORTSTAT and follow-up months come from the fixed-width columns."""
        self.assertEqual(self.table.loc[3, "MORTSTAT"], 1)
        self.assertEqual(self.table.loc[3, "PERMTH_INT"], 40)
        self.assertEqual(self.table.loc[1, "PERMTH_EXM"], 179)

    def test_blood_pressure_means_ignore_zero_diastolic(self):
        """BP is the mean of available readings; diastolic 0 is missing."""
        self.assertEqual(self.table.loc[1, "sbp"], 122.0)
        self.assertEqual(self.table.loc[1, "dbp"], 81.0)
        self.assertTrue(np.isnan(self.table.loc[3, "dbp"]))

    def test_missing_codes_and_rows(self):
        """Income 77/99 and absent BMX rows become missing."""
        self.assertTrue(np.isnan(self.table.loc[1, "INDHHINC"]))
        self.assertEqual(self.table.loc[3, "INDHHINC"], 5)
        self.assertTrue(np.isnan(self.table.loc[3, "BMXBMI"]))
        self.assertEqual(self.table.loc[1, "cycle"], "2003-2004")


class TestFileDiscovery(unittest.TestCase):
    """Finding each cycle's files under root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def touch(self, *names: str) -> None:
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_finds_files_case_insensitively_preferring_xpt_over_zip(self):
        """Names match in any case; .xpt beats .zip in the same folder."""
        self.touch(
            "a/paxraw_c.xpt",
            "a/PAXRAW_C.zip",
            "a/DEMO_C.xpt",
            "a/BMX_C.xpt",
            "a/BPX_C.xpt",
            "m/NHANES_2003_2004_MORT_2019_PUBLIC.dat",
        )
        files = nhanes._find_cycle_files(self.root)
        self.assertEqual(list(files), ["C"])
        self.assertEqual(files["C"]["paxraw"].name, "paxraw_c.xpt")

    def test_incomplete_cycle_raises(self):
        """A cycle with DEMO but no BPX is an error."""
        self.touch("DEMO_C.csv", "BMX_C.csv", "PAXRAW_C.csv")
        with self.assertRaises(FileNotFoundError):
            nhanes._find_cycle_files(self.root)

    def test_no_cycle_raises(self):
        """No DEMO file at all is an error."""
        with self.assertRaises(FileNotFoundError):
            nhanes._find_cycle_files(self.root)

    def test_same_table_in_two_folders_raises(self):
        """Copies in different folders (e.g. real + synthetic) are ambiguous."""
        self.touch("real/DEMO_C.xpt", "synthetic/DEMO_C.csv")
        with self.assertRaises(ValueError):
            nhanes._find_cycle_files(self.root)


class TestNHANESDataset(unittest.TestCase):
    """End to end on the generator's five edge-case participants (2 days)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name) / "raw"
        cls.data = generate_synthetic_nhanes(
            str(root), n_participants=5, n_days=2, seed=0
        )
        cls.people = cls.data.participants.set_index("role")
        cls.paxraw = pd.read_csv(cls.data.paths["paxraw"])
        cls.dataset = NHANESDataset(
            root=str(root),
            nonwear="choi",
            cache_dir=str(Path(cls._tmp.name) / "cache"),
        )
        cls.table = pd.read_parquet(
            cls.dataset.prepared_dir / "nhanes-participants.parquet"
        ).set_index("seqn")
        # PyHealth's first load starts a Dask cluster (~1-2 s); do it once here.
        cls.patient_ids = cls.dataset.unique_patient_ids
        seqn = int(cls.people.loc["died_within_5y", "SEQN"])
        cls.event = cls.dataset.get_patient(str(seqn)).get_events(event_type="nhanes")[
            0
        ]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def seqn(self, role: str) -> int:
        return int(self.people.loc[role, "SEQN"])

    def test_cohort_is_eligible_adults_with_valid_monitor(self):
        """Child, not-examined and ineligible participants are excluded."""
        pax = self.paxraw
        valid = set(pax.loc[(pax.PAXSTAT == 1) & (pax.PAXCAL == 1), "SEQN"])
        p = self.data.participants
        keep = (p.eligstat == 1) & (p.age >= 18) & p.SEQN.isin(valid)
        expected = set(p.loc[keep, "SEQN"])
        self.assertEqual(set(self.table.index), expected)
        self.assertIn(self.seqn("died_within_5y"), expected)
        for role in ("child", "not_examined", "ineligible_missing_covariates"):
            self.assertNotIn(self.seqn(role), self.table.index)

    def test_participant_fields_match_generator(self):
        """Age, sex and mortality fields round-trip through the files."""
        people = self.data.participants.set_index("SEQN")
        for seqn, row in self.table.iterrows():
            self.assertEqual(row.age, people.loc[seqn, "age"])
            self.assertEqual(row.sex, people.loc[seqn, "sex"])
            self.assertEqual(row.mortstat, people.loc[seqn, "mortstat"])
            self.assertEqual(row.permth_int, people.loc[seqn, "permth_int"])

    def test_epoch_files(self):
        """2 days -> 576 epochs (432 when recording ends at noon), valid ranges."""
        expected = {
            self.seqn("died_within_5y"): 576,
            self.seqn("died_after_5y"): 432,
        }
        for seqn, n_epochs in expected.items():
            if seqn not in self.table.index:
                continue
            epochs = pd.read_parquet(self.table.loc[seqn, "epoch_file"])
            self.assertEqual(len(epochs), n_epochs)
            self.assertEqual(self.table.loc[seqn, "n_epochs"], n_epochs)
            self.assertEqual(
                list(epochs.columns),
                [
                    "epoch",
                    "day",
                    "hour",
                    "minute",
                    "intensity",
                    "wear_minutes",
                    "wear_mask",
                ],
            )
            self.assertTrue(epochs["day"].between(0, 6).all())
            self.assertTrue(epochs["hour"].between(0, 23).all())
            self.assertTrue(set(epochs["minute"]) <= set(range(0, 60, 5)))
            self.assertTrue(epochs["wear_minutes"].between(0, 5).all())

    def test_choi_epochs_match_wear_truth(self):
        """Worn minutes per epoch equal the generator's true wear minutes."""
        for seqn in self.table.index:
            truth = self.data.wear_truth[seqn].astype(int)
            truth = np.r_[truth, np.zeros(-truth.size % 5, int)].reshape(-1, 5).sum(1)
            epochs = pd.read_parquet(self.table.loc[seqn, "epoch_file"])
            np.testing.assert_array_equal(epochs["wear_minutes"].to_numpy(), truth)

    def test_planted_patterns(self):
        """Planted 180-min block: Choi flags it, the authors' rule does not."""
        rows = self.paxraw[self.paxraw.SEQN == self.seqn("died_within_5y")]
        counts = rows.sort_values("PAXN")["PAXINTEN"].to_numpy(float)
        start, end = PLANTED_NONWEAR
        self.assertTrue(nhanes._choi_nonwear(counts)[start:end].all())
        self.assertFalse(nhanes._authors_nonwear(counts)[start:end].any())
        start, end = PLANTED_SEDENTARY
        for detector in (nhanes._choi_nonwear, nhanes._authors_nonwear):
            self.assertFalse(detector(counts)[start:end].any())

    def test_patients_and_events(self):
        """PyHealth sees one patient per participant with readable attributes."""
        self.assertEqual(
            sorted(self.patient_ids), sorted(str(s) for s in self.table.index)
        )
        self.assertTrue(os.path.isfile(self.event.epoch_file))
        self.assertEqual(float(self.event.age), 72.0)
        self.assertEqual(int(float(self.event.mortstat)), 1)

    def test_task_on_dataset_patients(self):
        """The mortality task turns loaded patients into labeled windows."""
        task = NHANESMortalityPrediction(seq_len=288)  # 1-day windows
        labels = {}
        for patient_id in self.patient_ids:
            samples = task(self.dataset.get_patient(patient_id))
            self.assertLessEqual(len(samples), 2)  # 2 days of data
            for sample in samples:
                self.assertEqual(sample["intensity"].shape, (288, 1))
                labels[int(patient_id)] = sample["mortality"]
        self.assertEqual(labels.get(self.seqn("died_within_5y")), 1)
        if self.seqn("died_after_5y") in labels:
            self.assertEqual(labels[self.seqn("died_after_5y")], 0)


class TestPreparedFolders(unittest.TestCase):
    """Per-setting output folders, reuse, sharing, relocation and errors."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "raw"
        generate_synthetic_nhanes(str(self.root), n_participants=5, n_days=1, seed=0)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, root=None, **kwargs) -> NHANESDataset:
        return NHANESDataset(
            root=str(root or self.root), cache_dir=str(self.base / "cache"), **kwargs
        )

    def test_methods_get_separate_folders(self):
        """Default is 'authors'; building it leaves the 'choi' files alone."""
        choi = self.build(nonwear="choi")
        table = choi.prepared_dir / "nhanes-participants.parquet"
        mtime = table.stat().st_mtime
        authors = self.build()
        self.assertEqual(authors.nonwear, "authors")
        self.assertEqual(authors.prepared_dir.name, "authors")
        self.assertEqual(choi.prepared_dir.name, "choi")
        self.assertEqual(table.stat().st_mtime, mtime)

    def test_same_settings_reuse_files(self):
        """A second build with the same settings does not rewrite anything."""
        first = self.build(max_participants=1)
        self.assertEqual(first.prepared_dir.name, "authors-1")
        table = first.prepared_dir / "nhanes-participants.parquet"
        mtime = table.stat().st_mtime
        self.build(max_participants=1)
        self.assertEqual(table.stat().st_mtime, mtime)
        self.assertLessEqual(len(pd.read_parquet(table)), 1)

    def test_prepared_folder_works_alone_and_after_moving(self):
        """A copied prepared folder needs no raw files; its paths are fixed."""
        self.build()
        shared = self.base / "shared"
        shutil.copytree(
            self.root / "nhanes-prepared" / "authors",
            shared / "nhanes-prepared" / "authors",
        )
        dataset = self.build(root=shared)
        table = pd.read_parquet(dataset.prepared_dir / "nhanes-participants.parquet")
        for path in table["epoch_file"]:
            self.assertTrue(path.startswith(str(shared.resolve())))
            self.assertTrue(os.path.isfile(path))

    def test_two_cycles(self):
        """2003-2004 and 2005-2006 files under one root are both used."""
        generate_synthetic_nhanes(
            str(self.root / "d"), n_participants=5, n_days=1, cycle="2005_2006", seed=1
        )
        dataset = self.build()
        table = pd.read_parquet(dataset.prepared_dir / "nhanes-participants.parquet")
        self.assertEqual(set(table["cycle"]), {"2003-2004", "2005-2006"})

    def test_invalid_arguments(self):
        """Bad roots and settings raise clear errors."""
        with self.assertRaises(FileNotFoundError):
            self.build(root=self.base / "missing")
        empty = self.base / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.build(root=empty)
        with self.assertRaises(ValueError):
            self.build(nonwear="troiano")
        with self.assertRaises(ValueError):
            self.build(max_participants=0)


if __name__ == "__main__":
    unittest.main()
