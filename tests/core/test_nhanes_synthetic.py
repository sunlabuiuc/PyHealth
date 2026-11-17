"""Unit tests for the synthetic NHANES generator used by the MotionAge tests."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pyhealth.datasets import nhanes_synthetic
from pyhealth.datasets.nhanes_synthetic import (
    MAX_INTENSITY,
    MAX_STEPS,
    PAXRAW_COLUMNS,
    PLANTED_BOUNDED_NONWEAR,
    PLANTED_LONE_SPIKE,
    PLANTED_NONWEAR,
    PLANTED_SEDENTARY,
    PLANTED_SPIKE,
    generate_synthetic_nhanes,
    main,
)

# NCHS public-use linked mortality file layout, as pandas colspecs.
MORT_COLSPECS = [
    (0, 6),
    (14, 15),
    (15, 16),
    (16, 19),
    (19, 20),
    (20, 21),
    (42, 45),
    (45, 48),
]
MORT_NAMES = [
    "SEQN",
    "ELIGSTAT",
    "MORTSTAT",
    "UCOD_LEADING",
    "DIABETES",
    "HYPERTEN",
    "PERMTH_INT",
    "PERMTH_EXM",
]


def read_mortality(path: str) -> pd.DataFrame:
    """Parses the mortality file the way the real NHANES loader will."""
    return pd.read_fwf(path, colspecs=MORT_COLSPECS, names=MORT_NAMES, na_values=["."])


def zero_runs(values: np.ndarray) -> list:
    """Returns (start, end) pairs of consecutive zeros, end exclusive."""
    padded = np.concatenate(([False], values == 0, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


class TestSyntheticNHANES(unittest.TestCase):
    """Tests on one 5-participant, 1-day dataset holding the edge cases."""

    @classmethod
    def setUpClass(cls):
        """Generates the dataset once into a temporary directory."""
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data = generate_synthetic_nhanes(
            cls._tmp.name, n_participants=5, n_days=1, seed=0
        )
        cls.people = cls.data.participants.set_index("role")
        cls.paxraw = pd.read_csv(cls.data.paths["paxraw"])
        cls.mortality = read_mortality(cls.data.paths["mortality"])

    @classmethod
    def tearDownClass(cls):
        """Removes the temporary directory."""
        cls._tmp.cleanup()

    def seqn(self, role: str) -> int:
        """Returns the SEQN of an edge-case participant."""
        return int(self.people.loc[role, "SEQN"])

    def intensity(self, role: str) -> np.ndarray:
        """Returns the minute-level PAXINTEN array of a participant."""
        rows = self.paxraw[self.paxraw["SEQN"] == self.seqn(role)]
        return rows.sort_values("PAXN")["PAXINTEN"].to_numpy()

    def test_writes_cycle_files(self):
        """All five files exist with the 2003-2004 NHANES names."""
        names = {key: Path(path).name for key, path in self.data.paths.items()}
        self.assertEqual(
            names,
            {
                "paxraw": "PAXRAW_C.csv",
                "demo": "DEMO_C.csv",
                "bmx": "BMX_C.csv",
                "bpx": "BPX_C.csv",
                "mortality": "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
            },
        )
        for path in self.data.paths.values():
            self.assertTrue(Path(path).is_file())

    def test_edge_case_roles_come_first(self):
        """The first five participants are the edge cases, in order."""
        self.assertEqual(
            self.data.participants["role"].tolist(),
            [
                "died_within_5y",
                "died_after_5y",
                "child",
                "not_examined",
                "ineligible_missing_covariates",
            ],
        )
        self.assertEqual(self.data.participants["SEQN"].iloc[0], 21005)

    def test_paxraw_columns_and_ranges(self):
        """PAXRAW uses the real column names and value ranges."""
        self.assertEqual(
            list(self.paxraw.columns),
            [
                "SEQN",
                "PAXSTAT",
                "PAXCAL",
                "PAXDAY",
                "PAXN",
                "PAXHOUR",
                "PAXMINUT",
                "PAXINTEN",
            ],
        )
        self.assertTrue(self.paxraw["PAXINTEN"].between(0, MAX_INTENSITY).all())
        self.assertTrue(self.paxraw["PAXDAY"].between(1, 7).all())
        self.assertTrue(self.paxraw["PAXHOUR"].between(0, 23).all())
        self.assertTrue(self.paxraw["PAXMINUT"].between(0, 59).all())
        for seqn, rows in self.paxraw.groupby("SEQN"):
            self.assertEqual(rows["PAXN"].tolist(), list(range(1, len(rows) + 1)))
            self.assertEqual(len(rows), len(self.data.wear_truth[seqn]))

    def test_accelerometer_coverage(self):
        """The child has PAXRAW; the not-examined and ELIGSTAT=3 cases do not."""
        self.assertEqual(
            set(self.paxraw["SEQN"]),
            {self.seqn(r) for r in ("died_within_5y", "died_after_5y", "child")},
        )
        self.assertEqual(len(self.intensity("died_within_5y")), 1440)
        self.assertEqual(len(self.intensity("died_after_5y")), 720)

    def test_mortality_fixed_width_layout(self):
        """Every mortality line fits the NCHS layout; SEQN fills cols 1-14."""
        with open(self.data.paths["mortality"]) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), 5)
        for line, seqn in zip(lines, self.data.participants["SEQN"]):
            self.assertLessEqual(len(line), 48)
            self.assertEqual(line[:14], str(seqn).ljust(14))
            self.assertEqual(line[21:42].strip(), "")
        self.assertEqual(
            self.mortality["SEQN"].tolist(),
            self.data.participants["SEQN"].tolist(),
        )

    def test_mortality_codes_and_missing_values(self):
        """Edge cases have the right codes; missing values parse as NaN."""
        mort = self.mortality.set_index("SEQN")
        died_after = mort.loc[self.seqn("died_after_5y")]
        self.assertEqual(died_after["MORTSTAT"], 1)
        self.assertEqual(died_after["PERMTH_EXM"], 98)

        child = mort.loc[self.seqn("child")]
        self.assertEqual(child["ELIGSTAT"], 2)
        self.assertTrue(child[["MORTSTAT", "PERMTH_INT", "PERMTH_EXM"]].isna().all())

        not_examined = mort.loc[self.seqn("not_examined")]
        self.assertEqual(not_examined["ELIGSTAT"], 1)
        self.assertFalse(pd.isna(not_examined["PERMTH_INT"]))
        self.assertTrue(pd.isna(not_examined["PERMTH_EXM"]))

        self.assertEqual(
            mort.loc[self.seqn("ineligible_missing_covariates"), "ELIGSTAT"], 3
        )

    def test_labels_match_mortality_file(self):
        """Labels from the parsed file, by the task's PERMTH_INT rule, equal
        the returned labels; the interview-only participant is labeled too."""
        mort = self.mortality
        dead = mort["MORTSTAT"] == 1
        known = (mort["ELIGSTAT"] == 1) & (dead | (mort["PERMTH_INT"] >= 60))
        label = (dead & (mort["PERMTH_INT"] <= 60)).astype("Int64")
        label[~known] = pd.NA
        truth = self.data.participants["died_within_5y"]
        self.assertEqual(label.tolist(), truth.tolist())
        self.assertEqual(truth.tolist()[:2], [1, 0])
        self.assertFalse(
            pd.isna(truth[self.data.participants["role"] == "not_examined"].iloc[0])
        )

    def test_labels_match_task(self):
        """Every labeled participant with accelerometer data gets the same
        label from NHANESMortalityPrediction's rule."""
        from types import SimpleNamespace

        from pyhealth.tasks.nhanes_mortality_prediction import (
            NHANESMortalityPrediction,
        )

        task = NHANESMortalityPrediction()
        with tempfile.TemporaryDirectory() as tmp:
            data = generate_synthetic_nhanes(tmp, 300, n_days=1)
        for person in data.participants.itertuples():
            if person.eligstat != 1:
                continue
            event = SimpleNamespace(
                mortstat=person.mortstat, permth_int=person.permth_int
            )
            expected = task._label(event)
            truth = None if pd.isna(person.died_within_5y) else person.died_within_5y
            self.assertEqual(truth, expected, person.SEQN)

    def test_covariates_for_ineligible_participant(self):
        """The ineligible participant refused income and has no BMX/BPX."""
        demo = pd.read_csv(self.data.paths["demo"]).set_index("SEQN")
        bmx = pd.read_csv(self.data.paths["bmx"])
        bpx = pd.read_csv(self.data.paths["bpx"])
        seqn = self.seqn("ineligible_missing_covariates")
        self.assertEqual(demo.loc[seqn, "INDHHINC"], 77)
        self.assertNotIn(seqn, set(bmx["SEQN"]))
        self.assertNotIn(seqn, set(bpx["SEQN"]))
        self.assertNotIn(self.seqn("not_examined"), set(bmx["SEQN"]))
        self.assertEqual(len(demo), 5)

    def test_planted_nonwear_block_with_spike(self):
        """The 180-minute block is all zeros except a 2-minute spike."""
        intensity = self.intensity("died_within_5y")
        wear = self.data.wear_truth[self.seqn("died_within_5y")]
        nw_start, nw_end = PLANTED_NONWEAR
        sp_start, sp_end = PLANTED_SPIKE
        self.assertFalse(wear[nw_start:nw_end].any())
        self.assertEqual(intensity[sp_start:sp_end].tolist(), [40, 60])
        self.assertTrue((intensity[nw_start:sp_start] == 0).all())
        self.assertTrue((intensity[sp_end:nw_end] == 0).all())
        # Worn minutes on either side are nonzero, so the block stays separate.
        self.assertGreater(intensity[nw_start - 1], 0)
        self.assertGreater(intensity[nw_end], 0)

    def test_planted_sedentary_run_is_worn(self):
        """The 60-minute zero run is marked as worn and bounded by activity."""
        intensity = self.intensity("died_within_5y")
        wear = self.data.wear_truth[self.seqn("died_within_5y")]
        start, end = PLANTED_SEDENTARY
        self.assertTrue(wear[start:end].all())
        self.assertTrue((intensity[start:end] == 0).all())
        self.assertEqual(intensity[start - 1], 50)
        self.assertEqual(intensity[end], 50)

    def test_planted_bounded_nonwear_block(self):
        """The 120-minute block is zeros plus one spike, between counts of 150."""
        intensity = self.intensity("died_within_5y")
        wear = self.data.wear_truth[self.seqn("died_within_5y")]
        start, end = PLANTED_BOUNDED_NONWEAR
        self.assertFalse(wear[start:end].any())
        self.assertTrue(wear[start - 1] and wear[end])
        self.assertEqual(intensity[PLANTED_LONE_SPIKE], 50)
        block = np.delete(intensity[start:end], PLANTED_LONE_SPIKE - start)
        self.assertTrue((block == 0).all())
        self.assertEqual(intensity[start - 1], 150)
        self.assertEqual(intensity[end], 150)

    def test_wear_truth_is_consistent_with_counts(self):
        """Non-wear minutes are zero (except the spikes) and runs last >= 90."""
        planted = self.seqn("died_within_5y")
        for seqn, wear in self.data.wear_truth.items():
            rows = self.paxraw[self.paxraw["SEQN"] == seqn]
            intensity = rows["PAXINTEN"].to_numpy()
            spike = np.zeros_like(wear)
            if seqn == planted:
                spike[slice(*PLANTED_SPIKE)] = True
                spike[PLANTED_LONE_SPIKE] = True
            self.assertTrue((intensity[~wear & ~spike] == 0).all())
            # Zero runs of the wear mask are the non-wear stretches.
            for start, end in zero_runs(wear.astype(int)):
                self.assertGreaterEqual(end - start, 90)
            # Zero runs fully inside worn time stay under the Choi threshold.
            for start, end in zero_runs(np.where(wear, intensity, -1)):
                self.assertLess(end - start, 90)


class TestSyntheticNHANESOptions(unittest.TestCase):
    """Tests for seeds, cycles and argument validation."""

    def setUp(self):
        """Creates a temporary directory."""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        """Removes the temporary directory."""
        self._tmp.cleanup()

    def test_same_seed_gives_identical_files(self):
        """Two runs with the same seed write byte-identical files."""
        a = generate_synthetic_nhanes(str(self.root / "a"), 5, n_days=1, seed=3)
        b = generate_synthetic_nhanes(str(self.root / "b"), 5, n_days=1, seed=3)
        for key in a.paths:
            self.assertEqual(
                Path(a.paths[key]).read_bytes(),
                Path(b.paths[key]).read_bytes(),
                key,
            )
        for seqn, wear in a.wear_truth.items():
            np.testing.assert_array_equal(wear, b.wear_truth[seqn])

    def test_different_seed_gives_different_counts(self):
        """A different seed changes the accelerometer counts."""
        a = generate_synthetic_nhanes(str(self.root / "a"), 5, n_days=1, seed=0)
        b = generate_synthetic_nhanes(str(self.root / "b"), 5, n_days=1, seed=1)
        self.assertNotEqual(
            Path(a.paths["paxraw"]).read_bytes(),
            Path(b.paths["paxraw"]).read_bytes(),
        )

    def test_cycle_2005_2006(self):
        """2005-2006 uses suffix D, its SEQN range, SDDSRVYR 4 and PAXSTEP."""
        data = generate_synthetic_nhanes(
            str(self.root / "d"), 5, n_days=1, cycle="2005_2006", seed=0
        )
        names = sorted(Path(path).name for path in data.paths.values())
        self.assertEqual(
            names,
            [
                "BMX_D.csv",
                "BPX_D.csv",
                "DEMO_D.csv",
                "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
                "PAXRAW_D.csv",
            ],
        )
        for path in data.paths.values():
            self.assertTrue(Path(path).is_file())
        demo = pd.read_csv(data.paths["demo"])
        self.assertEqual(demo["SEQN"].tolist(), list(range(31127, 31132)))
        self.assertTrue((demo["SDDSRVYR"] == 4).all())
        mortality = read_mortality(data.paths["mortality"])
        self.assertEqual(mortality["SEQN"].tolist(), demo["SEQN"].tolist())
        self.assertEqual(data.participants["died_within_5y"].tolist()[:2], [1, 0])

        # PAXRAW_D has a 9th column, PAXSTEP; PAXRAW_C does not.
        paxraw_d = pd.read_csv(data.paths["paxraw"])
        self.assertEqual(list(paxraw_d.columns)[-1], "PAXSTEP")
        self.assertEqual(len(paxraw_d.columns), 9)
        c_data = generate_synthetic_nhanes(str(self.root / "c"), 5, n_days=1)
        paxraw_c = pd.read_csv(c_data.paths["paxraw"])
        self.assertNotIn("PAXSTEP", paxraw_c.columns)
        self.assertEqual(len(paxraw_c.columns), 8)

        # Steps are 0 exactly when intensity is 0, else 1..MAX_STEPS, and
        # never decrease as intensity rises.
        steps, intensity = paxraw_d["PAXSTEP"], paxraw_d["PAXINTEN"]
        self.assertTrue(((steps == 0) == (intensity == 0)).all())
        self.assertTrue(steps.between(0, MAX_STEPS).all())
        by_intensity = paxraw_d.sort_values(["PAXINTEN", "PAXSTEP"])["PAXSTEP"]
        self.assertTrue(by_intensity.is_monotonic_increasing)
        self.assertEqual(steps.max(), MAX_STEPS)

    def test_no_accelerometer_data_writes_empty_paxraw(self):
        """With no device wearers, PAXRAW is written with headers only."""
        real = nhanes_synthetic._make_participants

        def nobody_wears_device(*args, **kwargs):
            participants = real(*args, **kwargs)
            participants["has_accelerometer"] = False
            return participants

        with mock.patch.object(
            nhanes_synthetic, "_make_participants", nobody_wears_device
        ):
            data = generate_synthetic_nhanes(str(self.root), 2, n_days=1)
        paxraw = pd.read_csv(data.paths["paxraw"])
        self.assertEqual(list(paxraw.columns), list(PAXRAW_COLUMNS))
        self.assertEqual(len(paxraw), 0)
        self.assertEqual(data.wear_truth, {})

    def test_without_edge_cases(self):
        """With edge cases off, every participant is randomly generated."""
        data = generate_synthetic_nhanes(
            str(self.root), 4, n_days=1, seed=0, include_edge_cases=False
        )
        self.assertEqual(set(data.participants["role"]), {"random"})

    def test_main_writes_dataset(self):
        """The command-line entry point writes the files and a summary."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main([str(self.root), "--n-participants", "5", "--n-days", "1"])
        self.assertTrue((self.root / "PAXRAW_C.csv").is_file())
        self.assertIn("5-year deaths: 1 of 3 labeled", out.getvalue())

    def test_invalid_arguments(self):
        """Unknown cycles and out-of-range counts raise ValueError."""
        with self.assertRaises(ValueError):
            generate_synthetic_nhanes(str(self.root), 2, cycle="1999_2000")
        for n_participants in (0, -1):
            with self.assertRaises(ValueError):
                generate_synthetic_nhanes(str(self.root), n_participants)
        for n_days in (0, 8):
            with self.assertRaises(ValueError):
                generate_synthetic_nhanes(str(self.root), 2, n_days=n_days)


if __name__ == "__main__":
    unittest.main()
