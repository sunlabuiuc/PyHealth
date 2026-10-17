"""Tests for ClinicalPresenceMortalityMIMIC3.

The fixtures are small frames shaped like MIMIC3Dataset's ``global_event_df``
(string attributes, nanosecond timestamps), so no MIMIC data is needed:
``cohort_events`` has one patient per cohort rule, and ``random_events`` has
seeded random patients for the imputers to fit.
"""

import functools
import json
import os
import pickle
import tempfile
import unittest
import warnings
from datetime import datetime, timedelta

import dask.dataframe as dd
import numpy as np
import polars as pl
import torch
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from pyhealth.data import Patient
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.tasks import ClinicalPresenceMortalityMIMIC3
from pyhealth.tasks.clinical_presence_mortality_mimic3 import (
    GROUP_FLAGS,
    LAB_ITEMIDS,
    LAB_NAMES,
    _assemble,
    _assign_split,
    _extract_cohort,
    _feature_names,
    _impute_draws,
    _OLSResidualNoise,
    _training_range,
)

T0 = datetime(2150, 6, 1, 8, 0)
EARLY = T0 - timedelta(days=30)
FMT = "%Y-%m-%d %H:%M:%S"

SCHEMA = {
    "patient_id": pl.String,
    "event_type": pl.String,
    "timestamp": pl.Datetime("ns"),
    "patients/gender": pl.String,
    "patients/dob": pl.String,
    "admissions/hadm_id": pl.String,
    "admissions/dischtime": pl.String,
    "admissions/hospital_expire_flag": pl.String,
    "admissions/ethnicity": pl.String,
    "admissions/insurance": pl.String,
    "labevents/itemid": pl.String,
    "labevents/valuenum": pl.String,
}

# Itemids: both glucose ones map to "Glucose"; 50800 is outside the 67 labs
GLUCOSE_A, GLUCOSE_B = 50809, 50931
SODIUM, POTASSIUM, CHLORIDE, CREATININE, HEMOGLOBIN = 50983, 50971, 50902, 50912, 51222
UNMAPPED = 50800


def _patient(pid, age=60, gender="M"):
    return {
        "patient_id": pid,
        "event_type": "patients",
        "patients/gender": gender,
        "patients/dob": (T0 - timedelta(days=365 * age)).strftime(FMT),
    }


def _admission(
    pid,
    hadm,
    start=T0,
    stay_hours=72,
    died=False,
    ethnicity="WHITE",
    insurance="Medicare",
):
    return {
        "patient_id": pid,
        "event_type": "admissions",
        "timestamp": start,
        "admissions/hadm_id": hadm,
        "admissions/dischtime": (start + timedelta(hours=stay_hours)).strftime(FMT),
        "admissions/hospital_expire_flag": "1" if died else "0",
        "admissions/ethnicity": ethnicity,
        "admissions/insurance": insurance,
    }


def _lab(pid, hours, itemid, value, start=T0):
    return {
        "patient_id": pid,
        "event_type": "labevents",
        "timestamp": start + timedelta(hours=hours),
        "labevents/itemid": str(itemid),
        "labevents/valuenum": None if value is None else str(value),
    }


def _simple(pid, hadm, age=60, gender="M", **admission):
    """A patient with one admission and one sodium value at +2 h."""
    return [
        _patient(pid, age=age, gender=gender),
        _admission(pid, hadm, **admission),
        _lab(pid, 2, SODIUM, 140),
    ]


def cohort_events() -> pl.DataFrame:
    """One patient per cohort rule; comments give the expected outcome."""
    rows = [
        # 1: survives; orders at +2 h and +6 h, other rows don't count
        _patient("1"),
        _admission("1", "101"),
        _lab("1", 6, SODIUM, 138),  # listed first but later, so kept
        _lab("1", 2, SODIUM, 140),
        _lab("1", 2, POTASSIUM, 4.1),
        _lab("1", 7, UNMAPPED, 5),
        _lab("1", 8, CHLORIDE, None),
        _lab("1", 30, CREATININE, 1.2),  # outside the window
        # 2: dies on day 5 -> 1; 3: dies on day 9 -> 0; 14: day 8 exactly -> 0
        *_simple("2", "201", stay_hours=5 * 24, died=True),
        *_simple("3", "301", stay_hours=9 * 24, died=True),
        *_simple("14", "1401", stay_hours=8 * 24, died=True),
        # 4: aged 10 -> excluded
        *_simple("4", "401", age=10),
        # 5: last admission lasts 12 h -> excluded, though the first qualifies
        _patient("5"),
        _admission("5", "501", start=EARLY, stay_hours=48),
        _lab("5", 2, SODIUM, 139, start=EARLY),
        _admission("5", "502", stay_hours=12),
        _lab("5", 2, SODIUM, 137),
        # 6: two admissions with labs, the later listed first -> 602's labs only
        _patient("6"),
        _admission("6", "602"),
        _lab("6", 2, SODIUM, 145),
        _admission("6", "601", start=EARLY),
        _lab("6", 2, SODIUM, 130, start=EARLY),
        _lab("6", 3, HEMOGLOBIN, 9.0, start=EARLY),
        # 7: labs only at exactly +0 h and +24 h -> excluded
        _patient("7"),
        _admission("7", "701"),
        _lab("7", 0, SODIUM, 140),
        _lab("7", 24, SODIUM, 141),
        # 8: same-time glucose pair and a potassium with a null copy are dropped
        _patient("8"),
        _admission("8", "801"),
        _lab("8", 1, GLUCOSE_A, 90),
        _lab("8", 3, GLUCOSE_A, 100),
        _lab("8", 3, GLUCOSE_B, 200),
        _lab("8", 4, POTASSIUM, 4.0),
        _lab("8", 4, POTASSIUM, None),
        # 9: every lab value null -> excluded
        _patient("9"),
        _admission("9", "901"),
        _lab("9", 2, SODIUM, None),
        _lab("9", 3, POTASSIUM, None),
        # 10: DOB shifted ~300 years, as MIMIC does for patients over 89
        *_simple("10", "1001", age=300),
        # 11-13: group flags
        *_simple("11", "1101", ethnicity="BLACK/HAITIAN"),
        *_simple("12", "1201", insurance="Self Pay"),
        *_simple("13", "1301", gender="F", insurance="Private"),
        # 15: stays exactly 24 h -> included
        *_simple("15", "1501", stay_hours=24),
    ]
    return pl.DataFrame(rows, schema=SCHEMA)


# The random patients draw from these labs, from always to rarely observed
RANDOM_LABS = LAB_NAMES[:24]
FOREST = {"n_estimators": 5, "n_jobs": 1}
TASK_LOGGER = "pyhealth.tasks.clinical_presence_mortality_mimic3"


def random_events(n_patients: int = 200, seed: int = 0) -> pl.DataFrame:
    """Seeded random adults with correlated labs, plus a child outside the cohort.

    Every adult is in the cohort; 43 of the 67 labs are never observed.
    """
    rng = np.random.default_rng(seed)
    observe = np.linspace(1.0, 0.05, len(RANDOM_LABS))
    rows = _simple("999", "99901", age=10)
    for k in range(n_patients):
        pid = str(1000 + k)
        severity = rng.normal()
        rows += [
            _patient(
                pid,
                age=int(rng.integers(20, 90)),
                gender="F" if rng.random() < 0.5 else "M",
            ),
            _admission(
                pid,
                f"{pid}01",
                died=bool(rng.random() < 0.2),
                ethnicity="BLACK/AFRICAN AMERICAN" if rng.random() < 0.3 else "WHITE",
                insurance="Private" if rng.random() < 0.5 else "Medicare",
            ),
        ]
        for lab, p in zip(RANDOM_LABS, observe):
            if rng.random() < p:
                value = 50 + 10 * severity + 5 * rng.normal()
                hour = int(rng.integers(1, 23))
                rows.append(_lab(pid, hour, LAB_ITEMIDS[lab][0], round(value, 3)))
    return pl.DataFrame(rows, schema=SCHEMA)


@functools.cache
def random_cohort() -> dict:
    """The random fixture's labs, flags, a 60/20/20 split and two draw seeds."""
    events = random_events()
    cohort = _extract_cohort(events.lazy())
    seeds = np.random.SeedSequence(0).spawn(3)
    return {
        "events": events,
        "X": cohort.select(LAB_NAMES).to_numpy().astype(np.float64),
        "G": cohort.select(GROUP_FLAGS).to_numpy(),
        "split": _assign_split(cohort.height, (0.6, 0.2, 0.2), seeds[0]),
        "draw_seeds": seeds[1:],
    }


def impute(imputation: str, X: np.ndarray | None = None) -> np.ndarray:
    """Two fast draws of ``imputation`` on the random fixture, or on ``X``."""
    data = random_cohort()
    X = data["X"] if X is None else X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _impute_draws(
            X, data["G"], data["split"], imputation, data["draw_seeds"], 2, FOREST
        )


@functools.cache
def imputed(imputation: str) -> np.ndarray:
    return impute(imputation)


def fast_task(imputation: str = "mice", **kwargs) -> ClinicalPresenceMortalityMIMIC3:
    params = {
        "n_draws": 1 if imputation == "median" else 2,
        "max_iter": 2,
        "split_ratios": (0.6, 0.2, 0.2),
        "forest_params": FOREST,
        **kwargs,
    }
    return ClinicalPresenceMortalityMIMIC3(imputation=imputation, **params)


@functools.cache
def fitted_task(
    imputation: str = "mice", add_missing_indicators: bool = False
) -> ClinicalPresenceMortalityMIMIC3:
    task = fast_task(imputation, add_missing_indicators=add_missing_indicators)
    task.pre_filter(random_cohort()["events"].lazy())
    return task


def samples_for(task: ClinicalPresenceMortalityMIMIC3, pid: str) -> list[dict]:
    return task(Patient(pid, random_cohort()["events"].clear()))


def cache_key(task: ClinicalPresenceMortalityMIMIC3) -> str:
    """The task part of BaseDataset.set_task's cache key."""
    return json.dumps(
        {
            **vars(task),
            "input_schema": task.input_schema,
            "output_schema": task.output_schema,
        },
        sort_keys=True,
        default=str,
    )


class _EventsDataset(BaseDataset):
    """A BaseDataset whose event table is a given frame."""

    def __init__(self, events: pl.DataFrame, cache_dir: str):
        self.events = events
        super().__init__(root="", tables=[], dataset_name="Events", cache_dir=cache_dir)

    def load_data(self) -> dd.DataFrame:
        return dd.from_pandas(self.events.to_pandas(), npartitions=1)


class TestExtractCohort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = cohort_events()
        cls.cohort = _extract_cohort(cls.events.lazy())

    def row(self, pid):
        rows = self.cohort.filter(pl.col("patient_id") == pid).to_dicts()
        self.assertEqual(len(rows), 1, f"patient {pid} not in the cohort once")
        return rows[0]

    def test_cohort_membership(self):
        ids = self.cohort["patient_id"].to_list()
        self.assertEqual(
            ids, ["1", "2", "3", "6", "8", "10", "11", "12", "13", "14", "15"]
        )
        self.assertEqual(self.row("6")["hadm_id"], "602")
        self.assertEqual(
            self.cohort.columns,
            [
                "patient_id",
                "hadm_id",
                "mortality",
                *GROUP_FLAGS,
                "n_lab_orders",
                *LAB_NAMES,
            ],
        )

    def test_label_horizon(self):
        labels = dict(self.cohort.select("patient_id", "mortality").iter_rows())
        self.assertEqual(labels["1"], 0)
        self.assertEqual(labels["2"], 1)
        self.assertEqual(labels["3"], 0)
        self.assertEqual(labels["14"], 0)
        longer = _extract_cohort(self.events.lazy(), horizon_days=10)
        self.assertEqual(
            longer.filter(pl.col("patient_id") == "3")["mortality"].item(), 1
        )

    def test_window_dedup_latest(self):
        p1, p6, p8 = self.row("1"), self.row("6"), self.row("8")
        self.assertEqual(p1["Sodium"], 138.0)
        self.assertEqual(p1["Potassium"], 4.1)
        self.assertIsNone(p1["Creatinine"])
        self.assertIsNone(p1["Chloride"])
        self.assertEqual(p6["Sodium"], 145.0)
        self.assertIsNone(p6["Hemoglobin"])
        self.assertEqual(p8["Glucose"], 90.0)
        self.assertIsNone(p8["Potassium"])
        observed = {lab for lab in LAB_NAMES if p1[lab] is not None}
        self.assertEqual(observed, {"Sodium", "Potassium"})

    def test_group_flags(self):
        flags = {
            pid: tuple(self.row(pid)[g] for g in GROUP_FLAGS)
            for pid in ("1", "11", "12", "13")
        }
        self.assertEqual(flags["1"], (0, 0, 1))
        self.assertEqual(flags["11"], (1, 0, 1))
        self.assertEqual(flags["12"], (0, 0, 1))
        self.assertEqual(flags["13"], (0, 1, 0))

    def test_lab_order_counts(self):
        self.assertEqual(self.row("1")["n_lab_orders"], 2)
        self.assertEqual(self.row("6")["n_lab_orders"], 1)
        self.assertEqual(self.row("8")["n_lab_orders"], 1)

    def test_missing_labevents_raises(self):
        no_labs = self.events.drop("labevents/itemid", "labevents/valuenum")
        with self.assertRaisesRegex(ValueError, r"tables=\['labevents'\]"):
            _extract_cohort(no_labs.lazy())


class TestSplit(unittest.TestCase):
    def test_split_deterministic_and_shared(self):
        def split(n, ratios, n_children=1):
            return _assign_split(
                n, ratios, np.random.SeedSequence(0).spawn(n_children)[0]
            )

        a = split(2670, (0.6, 0.1, 0.3))
        np.testing.assert_array_equal(a, split(2670, (0.6, 0.1, 0.3)))
        np.testing.assert_array_equal(a, split(2670, (0.6, 0.1, 0.3), n_children=11))
        for n, ratios, sizes in [
            (2670, (0.6, 0.1, 0.3), (1602, 267, 801)),
            (36313, (0.8, 0.1, 0.1), (29050, 3632, 3631)),
        ]:
            counts = [
                int((split(n, ratios) == p).sum()) for p in ("train", "val", "test")
            ]
            self.assertEqual(tuple(counts), sizes)

        median = [row["split"] for row in fitted_task("median")._state.fields]
        mice = [row["split"] for row in fitted_task("mice", True)._state.fields]
        self.assertEqual(median, mice)
        self.assertEqual(median, random_cohort()["split"].tolist())


class TestImputation(unittest.TestCase):
    """The imputers and the feature assembly, on the random fixture."""

    @classmethod
    def setUpClass(cls):
        data = random_cohort()
        cls.X, cls.split = data["X"], data["split"]
        cls.missing = np.isnan(cls.X)
        cls.train = cls.split == "train"

    def test_no_nan_all_strategies(self):
        for imputation in ClinicalPresenceMortalityMIMIC3.IMPUTATIONS:
            for indicators in (False, True):
                with self.subTest(imputation=imputation, indicators=indicators):
                    F = _assemble(
                        imputed(imputation), self.X, self.split, indicators, True
                    )
                    self.assertTrue(np.isfinite(F).all())

    def test_observed_values_preserved(self):
        observed = self.X[~self.missing].astype(np.float32)
        for imputation in ClinicalPresenceMortalityMIMIC3.IMPUTATIONS:
            with self.subTest(imputation=imputation):
                F = _assemble(imputed(imputation), self.X, self.split, False, False)
                for draw in F:
                    np.testing.assert_array_equal(draw[~self.missing], observed)

    def test_train_rows_independent_of_eval_rows(self):
        rng = np.random.default_rng(1)
        perturbed = self.X.copy()
        perturbed[~self.train] += 1.0
        hide = ~self.train[:, None] & (rng.random(self.X.shape) < 0.3)
        perturbed[hide] = np.nan
        for imputation in ClinicalPresenceMortalityMIMIC3.IMPUTATIONS:
            with self.subTest(imputation=imputation):
                np.testing.assert_array_equal(
                    impute(imputation, perturbed)[:, self.train],
                    imputed(imputation)[:, self.train],
                )

    def test_draws_reproducible_and_distinct(self):
        for imputation in ("mice", "group_mice", "corrected_mice"):
            with self.subTest(imputation=imputation):
                again = impute(imputation)
                np.testing.assert_array_equal(again, imputed(imputation))
                self.assertFalse(np.array_equal(again[0], again[1]))
        with self.assertRaisesRegex(ValueError, "deterministic"):
            ClinicalPresenceMortalityMIMIC3(imputation="median", n_draws=2)

    def test_mice_missing_shares_draws(self):
        plain = fitted_task("mice")._state.features
        with_indicators = fitted_task("mice", True)._state.features
        n_labs = len(LAB_NAMES)
        np.testing.assert_array_equal(plain, with_indicators[:, :, :n_labs])

    def test_group_flags_never_features(self):
        for imputation in ClinicalPresenceMortalityMIMIC3.IMPUTATIONS:
            self.assertEqual(imputed(imputation).shape[2], len(LAB_NAMES))
            for indicators, width in ((False, 67), (True, 134)):
                with self.subTest(imputation=imputation, indicators=indicators):
                    F = _assemble(
                        imputed(imputation), self.X, self.split, indicators, True
                    )
                    self.assertEqual(F.shape[2], width)
                    self.assertEqual(len(_feature_names(indicators)), width)
        names = fast_task(add_missing_indicators=True).feature_names()
        self.assertEqual(names[:67], LAB_NAMES)
        self.assertEqual(names[67], f"{LAB_NAMES[0]} (missing)")

    def test_missing_indicators(self):
        F = _assemble(imputed("mice"), self.X, self.split, True, False)
        for draw in F:
            np.testing.assert_array_equal(draw[:, 67:], self.missing)

    def test_standardization_train_stats(self):
        F = _assemble(imputed("mice"), self.X, self.split, True, True)
        train_rows = F[:, self.train].astype(np.float64)
        np.testing.assert_allclose(train_rows.mean(axis=1), 0, atol=1e-5)
        sd = train_rows.std(axis=1)
        constant = np.ptp(train_rows, axis=1) == 0
        self.assertTrue(constant.any())
        np.testing.assert_allclose(sd[~constant], 1, atol=1e-4)
        self.assertTrue((train_rows.transpose(0, 2, 1)[constant] == 0).all())

    def test_corrected_mice_bounded(self):
        lo, hi = _training_range(self.X[self.train])
        lo32 = np.broadcast_to(lo.astype(np.float32), self.X.shape)
        hi32 = np.broadcast_to(hi.astype(np.float32), self.X.shape)
        observed = self.X[~self.missing].astype(np.float32)
        for imputation in ("corrected_mice", "group_corrected_mice"):
            with self.subTest(imputation=imputation):
                for draw in imputed(imputation):
                    cells = draw[self.missing]
                    self.assertTrue((cells >= lo32[self.missing]).all())
                    self.assertTrue((cells <= hi32[self.missing]).all())
                    np.testing.assert_array_equal(draw[~self.missing], observed)

        Z = np.array([[1.0, np.nan, 5.0], [3.0, np.nan, 5.0], [np.nan, np.nan, 5.0]])
        lo, hi = _training_range(Z)
        np.testing.assert_array_equal(lo, [1.0, -np.inf, -np.inf])
        np.testing.assert_array_equal(hi, [3.0, np.inf, np.inf])

        data = random_cohort()
        X = self.X.copy()
        X[~self.missing[:, 10], 10] = 5.0
        G = data["G"].copy()
        G[:, 0] = 1
        out = _impute_draws(
            X, G, self.split, "group_corrected_mice", data["draw_seeds"], 2, {}
        )
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(self.missing[:, 10].any())
        np.testing.assert_array_equal(out[:, :, 10], 5.0)

    def test_ols_residual_noise_options(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 5))
        y = X @ np.arange(5.0) + rng.normal(size=30)

        faithful = _OLSResidualNoise().fit(X, y)
        resid = y - faithful.predict(X)
        self.assertTrue(np.any(faithful.coef_))
        self.assertAlmostEqual(faithful.sigma_, np.std(np.abs(resid)))

        sparse = _OLSResidualNoise(min_rows_per_feature=10).fit(X, y)
        np.testing.assert_array_equal(sparse.coef_, 0)
        self.assertAlmostEqual(sparse.intercept_, y.mean())
        self.assertAlmostEqual(sparse.sigma_, np.std(np.abs(y - y.mean())))

        proper = _OLSResidualNoise(min_rows_per_feature=6, proper_noise=True)
        proper.fit(X, y)
        np.testing.assert_allclose(proper.coef_, faithful.coef_)
        self.assertAlmostEqual(proper.sigma_, np.std(resid))

        params = clone(proper).get_params()
        self.assertEqual(params, {"min_rows_per_feature": 6, "proper_noise": True})


class TestTask(unittest.TestCase):
    """The task class, through pre_filter and __call__."""

    def test_sample_schema(self):
        task = fitted_task("mice", True)
        types = {
            "patient_id": str,
            "hadm_id": str,
            "mortality": int,
            "split": str,
            "draw": int,
            **{flag: int for flag in GROUP_FLAGS},
            "n_lab_orders": int,
            "n_observed_labs": int,
            "n_extreme_imputations": int,
            "labs": np.ndarray,
        }
        for pid in ("1000", "1001", "1199"):
            samples = samples_for(task, pid)
            self.assertEqual([s["draw"] for s in samples], [0, 1])
            for sample in samples:
                self.assertEqual(set(sample), set(types))
                for key, kind in types.items():
                    self.assertIsInstance(sample[key], kind, key)
                self.assertEqual(sample["patient_id"], pid)
                self.assertEqual(sample["labs"].dtype, np.float32)
                self.assertEqual(sample["labs"].shape, (134,))
                self.assertIn(sample["split"], ("train", "val", "test"))
                self.assertIn(sample["mortality"], (0, 1))
                self.assertGreater(sample["n_observed_labs"], 0)
        self.assertEqual(samples_for(task, "999"), [])
        with self.assertRaisesRegex(RuntimeError, "set_task"):
            samples_for(fast_task(), "1000")

    def test_state_survives_pickle(self):
        task = fitted_task("mice", True)
        restored = pickle.loads(pickle.dumps(task))
        for pid in ("1000", "1001"):
            for a, b in zip(samples_for(task, pid), samples_for(restored, pid)):
                np.testing.assert_array_equal(a.pop("labs"), b.pop("labs"))
                self.assertEqual(a, b)

    def test_cache_key_stable(self):
        task = fast_task("group_mice", add_missing_indicators=True)
        before = cache_key(task)
        kept = task.pre_filter(random_cohort()["events"].lazy()).collect()
        self.assertEqual(cache_key(task), before)
        self.assertEqual(set(kept["event_type"]), {"patients"})
        self.assertEqual(kept.height, 200)
        default = ClinicalPresenceMortalityMIMIC3(imputation="mice")
        explicit = ClinicalPresenceMortalityMIMIC3(imputation="mice", n_draws=10)
        self.assertEqual(cache_key(default), cache_key(explicit))

    def test_invalid_args(self):
        cases = [
            ({"imputation": "mean"}, "imputation must be one of"),
            ({"n_draws": 0}, "n_draws and max_iter must be at least 1"),
            ({"max_iter": 0}, "n_draws and max_iter must be at least 1"),
            ({"imputation": "median", "n_draws": 2}, "deterministic"),
            ({"split_ratios": (0.8, 0.2)}, "three positive numbers"),
            ({"split_ratios": (0.9, 0.2, -0.1)}, "three positive numbers"),
            ({"split_ratios": (0.5, 0.2, 0.2)}, "summing to 1"),
        ]
        for kwargs, message in cases:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                ClinicalPresenceMortalityMIMIC3(**kwargs)

    def test_logs_aggregates(self):
        task = fast_task("missforest")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs(TASK_LOGGER, "INFO") as logs:
                task.pre_filter(random_cohort()["events"].lazy())
        self.assertFalse([w for w in caught if w.category is ConvergenceWarning])
        output = "\n".join(logs.output)
        for line in (
            "Cohort: 200 patients",
            "Split (seed 0): 120 train / 40 val / 40 test.",
            "Imputation 'missforest': 2 draw(s)",
            "The imputer ran all 2 sweeps without meeting scikit-learn's tol",
            "Imputed labs beyond the training range ± 3 sd: 0.000%",
        ):
            self.assertIn(line, output)


class TestSetTask(unittest.TestCase):
    """End to end through BaseDataset.set_task, on a temporary cache."""

    def run_set_task(self, num_workers: int) -> dict[tuple[str, int], dict]:
        with tempfile.TemporaryDirectory() as cache_dir:
            dataset = _EventsDataset(random_cohort()["events"], cache_dir)
            task = fast_task("mice", add_missing_indicators=True)
            samples = dataset.set_task(task, num_workers=num_workers)
            return {(s["patient_id"], s["draw"]): s for s in samples}

    def test_set_task_end_to_end(self):
        samples = self.run_set_task(num_workers=1)
        self.assertEqual(len(samples), 200 * 2)
        sample = samples[("1000", 0)]
        self.assertIsInstance(sample["labs"], torch.Tensor)
        self.assertEqual(tuple(sample["labs"].shape), (134,))
        for key in ("hadm_id", "split", *GROUP_FLAGS, "n_extreme_imputations"):
            self.assertIn(key, sample)
        splits = [s["split"] for (_, draw), s in samples.items() if draw == 0]
        self.assertEqual(splits.count("test"), 40)

    @unittest.skipUnless(
        os.environ.get("PYHEALTH_RUN_SLOW") == "1",
        "spawn start-up takes about 25 s; set PYHEALTH_RUN_SLOW=1",
    )
    def test_set_task_multiworker_parity(self):
        one, two = self.run_set_task(1), self.run_set_task(2)
        self.assertEqual(one.keys(), two.keys())
        for key, sample in one.items():
            torch.testing.assert_close(sample["labs"], two[key]["labs"])
            self.assertEqual(sample["split"], two[key]["split"])


if __name__ == "__main__":
    unittest.main()
