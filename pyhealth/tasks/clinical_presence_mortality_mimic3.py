"""
PyHealth task for studying how imputation affects algorithmic fairness on
MIMIC-III, when which lab tests are missing reflects clinical presence: how
patients interact with the healthcare system.

The cohort is each adult's last hospital admission, kept if it lasts at least
24 hours. Features are the latest value of each of 67 lab tests drawn in the
first 24 hours, with missing values imputed at the cohort level (median, MICE,
group MICE, a corrected version of either, or MissForest-style; fit on training
patients only). The label is
death within 8 days of admission. Black, female and public-insurance flags
travel with each sample for subgroup evaluation, but are never features.

Dataset link:
    https://physionet.org/content/mimiciii/1.4/

Task paper: (please cite if you use this task)
    Vincent Jeanselme, Maria De-Arteaga, Zhe Zhang, Jessica Barrett and Brian
    Tom. "Imputation Strategies Under Clinical Presence: Impact on Algorithmic
    Fairness." Machine Learning for Health (ML4H) 2022, PMLR 193:12-34.

Paper link:
    https://proceedings.mlr.press/v193/jeanselme22a.html

Reference code:
    https://github.com/Jeanselme/ClinicalPresenceFairness

Authors:
    Chunyi Yu (cyu410@gatech.edu)
    John Nguyen (jnguyen461@gatech.edu)
"""

import logging
import math
import time
import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from pyhealth.data import Patient

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Copy mapping from Jeanselme's preprocessing.ipynb
LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "Alanine aminotransferase": (50861,),
    "Albumin": (50862,),
    "Albumin (ascites)": (50835,),
    "Albumin (pleural)": (51046,),
    "Albumin (urine)": (51069,),
    "Alkaline phosphate": (50863,),
    "Anion gap": (50868,),
    "Asparate aminotransferase": (50878,),
    "Basophils": (51146,),
    "Bicarbonate": (50882,),
    "Bicarbonate (whole blood)": (50803,),
    "Bilirubin (conjugated)": (50883,),
    "Bilirubin (total)": (50885,),
    "Bilirubin (unconjugated)": (50884,),
    "Blood urea nitrogen": (51006,),
    "Calcium (total)": (50893,),
    "Calcium (urine)": (51077,),
    "Calcium ionized": (50808,),
    "Chloride": (50902,),
    "Chloride (urine)": (51078,),
    "Chloride (whole blood)": (50806,),
    "Cholesterol (HDL)": (50904,),
    "Cholesterol (LDL)": (50905,),
    "Cholesterol (total)": (50907,),
    "CO2 (ETCO2, PCO2, etc.)": (50804,),
    "Creatinine": (50912,),
    "Creatinine (ascites)": (50841,),
    "Creatinine (body fluid)": (51032,),
    "Creatinine (pleural)": (51052,),
    "Creatinine (urine)": (51082,),
    "Eosinophils": (51114, 51347, 51419, 51444),
    "Fibrinogen": (51214,),
    "Glucose": (50809, 50931),
    "Hematocrit": (50810, 51221),
    "Hemoglobin": (50811, 51222),
    "Lactate": (50813,),
    "Lactate dehydrogenase": (50954, 51054),
    "Lymphocytes": (51116, 51244, 51375, 51427, 51446),
    "Lymphocytes (atypical)": (51143, 51343),
    "Lymphocytes (percent)": (51245,),
    "Magnesium": (50960,),
    "Mean corpuscular hemoglobin": (51248,),
    "Mean corpuscular hemoglobin concentration": (51249,),
    "Mean corpuscular volume": (51250,),
    "Monocytes": (51254, 51355),
    "Neutrophils": (51256,),
    "Oxygen saturation": (50817,),
    "Partial pressure of carbon dioxide": (50818,),
    "Partial thromboplastin time": (51275,),
    "pH": (50820, 50831, 51094, 51491),
    "Phosphate": (50970,),
    "Platelets": (51265,),
    "Positive end-expiratory pressure": (50819,),
    "Potassium": (50971,),
    "Potassium (whole blood)": (50822,),
    "Prothrombin time": (51237, 51274),
    "Red blood cell count": (51279, 51493),
    "Red blood cell count (ascites)": (51127,),
    "Red blood cell count (CSF)": (51362,),
    "Red blood cell count (pleural)": (51457,),
    "Sodium": (50983,),
    "Sodium (whole blood)": (50824,),
    "Tidal Volume Observed": (50826,),
    "Total Protein Urine": (51102,),
    "Troponin-I": (51002,),
    "Troponin-T": (51003,),
    "White blood cell count": (51300, 51301, 51516),
}

# LABEVENTS itemid -> lab name
ITEM_TO_LAB: dict[str, str] = {
    str(itemid): lab for lab, itemids in LAB_ITEMIDS.items() for itemid in itemids
}

LAB_NAMES: list[str] = sorted(LAB_ITEMIDS)

# Group flags, where 1 marks the marginalised group
GROUP_FLAGS: tuple[str, ...] = ("black", "female", "public_insurance")

# global_event_df columns that _extract_cohort reads
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "event_type",
    "timestamp",
    "patients/gender",
    "patients/dob",
    "admissions/hadm_id",
    "admissions/dischtime",
    "admissions/hospital_expire_flag",
    "admissions/ethnicity",
    "admissions/insurance",
    "labevents/itemid",
    "labevents/valuenum",
)

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _check_columns(df: pl.LazyFrame) -> None:
    """Raises ValueError if ``df`` lacks column that ``_extract_cohort`` reads."""
    present = set(df.collect_schema().names())
    missing = [col for col in _REQUIRED_COLUMNS if col not in present]
    if missing:
        raise ValueError(
            f"global_event_df has no column(s) {missing}. This task needs MIMIC-III "
            "lab events: load MIMIC3Dataset(..., tables=['labevents'])."
        )


def _to_datetime(col: str) -> pl.Expr:
    # Milliseconds: patients over 89 are ~300 years old, which overflows ns
    # This is done for patient anonymization
    return pl.col(col).str.to_datetime(_DATETIME_FORMAT, time_unit="ms", strict=False)


def _extract_cohort(
    df: pl.LazyFrame,
    window_hours: int = 24,
    horizon_days: int = 8,
    min_age: int = 18,
) -> pl.DataFrame:
    """Builds the paper's cohort, labels, group flags and raw labs.

    Args:
        df: The dataset's ``global_event_df``, loaded with
            ``tables=["labevents"]``.
        window_hours: Length of the observation window, and the minimum stay.
        horizon_days: The label is in-hospital death within this many days of
            admission.
        min_age: Patients must be older than this at admission, in years.

    Returns:
        One row per patient, sorted by integer ``patient_id``, with columns
        ``patient_id``, ``hadm_id``, ``mortality``, the ``GROUP_FLAGS``,
        ``n_lab_orders`` (distinct lab charttimes) and the ``LAB_NAMES``
        (null where not observed).

    Raises:
        ValueError: If ``df`` lacks a required column.
    """
    _check_columns(df)
    window = pl.duration(hours=window_hours)
    patients = df.filter(pl.col("event_type") == "patients").select(
        "patient_id",
        pl.col("patients/gender").alias("gender"),
        _to_datetime("patients/dob").alias("dob"),
    )
    admissions = df.filter(pl.col("event_type") == "admissions").select(
        "patient_id",
        pl.col("timestamp").cast(pl.Datetime("ms")).alias("admittime"),
        pl.col("admissions/hadm_id").alias("hadm_id"),
        _to_datetime("admissions/dischtime").alias("dischtime"),
        pl.col("admissions/hospital_expire_flag")
        .cast(pl.Int8, strict=False)
        .alias("expire"),
        pl.col("admissions/ethnicity").alias("ethnicity"),
        pl.col("admissions/insurance").alias("insurance"),
    )
    cohort = (
        admissions.join(patients, on="patient_id", how="inner")
        .filter((pl.col("admittime") - pl.col("dob")).dt.total_days() / 365 > min_age)
        # Last adult admission; a short one drops the patient
        .sort("patient_id", "admittime", "hadm_id")
        .group_by("patient_id", maintain_order=True)
        .last()
        .filter(pl.col("dischtime") - pl.col("admittime") >= window)
        .collect()
    )
    labs = (
        df.filter(
            (pl.col("event_type") == "labevents")
            & pl.col("labevents/itemid").is_in(list(ITEM_TO_LAB))
        )
        .select(
            "patient_id",
            pl.col("timestamp").cast(pl.Datetime("ms")).alias("charttime"),
            pl.col("labevents/itemid").replace_strict(ITEM_TO_LAB).alias("lab"),
            pl.col("labevents/valuenum").cast(pl.Float64, strict=False).alias("value"),
        )
        .join(cohort.lazy().select("patient_id", "admittime"), on="patient_id")
        .filter(
            (pl.col("charttime") > pl.col("admittime"))
            & (pl.col("charttime") < pl.col("admittime") + window)
        )
        # Drop every copy of a duplicate, null-valued copies included
        .filter(pl.len().over("patient_id", "charttime", "lab") == 1)
        .filter(pl.col("value").is_not_null())
    )
    latest, orders = pl.collect_all(
        [
            labs.group_by("patient_id", "lab").agg(
                pl.col("value").sort_by("charttime").last()
            ),
            labs.group_by("patient_id").agg(
                pl.col("charttime").n_unique().alias("n_lab_orders")
            ),
        ],
        engine="streaming",
    )
    wide = latest.pivot(on="lab", index="patient_id", values="value")
    wide = wide.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(lab)
            for lab in LAB_NAMES
            if lab not in wide.columns
        ]
    )
    died_in_horizon = (pl.col("expire") == 1) & (
        pl.col("dischtime") - pl.col("admittime") < pl.duration(days=horizon_days)
    )
    return (
        cohort.join(wide, on="patient_id", how="inner")
        .join(orders, on="patient_id", how="left")
        .with_columns(
            mortality=died_in_horizon.cast(pl.Int8),
            black=pl.col("ethnicity")
            .str.contains("BLACK", literal=True)
            .fill_null(False)
            .cast(pl.Int8),
            female=(pl.col("gender") == "F").fill_null(False).cast(pl.Int8),
            public_insurance=(pl.col("insurance") != "Private")
            .fill_null(True)
            .cast(pl.Int8),
        )
        .select(
            "patient_id",
            "hadm_id",
            "mortality",
            *GROUP_FLAGS,
            "n_lab_orders",
            *LAB_NAMES,
        )
        .sort(pl.col("patient_id").cast(pl.Int64))
    )


def _assign_split(
    n: int, ratios: Sequence[float], seed_seq: np.random.SeedSequence
) -> np.ndarray:
    """Randomly assigns ``n`` cohort rows to "train", "val" or "test".

    Args:
        n: Number of cohort rows, in ``_extract_cohort``'s order.
        ratios: Train, validation and test fractions, summing to 1.
        seed_seq: Child 0 of ``SeedSequence(seed).spawn(1 + n_draws)``, so the
            split depends only on the cohort and the seed.

    Returns:
        An array of ``n`` strings. Each split's size is within one row of
        ``n * ratio``, and exact when that is an integer.
    """
    perm = np.random.default_rng(seed_seq).permutation(n)
    n_train, n_train_val = np.rint(np.cumsum(ratios[:2]) * n).astype(int)
    split = np.empty(n, dtype="<U5")
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train_val]] = "val"
    split[perm[n_train_val:]] = "test"
    return split


class _OLSResidualNoise(RegressorMixin, BaseEstimator):
    """OLS with a constant predictive sd; the defaults match the reference MICE.

    Solves the normal equations, which is several times faster than
    ``LinearRegression`` on a cohort-sized matrix. By default the sd is that of
    |residual| and the full model is always fit, as in the reference.

    Args:
        min_rows_per_feature: Fit the intercept only when there are fewer than
            this many rows per predictor. 0 always fits the full model.
        proper_noise: Use the sd of the residuals, not of their absolute values.
    """

    def __init__(self, min_rows_per_feature: int = 0, proper_noise: bool = False):
        self.min_rows_per_feature = min_rows_per_feature
        self.proper_noise = proper_noise

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_OLSResidualNoise":
        if len(y) < self.min_rows_per_feature * X.shape[1]:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = y.mean()
        else:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            Xc = X - x_mean
            self.coef_ = np.linalg.lstsq(Xc.T @ Xc, Xc.T @ (y - y_mean), rcond=None)[0]
            self.intercept_ = y_mean - x_mean @ self.coef_
        resid = y - self.predict(X)
        self.sigma_ = float(np.std(resid if self.proper_noise else np.abs(resid)))
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        mu = X @ self.coef_ + self.intercept_
        if return_std:
            return mu, np.full_like(mu, self.sigma_)
        return mu


def _training_range(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column observed min and max; ∓inf for columns with < 2 distinct values.

    scikit-learn's ``IterativeImputer`` needs min < max for every column.
    """
    observed = ~np.isnan(Z)
    lo = np.min(Z, axis=0, where=observed, initial=np.inf)
    hi = np.max(Z, axis=0, where=observed, initial=-np.inf)
    unbounded = ~(lo < hi)
    lo[unbounded], hi[unbounded] = -np.inf, np.inf
    return lo, hi


def _make_imputer(
    imputation: str,
    random_state: int,
    max_iter: int,
    forest_params: dict[str, Any],
    min_value: float | np.ndarray = -np.inf,
    max_value: float | np.ndarray = np.inf,
) -> SimpleImputer | IterativeImputer:
    """Builds one draw's imputer for an ``imputation`` strategy.

    MICE draws each value from N(OLS prediction, sd of |residual|), starting
    from the median and visiting labs from least to most missing. Corrected
    MICE fits the intercept only for labs with fewer than 10 observed rows per
    predictor and uses the sd of the residuals. The MissForest-style imputer
    predicts with extra trees and stops early on scikit-learn's ``tol``.
    ``keep_empty_features`` keeps the column layout when a lab is never
    observed in the training rows. ``min_value`` and ``max_value`` bound the
    iterative imputers' values per column, as in ``IterativeImputer``.
    """
    if imputation == "median":
        return SimpleImputer(strategy="median", keep_empty_features=True)
    common = {
        "max_iter": max_iter,
        "initial_strategy": "median",
        "imputation_order": "ascending",
        "keep_empty_features": True,
        "min_value": min_value,
        "max_value": max_value,
        "random_state": random_state,
    }
    if imputation.endswith("corrected_mice"):
        estimator = _OLSResidualNoise(min_rows_per_feature=10, proper_noise=True)
        return IterativeImputer(estimator=estimator, sample_posterior=True, **common)
    if imputation.endswith("mice"):
        return IterativeImputer(
            estimator=_OLSResidualNoise(), sample_posterior=True, **common
        )
    if imputation.endswith("missforest"):
        params = {
            "n_estimators": 50,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "n_jobs": -1,
            **forest_params,
            "random_state": random_state,
        }
        return IterativeImputer(estimator=ExtraTreesRegressor(**params), **common)
    raise ValueError(f"Unknown imputation {imputation!r}")


def _impute_draws(
    X: np.ndarray,
    G: np.ndarray,
    split: np.ndarray,
    imputation: str,
    draw_seeds: Sequence[np.random.SeedSequence],
    max_iter: int,
    forest_params: dict[str, Any],
) -> np.ndarray:
    """Imputes the labs once per draw seed, fitting on the training rows only.

    Validation and test rows are transformed in separate calls, so the
    training rows' values never depend on them. The ``*corrected_mice``
    strategies keep imputed values within the training rows' observed range.

    Args:
        X: Raw labs, shape (n, n_labs), NaN where not observed.
        G: Group flags, shape (n, n_flags). The ``group_*`` strategies use them
            as extra predictors; they are never part of the output.
        split: "train", "val" or "test" per row.
        imputation: One of the task's imputation strategies.
        draw_seeds: One seed sequence per draw.
        max_iter: Imputation sweeps for the iterative strategies.
        forest_params: Overrides for the MissForest-style extra trees.

    Returns:
        Imputed labs, float32 of shape (len(draw_seeds), n, n_labs).
    """
    Z = np.hstack([X, G]) if imputation.startswith("group_") else X
    Z = Z.astype(np.float64)
    train = split == "train"
    bounds = (-np.inf, np.inf)
    if "corrected" in imputation:
        bounds = _training_range(Z[train])
    out = np.empty((len(draw_seeds), *X.shape), dtype=np.float32)
    for draw, seed_seq in enumerate(draw_seeds):
        random_state = int(seed_seq.generate_state(1)[0])
        imputer = _make_imputer(
            imputation, random_state, max_iter, forest_params, *bounds
        )
        imputed = np.empty_like(Z)
        imputed[train] = imputer.fit_transform(Z[train])
        for part in ("val", "test"):
            rows = split == part
            if rows.any():
                imputed[rows] = imputer.transform(Z[rows])
        out[draw] = imputed[:, : X.shape[1]]
    return out


def _assemble(
    imputed: np.ndarray,
    X: np.ndarray,
    split: np.ndarray,
    add_missing_indicators: bool,
    standardize: bool,
) -> np.ndarray:
    """Builds each draw's features from its imputed labs.

    Missingness indicators are appended after imputation, so the imputers never
    see them. Each draw is standardised with its own training-row statistics.

    Args:
        imputed: Imputed labs from ``_impute_draws``, shape (n_draws, n, n_labs).
        X: Raw labs, shape (n, n_labs), NaN where not observed.
        split: "train", "val" or "test" per row.
        add_missing_indicators: Append one column per lab, 1 where missing.
        standardize: Scale every column, indicators included, to mean 0 and
            sd 1 over the training rows.

    Returns:
        Features, float32 of shape (n_draws, n, n_labs), or (n_draws, n,
        2 * n_labs) with indicators, in ``_feature_names`` order.
    """
    n_draws, n, n_labs = imputed.shape
    width = 2 * n_labs if add_missing_indicators else n_labs
    out = np.empty((n_draws, n, width), dtype=np.float32)
    out[:, :, :n_labs] = imputed
    if add_missing_indicators:
        out[:, :, n_labs:] = np.isnan(X)
    if standardize:
        train = split == "train"
        for features in out:
            features[:] = StandardScaler().fit(features[train]).transform(features)
    return out


def _feature_names(add_missing_indicators: bool) -> list[str]:
    """Names of ``_assemble``'s columns: the labs, then "<lab> (missing)"."""
    names = list(LAB_NAMES)
    if add_missing_indicators:
        names += [f"{lab} (missing)" for lab in LAB_NAMES]
    return names


def _extreme_imputations(
    imputed: np.ndarray, X: np.ndarray, train: np.ndarray
) -> np.ndarray:
    """Counts each row's imputed labs beyond the training range ± 3 sd, per draw.

    Linear MICE extrapolates for labs with few observed training rows; this
    measures how often, for every strategy.

    Args:
        imputed: Imputed labs from ``_impute_draws``, shape (n_draws, n, n_labs).
        X: Raw labs, shape (n, n_labs), NaN where not observed.
        train: Boolean mask of the training rows.

    Returns:
        Integer counts of shape (n_draws, n).
    """
    seen = (~np.isnan(X[train])).any(axis=0)
    observed = X[train][:, seen]
    sd = np.nanstd(observed, axis=0)
    lo = np.full(X.shape[1], -np.inf)
    hi = np.full(X.shape[1], np.inf)
    lo[seen] = np.nanmin(observed, axis=0) - 3 * sd
    hi[seen] = np.nanmax(observed, axis=0) + 3 * sd
    # a lab with one training value has lo == hi
    lo, hi = lo.astype(imputed.dtype), hi.astype(imputed.dtype)
    return (((imputed < lo) | (imputed > hi)) & np.isnan(X)).sum(axis=2)


class _CohortState:
    """Everything ``pre_filter`` fits; it travels to the workers with the task.

    Its ``repr`` is constant, so the task's cache key is the same before and
    after ``pre_filter``.
    """

    def __init__(self) -> None:
        self.row: dict[str, int] = {}
        self.fields: list[dict[str, Any]] = []
        self.features: np.ndarray | None = None
        self.n_extreme: np.ndarray | None = None

    def __repr__(self) -> str:
        return "_CohortState()"


class ClinicalPresenceMortalityMIMIC3(BaseTask):
    """Predicts 8-day mortality from first-day labs imputed at the cohort level.

    Reproduces the MIMIC-III experiment of Jeanselme et al., "Imputation
    Strategies Under Clinical Presence: Impact on Algorithmic Fairness" (ML4H
    2022), following its reference code
    (https://github.com/Jeanselme/ClinicalPresenceFairness). Imputation needs
    the whole cohort, so ``pre_filter`` extracts the cohort, assigns a seeded
    patient split, and imputes and standardises the labs, fitting on the
    training patients only. Each patient then yields one sample per draw.

    Strategies (``imputation``):
        - ``"median"``: each lab's training median.
        - ``"mice"``: the reference's chained linear regressions, with noise
          N(0, sd of the absolute residuals); 10 draws by default.
        - ``"group_mice"``: MICE with the group flags as extra predictors.
        - ``"missforest"``, ``"group_missforest"``: MissForest-style chained
          extra trees, one draw. They run all ``max_iter`` sweeps on MIMIC-III.
        - ``"corrected_mice"``, ``"group_corrected_mice"``: a sensitivity
          variant, not in the paper. Rare labs are fit by their training mean,
          the noise is N(0, sd of residual), and imputed values stay within the
          training range.

    ``add_missing_indicators=True`` gives the paper's "Missing" variants, such
    as Group MICE Missing. The group flags are never features.

    Differences from the reference code: the index admission is the latest by
    admission time; the label uses ``hospital_expire_flag`` and the discharge
    time; labs are linked to the admission by the time window only; the split
    is 80/10/10, with the imputer, scaler and classifier all fit on the same
    80%; indicators are appended after imputation; Group MICE drops all three
    flags; the sex flag is real; MICE noise is seeded per draw.

    Select samples by their ``split`` field and never re-split them: the
    imputer and the scaler were fit on the training patients.

    Each sample has ``patient_id``, ``hadm_id``, ``labs`` (float32, in
    ``feature_names()`` order), ``mortality``, ``split``, ``draw``, the three
    group flags, ``n_lab_orders``, ``n_observed_labs`` and
    ``n_extreme_imputations`` (imputed labs beyond the training range ± 3 sd).

    Args:
        imputation: One of ``IMPUTATIONS``.
        add_missing_indicators: Append one indicator per lab, 1 where missing.
        n_draws: Imputation draws per patient. Defaults to 10 for the MICE
            strategies and 1 otherwise.
        seed: Seeds the split and the draws. The split depends only on the
            cohort and the seed, so every strategy shares it.
        max_iter: Sweeps of the iterative imputers.
        split_ratios: Train, validation and test fractions.
        standardize: Scale every feature to mean 0 and sd 1 over the training
            patients.
        forest_params: Overrides for the MissForest-style ``ExtraTreesRegressor``.

    Raises:
        ValueError: If an argument is invalid.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import ClinicalPresenceMortalityMIMIC3
        >>> base = MIMIC3Dataset(root="/path/to/mimic-iii/1.4", tables=["labevents"])
        >>> task = ClinicalPresenceMortalityMIMIC3(
        ...     imputation="group_mice", add_missing_indicators=True
        ... )
        >>> samples = base.set_task(task)
        >>> samples[0]["labs"].shape
        torch.Size([134])
    """

    task_name: str = "ClinicalPresenceMortalityMIMIC3"
    input_schema: ClassVar[dict[str, str]] = {"labs": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}
    IMPUTATIONS: ClassVar[tuple[str, ...]] = (
        "median",
        "mice",
        "group_mice",
        "missforest",
        "group_missforest",
        "corrected_mice",
        "group_corrected_mice",
    )
    GROUP_FLAGS: ClassVar[tuple[str, ...]] = GROUP_FLAGS

    def __init__(
        self,
        imputation: str = "median",
        add_missing_indicators: bool = False,
        n_draws: int | None = None,
        seed: int = 0,
        max_iter: int = 10,
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        standardize: bool = True,
        forest_params: dict[str, Any] | None = None,
    ) -> None:
        if imputation not in self.IMPUTATIONS:
            raise ValueError(
                f"imputation must be one of {self.IMPUTATIONS}, got {imputation!r}."
            )
        if n_draws is None:
            n_draws = 10 if imputation.endswith("mice") else 1
        if n_draws < 1 or max_iter < 1:
            raise ValueError(
                f"n_draws and max_iter must be at least 1, got {n_draws} and "
                f"{max_iter}."
            )
        if imputation == "median" and n_draws > 1:
            raise ValueError("Median imputation is deterministic: use n_draws=1.")
        ratios = [float(r) for r in split_ratios]
        if len(ratios) != 3 or min(ratios) <= 0 or not math.isclose(sum(ratios), 1):
            raise ValueError(
                "split_ratios must be three positive numbers summing to 1, got "
                f"{split_ratios}."
            )
        super().__init__()
        self.imputation = imputation
        self.add_missing_indicators = add_missing_indicators
        self.n_draws = n_draws
        self.seed = seed
        self.max_iter = max_iter
        self.split_ratios = ratios
        self.standardize = standardize
        self.forest_params = dict(forest_params or {})
        self._state = _CohortState()

    def feature_names(self) -> list[str]:
        """Names of the ``labs`` columns, in order."""
        return _feature_names(self.add_missing_indicators)

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Fits the cohort's split and features; keeps the cohort's patients.

        Args:
            df: The dataset's ``global_event_df``, loaded with
                ``tables=["labevents"]``.

        Returns:
            The ``patients`` rows of the cohort, so ``__call__`` runs once per
            cohort patient.

        Raises:
            ValueError: If ``df`` lacks a column the cohort needs.
        """
        cohort = _extract_cohort(df)
        X = cohort.select(LAB_NAMES).to_numpy().astype(np.float64)
        G = cohort.select(GROUP_FLAGS).to_numpy()
        seeds = np.random.SeedSequence(self.seed).spawn(1 + self.n_draws)
        split = _assign_split(len(cohort), self.split_ratios, seeds[0])
        train = split == "train"

        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            imputed = _impute_draws(
                X,
                G,
                split,
                self.imputation,
                seeds[1:],
                self.max_iter,
                self.forest_params,
            )
        elapsed = time.perf_counter() - start
        unconverged = 0
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                unconverged += 1
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        state = self._state
        state.features = _assemble(
            imputed, X, split, self.add_missing_indicators, self.standardize
        )
        state.n_extreme = _extreme_imputations(imputed, X, train)
        n_observed = (~np.isnan(X)).sum(axis=1)
        state.fields = cohort.select(
            "patient_id",
            pl.col("hadm_id").cast(pl.String),
            pl.col("mortality").cast(pl.Int64),
            *[pl.col(flag).cast(pl.Int64) for flag in GROUP_FLAGS],
            pl.col("n_lab_orders").fill_null(0).cast(pl.Int64),
            pl.Series("n_observed_labs", n_observed, dtype=pl.Int64),
            pl.Series("split", split, dtype=pl.String),
        ).to_dicts()
        ids = cohort["patient_id"].to_list()
        state.row = {pid: row for row, pid in enumerate(ids)}

        self._log_summary(cohort, G, X, split, elapsed, unconverged, state.n_extreme)
        return df.filter(
            (pl.col("event_type") == "patients") & pl.col("patient_id").is_in(ids)
        )

    def _log_summary(
        self,
        cohort: pl.DataFrame,
        G: np.ndarray,
        X: np.ndarray,
        split: np.ndarray,
        elapsed: float,
        unconverged: int,
        n_extreme: np.ndarray,
    ) -> None:
        """Logs aggregate statistics of the fitted cohort; never per-patient data."""
        groups = ", ".join(f"{f} {int(n)}" for f, n in zip(GROUP_FLAGS, G.sum(0)))
        logger.info(
            f"Cohort: {len(cohort)} patients, "
            f"{cohort['mortality'].mean():.1%} died within 8 days; {groups}; "
            f"{np.isnan(X).sum(1).mean():.1f} of {X.shape[1]} labs missing on "
            f"average."
        )
        sizes = " / ".join(
            f"{(split == p).sum()} {p}" for p in ("train", "val", "test")
        )
        logger.info(f"Split (seed {self.seed}): {sizes}.")
        logger.info(
            f"Imputation {self.imputation!r}: {self.n_draws} draw(s) in "
            f"{elapsed:.0f} s."
        )
        if unconverged:
            logger.info(
                f"The imputer ran all {self.max_iter} sweeps without meeting "
                f"scikit-learn's tol in {unconverged} of {self.n_draws} draw(s)."
            )

        def share(mask: np.ndarray) -> str:
            return f"{mask.mean():.2%}" if mask.size else "n/a"

        affected = n_extreme > 0
        by_group = ", ".join(
            f"{flag} {share(affected[:, G[:, g] == 1])} vs. "
            f"{share(affected[:, G[:, g] == 0])}"
            for g, flag in enumerate(GROUP_FLAGS)
        )
        logger.info(
            "Imputed labs beyond the training range ± 3 sd: "
            f"{n_extreme.sum() / max(np.isnan(X).sum() * self.n_draws, 1):.3%} "
            f"of imputed values; patients with any, per draw: {affected.mean():.2%} "
            f"({by_group})."
        )

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Returns the patient's samples, one per draw; none outside the cohort.

        Args:
            patient: A patient of the dataset.

        Returns:
            ``n_draws`` samples that differ only in ``labs``, ``draw`` and
            ``n_extreme_imputations``.

        Raises:
            RuntimeError: If ``pre_filter`` has not run, i.e. the task was called
                outside ``dataset.set_task``.
        """
        state = self._state
        if state.features is None:
            raise RuntimeError(
                "ClinicalPresenceMortalityMIMIC3 fits the cohort in pre_filter; "
                "use it through dataset.set_task(task)."
            )
        row = state.row.get(patient.patient_id)
        if row is None:
            return []
        return [
            {
                **state.fields[row],
                "labs": state.features[draw, row].copy(),
                "draw": draw,
                "n_extreme_imputations": int(state.n_extreme[draw, row]),
            }
            for draw in range(self.n_draws)
        ]
