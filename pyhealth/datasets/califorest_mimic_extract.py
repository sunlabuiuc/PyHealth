# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: MIMIC-Extract HDF5 loader for CaliForest flat features
"""MIMIC-Extract dataset for CaliForest replication.

Reads the ``all_hourly_data.h5`` file produced by the `MIMIC-Extract
<https://github.com/MLforHealth/MIMIC_Extract>`_ pipeline and
produces a flat (n_patients x n_features) matrix suitable for
tree-based models such as CaliForest.

The preprocessing follows Park and Ho (2020):

1. Filter to patients with ``max_hours > 30`` (24 h window + 6 h
   gap).
2. Use only the first 24 hours of time-series data.
3. Impute missing values (forward-fill, ICU-stay mean, then 0).
4. Add binary mask and time-since-measured features.
5. Mean-centre features using training-set statistics.
6. Flatten hourly data into a single row per patient.
7. Extract four binary labels: ``mort_hosp``, ``mort_icu``,
   ``los_3`` (LOS > 3 days), ``los_7`` (LOS > 7 days).

The user must have already generated the HDF5 file via the
MIMIC-Extract pipeline or obtained it from GCP (requires PhysioNet
credentials).

Paper: Y. Park and J. C. Ho. "CaliForest: Calibrated Random Forest
for Health Data." ACM CHIL, 2020.
https://doi.org/10.1145/3368555.3384461
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

GAP_TIME = 6  # hours
WINDOW_SIZE = 24  # hours
ID_COLS = ["subject_id", "hadm_id", "icustay_id"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Aggregation Function is the first column level.

    The official MIMIC-Extract HDF5 stores columns as
    ``(LEVEL2, Aggregation Function)`` while some generated
    versions use ``(Aggregation Function, LEVEL2)``.  This
    helper normalises to ``(Aggregation Function, LEVEL2)``.

    Args:
        df: DataFrame with a two-level MultiIndex on columns.

    Returns:
        DataFrame with normalised column order.
    """
    if df.columns.nlevels != 2:
        return df
    agg_vals = {"mean", "count", "std", "mask"}
    level0_vals = set(df.columns.get_level_values(0).unique())
    if not level0_vals & agg_vals:
        df = df.swaplevel(axis=1).sort_index(axis=1)
    return df


def _simple_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values following Che et al. (2018).

    Produces three feature channels per variable: ``mean``
    (imputed value), ``mask`` (observation indicator), and
    ``time_since_measured``.

    Args:
        df: Multi-level column DataFrame from MIMIC-Extract
            ``vitals_labs`` table.

    Returns:
        Imputed DataFrame with ``mean``, ``mask``, and
        ``time_since_measured`` columns.
    """
    idx = pd.IndexSlice
    df = df.copy()

    if df.columns.nlevels > 2:
        df.columns = df.columns.droplevel(
            [n for n in df.columns.names
             if n not in ("Aggregation Function", "LEVEL2")]
        )

    df = _normalize_columns(df)

    df_out = df.loc[:, idx[["mean", "count"], :]]
    icustay_means = (
        df_out.loc[:, idx["mean", :]].groupby(ID_COLS).mean()
    )
    imputed = (
        df_out.loc[:, idx["mean", :]]
        .groupby(ID_COLS)
        .ffill()
    )
    # Fill remaining NaNs with per-ICU-stay means, then 0
    for col in imputed.columns:
        imputed[col] = imputed[col].fillna(icustay_means[col])
    imputed = imputed.fillna(0).copy()
    df_out.loc[:, idx["mean", :]] = imputed

    mask = (
        df.loc[:, idx["count", :]] > 0
    ).astype(float).copy()
    df_out.loc[:, idx["count", :]] = mask
    df_out = df_out.rename(
        columns={"count": "mask"},
        level="Aggregation Function",
    )

    is_absent = 1 - df_out.loc[:, idx["mask", :]].copy()
    hours_of_absence = is_absent.cumsum()
    time_since = hours_of_absence - hours_of_absence[
        is_absent == 0
    ].ffill()
    time_since = time_since.rename(
        columns={"mask": "time_since_measured"},
        level="Aggregation Function",
    )
    df_out = pd.concat((df_out, time_since), axis=1)
    df_out.loc[:, idx["time_since_measured", :]] = (
        df_out.loc[:, idx["time_since_measured", :]]
        .fillna(WINDOW_SIZE + 1)
        .copy()
    )
    df_out.sort_index(axis=1, inplace=True)
    return df_out


class CaliForestMIMICExtractDataset(BaseDataset):
    """MIMIC-Extract dataset for CaliForest experiments.

    Wraps the MIMIC-Extract ``all_hourly_data.h5`` output and
    provides the flat feature matrix and binary labels used in
    the CaliForest paper.

    Unlike the existing ``MIMICExtractDataset`` which produces
    event-level data for deep learning models, this class
    produces the flat, imputed, aggregated feature matrix that
    tree-based models like CaliForest require.

    Args:
        root: Directory containing ``all_hourly_data.h5``.
        tables: Ignored (kept for BaseDataset compatibility).
            Defaults to ``["vitals_labs"]``.
        dataset_name: Name of the dataset. Defaults to
            ``"califorest_mimic_extract"``.
        **kwargs: Additional arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import (
        ...     CaliForestMIMICExtractDataset,
        ... )
        >>> dataset = CaliForestMIMICExtractDataset(
        ...     root="/path/to/mimic_extract_output",
        ... )
        >>> X_tr, X_te, y_tr, y_te = dataset.load_califorest_data(
        ...     target="mort_hosp",
        ... )
        >>> print(X_tr.shape)  # (n_train, 7488)
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._hdf5_path = os.path.join(
            root, "all_hourly_data.h5"
        )
        if not os.path.exists(self._hdf5_path):
            raise FileNotFoundError(
                f"all_hourly_data.h5 not found in {root}. "
                "Run the MIMIC-Extract pipeline first."
            )
        # BaseDataset requires tables and config_path but we
        # handle data loading ourselves via HDF5.  Pass minimal
        # args so the parent __init__ stores root/dataset_name
        # without attempting CSV loading.
        super().__init__(
            root=root,
            tables=tables or ["vitals_labs"],
            dataset_name=(
                dataset_name or "califorest_mimic_extract"
            ),
            **kwargs,
        )

    # Override to prevent BaseDataset from scanning CSVs
    def load_data(self):  # type: ignore[override]
        """Not used. Data is loaded via :meth:`load_califorest_data`."""
        raise NotImplementedError(
            "Use load_califorest_data() instead."
        )

    def load_califorest_data(
        self,
        target: str = "mort_hosp",
        random_seed: int = 0,
        train_frac: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data for CaliForest.

        Replicates the exact data pipeline from the CaliForest
        paper, producing the flat feature matrix and binary
        labels used in the original experiments.

        Args:
            target: Prediction target. One of ``"mort_hosp"``,
                ``"mort_icu"``, ``"los_3"``, ``"los_7"``.
            random_seed: Random seed for train/test split.
            train_frac: Fraction of subjects for training.

        Returns:
            A tuple ``(X_train, X_test, y_train, y_test)``.

        Examples:
            >>> X_tr, X_te, y_tr, y_te = (
            ...     dataset.load_califorest_data("mort_hosp")
            ... )
        """
        return load_califorest_data(
            self._hdf5_path,
            target=target,
            random_seed=random_seed,
            train_frac=train_frac,
        )


def load_califorest_data(
    hdf5_path: str,
    target: str = "mort_hosp",
    random_seed: int = 0,
    train_frac: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess MIMIC-Extract data for CaliForest.

    This is a standalone convenience function. For integration
    with PyHealth pipelines, prefer
    :class:`CaliForestMIMICExtractDataset`.

    Args:
        hdf5_path: Path to ``all_hourly_data.h5``.
        target: One of ``"mort_hosp"``, ``"mort_icu"``,
            ``"los_3"``, ``"los_7"``.
        random_seed: Random seed for train/test split.
        train_frac: Fraction of subjects for training.

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)``.
    """
    statics = pd.read_hdf(hdf5_path, "patients")
    data_full = pd.read_hdf(hdf5_path, "vitals_labs")

    # Cohort filter
    statics = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME]

    # Labels
    Ys = statics.loc[:, ["mort_hosp", "mort_icu", "los_icu"]]
    Ys["mort_hosp"] = Ys["mort_hosp"].astype(int)
    Ys["mort_icu"] = Ys["mort_icu"].astype(int)
    Ys["los_3"] = (Ys["los_icu"] > 3).astype(int)
    Ys["los_7"] = (Ys["los_icu"] > 7).astype(int)
    Ys.drop(columns=["los_icu"], inplace=True)

    # Filter time-series to cohort and first 24 h
    valid_ids = set(
        Ys.index.get_level_values("icustay_id")
    )
    lvl2 = data_full.loc[
        data_full.index.get_level_values("icustay_id").isin(
            valid_ids
        )
        & (
            data_full.index.get_level_values("hours_in")
            < WINDOW_SIZE
        ),
        :,
    ]

    # Train/test split by subject
    subjects = list(
        set(lvl2.index.get_level_values("subject_id"))
    )
    np.random.seed(random_seed)
    subjects = np.random.permutation(subjects)
    n_train = int(train_frac * len(subjects))
    train_subj = set(subjects[:n_train])
    test_subj = set(subjects[n_train:])

    def _filter(df, subj_set):
        return df.loc[
            df.index.get_level_values("subject_id").isin(
                subj_set
            )
        ]

    lvl2_train = _filter(lvl2, train_subj)
    lvl2_test = _filter(lvl2, test_subj)
    Ys_train = _filter(Ys, train_subj)
    Ys_test = _filter(Ys, test_subj)

    # Normalize column level order
    lvl2_train = _normalize_columns(lvl2_train)
    lvl2_test = _normalize_columns(lvl2_test)

    # Mean-centre using training statistics
    idx = pd.IndexSlice
    lvl2_means = lvl2_train.loc[:, idx["mean", :]].mean(axis=0)
    lvl2_train.loc[:, idx["mean", :]] -= lvl2_means
    lvl2_test.loc[:, idx["mean", :]] -= lvl2_means

    # Impute
    lvl2_train = _simple_imputer(lvl2_train)
    lvl2_test = _simple_imputer(lvl2_test)

    # Flatten
    X_train_df = lvl2_train.pivot_table(
        index=ID_COLS, columns=["hours_in"]
    )
    X_test_df = lvl2_test.pivot_table(
        index=ID_COLS, columns=["hours_in"]
    )

    # Align features and labels to shared patients
    common_tr = Ys_train.index.intersection(X_train_df.index)
    common_te = Ys_test.index.intersection(X_test_df.index)
    X_train = X_train_df.loc[common_tr].values
    X_test = X_test_df.loc[common_te].values
    y_train = Ys_train.loc[common_tr, target].values
    y_test = Ys_test.loc[common_te, target].values

    logger.info(
        f"Loaded MIMIC-Extract: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, "
        f"prevalence={y_train.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test
