"""PyHealth dataset for MIMIC-Extract.

Dataset links:
    https://github.com/MLforHealth/MIMIC_Extract (pipeline and instructions)
    https://physionet.org/content/mimiciii/1.4/ (the underlying MIMIC-III data)

Dataset paper: (please cite if you use this dataset)
    Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Marzyeh Ghassemi,
    Michael C. Hughes, and Tristan Naumann. "MIMIC-Extract: A Data Extraction,
    Preprocessing, and Representation Pipeline for MIMIC-III." ACM Conference
    on Health, Inference, and Learning (CHIL), 2020.

Used by:
    Yubin Park and Joyce C. Ho. "CaliForest: Calibrated Random Forest for
    Health Data." ACM Conference on Health, Inference, and Learning (CHIL),
    2020. See :mod:`pyhealth.tasks.mimic_extract`.

Description:
    MIMIC-Extract turns MIMIC-III into a cohort of adult ICU stays (by default
    the first ICU stay of each patient, at least 12 hours long) with hourly
    vitals and labs. Its main output, ``all_hourly_data.h5``, holds a
    ``patients`` table (one row per stay: demographics, admission and ICU
    times, and outcomes such as ``mort_hosp``, ``mort_icu`` and ``los_icu``)
    and a wide ``vitals_labs`` table (one row per stay and hour, with a
    ``(variable, aggregation)`` column for each of ``mean``, ``count`` and
    ``std``).

    MIMIC-III is credentialed: access requires PhysioNet credentialing and a
    signed data use agreement, so this dataset cannot download it. Ways to
    get ``all_hourly_data.h5``:

    * The pre-built default cohort in the ``mimic_extract`` bucket on Google
      Cloud, available once your PhysioNet access is linked to GCP.
    * The step-by-step instructions in the MIMIC_Extract repository, which
      build MIMIC-III in PostgreSQL first.
    * MIMICExtractEasy (https://github.com/SphtKr/MIMICExtractEasy), which uses
      DuckDB instead of PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import narwhals as nw
import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables

class MIMICExtractDataset(BaseEHRDataset):
    """Base dataset for MIMIC-Extract dataset.

    Reads the HDF5 data produced by
    `MIMIC-Extract <https://github.com/MLforHealth/MIMIC_Extract#step-4-set-cohort-selection-and-extraction-criteria>`__.
    Works with files created with or without LEVEL2 grouping and with restricted cohort population
    sizes, other optional parameter values, and should work with many customized versions of the pipeline.

    MIMIC-Extract is derived from MIMIC-III, so every route below first needs
    credentialed PhysioNet access to MIMIC-III
    (https://physionet.org/content/mimiciii/1.4/; see :class:`MIMIC3Dataset`).
    You can create or obtain a MIMIC-Extract dataset in several ways:

    * The default cohort dataset is `available on GCP <https://console.cloud.google.com/storage/browser/mimic_extract>`__
      (requires PhysioNet access provisioned in GCP).
    * Follow the `step-by-step instructions <https://github.com/MLforHealth/MIMIC_Extract#step-by-step-instructions>`__
      on the MIMIC_Extract github site, which includes setting up a PostgreSQL database and loading
      the MIMIC-III data files.
    * Use the instructions at `MIMICExtractEasy <https://github.com/SphtKr/MIMICExtractEasy>`__ which uses DuckDB
      instead and should be a good bit simpler.
    
    Any of these methods will provide you with a set of HDF5 files containing a cleaned subset of the MIMIC-III dataset.
    This class can be used to read that dataset (mainly the `all_hourly_data.h5` file). Consult the MIMIC-Extract
    documentation for all the options available for dataset generation (cohort selection, aggregation level, etc.).

    Getting the data:
        - Source: MIMIC-III on PhysioNet,
          https://physionet.org/content/mimiciii/1.4/, processed with
          MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract).
        - Access: credentialed, the same as :class:`MIMIC3Dataset`.
        - Download: build the HDF5 files with one of the routes above.
        - ``root``: the folder holding ``all_hourly_data.h5`` (or
          ``all_hourly_data_<pop_size>.h5`` when ``pop_size`` is set).

    Args:
        patients: The ``patients`` table, indexed by (or with columns)
            ``subject_id``, ``hadm_id`` and ``icustay_id``.
        vitals_labs: The wide ``vitals_labs`` table, indexed by
            ``subject_id``, ``hadm_id``, ``icustay_id`` and ``hours_in``, with
            two column levels: variable name and aggregation function.
        root: Directory to write the CSVs into.
        max_hours: If set, only hours ``0 <= hours_in < max_hours`` are kept.
        chunk_rows: Wide hourly rows reshaped at a time, to bound memory use.

    Raises:
        ValueError: If a required column, index level or aggregation is
            missing, the column levels are not (variable, aggregation), an
            hourly row belongs to an unknown stay, or ``max_hours`` is not
            positive.

    Examples:
        >>> build_mimic_extract_tables(
        ...     patients, vitals_labs, "data/mimic_extract", max_hours=48
        ... )  # doctest: +SKIP
    """
    if max_hours is not None and max_hours <= 0:
        raise ValueError(f"max_hours must be positive, got {max_hours}")
    root = Path(root)
    stays = _patients_table(patients)
    vitals_labs = _check_vitals_labs(vitals_labs)
    intime = stays.set_index("icustay_id")["intime"]
    _write_csv(_format_datetimes(stays), root / PATIENTS_CSV)

    if max_hours is not None:
        hours = vitals_labs.index.get_level_values("hours_in")
        vitals_labs = vitals_labs[(hours >= 0) & (hours < max_hours)]
    dest = root / VITALS_LABS_CSV
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        for start in range(0, max(len(vitals_labs), 1), chunk_rows):
            chunk = _long_vitals_labs(
                vitals_labs.iloc[start : start + chunk_rows], intime
            )
            first = start == 0
            chunk.to_csv(tmp, index=False, mode="w" if first else "a", header=first)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_mimic_extract(
    root: str | Path,
    pop_size: int | None = None,
    max_hours: int | None = None,
) -> None:
    """Builds the dataset's CSV tables from ``all_hourly_data.h5`` in ``root``.

    Reads the ``patients`` and ``vitals_labs`` tables and writes them with
    :func:`build_mimic_extract_tables`. Reading HDF5 needs PyTables
    (``pip install tables``), which PyHealth does not install by default.

    The whole ``vitals_labs`` table is loaded into memory (several GB for the
    default cohort), since the pipeline stores it in pandas' fixed format.

    Args:
        root: Directory with ``all_hourly_data.h5`` (or
            ``all_hourly_data_{pop_size}.h5``).
        pop_size: The ``pop_size`` the extract was built with, if any; it is
            part of the file name.
        max_hours: If set, only the first ``max_hours`` hours of each stay
            are kept, which makes the hourly table much smaller.

    Raises:
        FileNotFoundError: If ``root`` or the HDF5 file does not exist.
        ImportError: If PyTables is not installed.
        ValueError: If the file lacks a table or a required column.

    Examples:
        >>> prepare_mimic_extract("data/mimic_extract", max_hours=48)  # doctest: +SKIP
    """
    path = _h5_path(root, pop_size)
    try:
        import tables  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Reading MIMIC-Extract's HDF5 file needs PyTables, which PyHealth "
            "does not install. Install it with `pip install tables`."
        ) from err
    with pd.HDFStore(path, mode="r") as store:
        keys = {key.lstrip("/") for key in store}
        missing = [key for key in SUPPORTED_TABLES if key not in keys]
        if missing:
            raise ValueError(f"{path} has no {missing} table; found {sorted(keys)}")
        patients = store["patients"]
        vitals_labs = store["vitals_labs"]
    logger.info("Writing MIMIC-Extract tables to %s", root)
    build_mimic_extract_tables(patients, vitals_labs, root, max_hours=max_hours)
    _write_params(Path(root), pop_size, max_hours)


def _h5_path(root: str | Path, pop_size: int | None) -> Path:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MIMIC-Extract root {root} is not a directory")
    suffix = "" if pop_size is None else f"_{pop_size}"
    path = root / f"all_hourly_data{suffix}.h5"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Put MIMIC-Extract's all_hourly_data.h5 in "
            f"{root} (see the MIMICExtractDataset docs for how to get it), "
            "and pass pop_size if the extract was built with one."
        )
    return path


def _patients_table(patients: pd.DataFrame) -> pd.DataFrame:
    df = patients.reset_index() if _has_index(patients, STAY_INDEX) else patients
    df = df.rename(columns=str.lower)
    missing = [c for c in PATIENTS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"patients table is missing columns {missing}")
    if df["icustay_id"].duplicated().any():
        raise ValueError("patients table has more than one row per icustay_id")
    df = df.copy()
    for column in PATIENTS_OPTIONAL:
        if column not in df.columns:
            df[column] = pd.NA
    for column in STAY_INDEX:
        df[column] = _id_strings(df[column])
    for column in _DATETIME_COLUMNS:
        df[column] = pd.to_datetime(df[column])
    if df["intime"].isna().any():
        raise ValueError("patients table has stays without an intime")
    return df[[*PATIENTS_REQUIRED, *PATIENTS_OPTIONAL]]


def _check_vitals_labs(vitals_labs: pd.DataFrame) -> pd.DataFrame:
    if not _has_index(vitals_labs, HOURLY_INDEX):
        raise ValueError(
            f"vitals_labs must be indexed by {list(HOURLY_INDEX)}, got "
            f"{list(vitals_labs.index.names)}"
        )
    if vitals_labs.columns.nlevels != 2:
        raise ValueError(
            "vitals_labs columns must have two levels (variable, aggregation "
            f"function), got {vitals_labs.columns.nlevels}. Extracts built "
            "without LEVEL2 grouping (--no_group_by_level2) are not supported."
        )
    aggregations = vitals_labs.columns.get_level_values(1)
    missing = [agg for agg in AGGREGATES if agg not in set(aggregations)]
    if missing:
        raise ValueError(f"vitals_labs has no {missing} aggregation columns")
    vitals_labs = vitals_labs.reorder_levels(list(HOURLY_INDEX))
    return vitals_labs.loc[:, aggregations.isin(AGGREGATES)]


def _long_vitals_labs(wide: pd.DataFrame, intime: pd.Series) -> pd.DataFrame:
    """Reshapes wide hourly rows to one row per (stay, hour, variable)."""
    columns = [*HOURLY_INDEX, "time", "variable", *AGGREGATES]
    if wide.empty:
        return pd.DataFrame(columns=columns)
    wide = wide.copy()
    wide.columns = pd.MultiIndex.from_tuples(
        [(str(var).lower(), agg) for var, agg in wide.columns],
        names=["variable", "aggregation"],
    )
    long = wide.stack(level="variable", future_stack=True).reset_index()
    count = long["count"].fillna(0)
    long = long[long["mean"].notna() | long["std"].notna() | (count > 0)]
    long = long.assign(
        subject_id=_id_strings(long["subject_id"]),
        hadm_id=_id_strings(long["hadm_id"]),
        icustay_id=_id_strings(long["icustay_id"]),
        hours_in=long["hours_in"].astype("int64"),
    )
    starts = long["icustay_id"].map(intime)
    if starts.isna().any():
        unknown = sorted(long.loc[starts.isna(), "icustay_id"].unique())[:5]
        raise ValueError(f"vitals_labs has stays not in patients, e.g. {unknown}")
    stamps = starts + pd.to_timedelta(long["hours_in"], unit="h")
    long["time"] = stamps.dt.strftime(_TIME_FORMAT)
    return long[columns]


def _has_index(df: pd.DataFrame, names: tuple[str, ...]) -> bool:
    return set(names) <= {str(n).lower() for n in df.index.names if n is not None}


def _id_strings(values: pd.Series) -> pd.Series:
    return values.astype("int64").astype(str)


def _format_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in _DATETIME_COLUMNS:
        df[column] = df[column].dt.strftime(_TIME_FORMAT)
    return df


def _write_csv(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def _params(pop_size: int | None, max_hours: int | None) -> dict:
    return {"pop_size": pop_size, "max_hours": max_hours}


def _write_params(root: Path, pop_size: int | None, max_hours: int | None) -> None:
    (root / PARAMS_JSON).write_text(json.dumps(_params(pop_size, max_hours)))


def _tables_current(root: Path, pop_size: int | None, max_hours: int | None) -> bool:
    """Whether the CSVs exist and were built with these arguments.

    CSVs without a params file (for example, written by
    :func:`build_mimic_extract_tables` directly) are used as they are.
    """
    if not all((root / name).is_file() for name in _EXPECTED_CSV_COLUMNS):
        return False
    params = root / PARAMS_JSON
    if not params.is_file():
        return True
    return json.loads(params.read_text()) == _params(pop_size, max_hours)


_EXPECTED_CSV_COLUMNS = {
    PATIENTS_CSV: (*PATIENTS_REQUIRED, *PATIENTS_OPTIONAL),
    VITALS_LABS_CSV: (*HOURLY_INDEX, "time", "variable", *AGGREGATES),
}


def _check_csv_columns(root: Path) -> None:
    """Raises ValueError if an existing CSV lacks columns the config reads."""
    for name, expected in _EXPECTED_CSV_COLUMNS.items():
        path = root / name
        if not path.is_file():
            continue
        found = set(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in expected if c not in found]
        if missing:
            raise ValueError(
                f"{path} is missing columns {missing}. Delete it so it is "
                "rebuilt from all_hourly_data.h5."
            )


class MIMICExtractDataset(BaseDataset):
    """ICU stays and hourly vitals/labs from MIMIC-Extract.

    Each patient is a MIMIC-III ``subject_id``. Two tables:

    * ``patients``: one event per ICU stay, timestamped at the stay's
      ``intime``, with ``icustay_id``, ``hadm_id``, the outcomes
      ``mort_hosp``, ``mort_icu`` and ``los_icu`` (ICU length of stay in
      fractional days), and demographics and admission details.
    * ``vitals_labs``: one event per (stay, hour, variable) with at least one
      measurement, timestamped at ``intime`` plus ``hours_in`` hours, with
      ``icustay_id``, ``hours_in``, ``variable`` (MIMIC-Extract's LEVEL2 name,
      lower case) and the hourly ``mean``, ``count`` and ``std``.

    ``hours_in`` is loaded as an integer and the measurements, outcomes and
    ``age`` as floats; identifiers, dates and categories stay strings. The
    tables are built once from ``all_hourly_data.h5`` by
    :func:`prepare_mimic_extract` and written into ``root``; they are rebuilt
    when ``pop_size`` or ``max_hours`` changes.

    Args:
        root: Directory with MIMIC-Extract's ``all_hourly_data.h5``, or with
            the CSVs already built from it.
        tables: Tables to load, from ``"patients"`` and ``"vitals_labs"``.
            Defaults to both.
        pop_size: The ``pop_size`` the extract was built with, if any.
        max_hours: Keep only the first ``max_hours`` hours of each stay. The
            CaliForest tasks use 24 hours plus a 6 hour gap, so 48 keeps the
            hourly table small while leaving room for longer windows.
        dataset_name: Name of the dataset. Defaults to ``"mimic_extract"``.
        config_path: Path to a YAML config. Defaults to the bundled one.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``cache_dir``, ``dev``, ``num_workers``).

    Raises:
        FileNotFoundError: If ``root`` does not exist, or the tables have not
            been built and ``all_hourly_data.h5`` is not in ``root``.
        ValueError: If ``tables`` names an unknown table, or a built CSV is
            missing columns the config reads.

    Examples:
        >>> from pyhealth.datasets import MIMICExtractDataset
        >>> from pyhealth.tasks import MIMICExtractMortalityHospital
        >>> dataset = MIMICExtractDataset(
        ...     root="data/mimic_extract", max_hours=48, cache_dir="cache"
        ... )  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     MIMICExtractMortalityHospital(variables=["heart rate", "glucose"])
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        pop_size: int | None = None,
        max_hours: int | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"MIMIC-Extract root {root} is not a directory")
        tables = list(tables) if tables is not None else list(SUPPORTED_TABLES)
        if config_path is None:
            unknown = [t for t in tables if t not in SUPPORTED_TABLES]
            if unknown:
                raise ValueError(
                    f"Unknown MIMIC-Extract table(s) {unknown}; supported "
                    f"tables are {list(SUPPORTED_TABLES)}"
                )
            config_path = str(Path(__file__).parent / "configs" / "mimic_extract.yaml")
        if not _tables_current(root_path, pop_size, max_hours):
            logger.info("Building the MIMIC-Extract tables in %s", root_path)
            prepare_mimic_extract(root_path, pop_size=pop_size, max_hours=max_hours)
        _check_csv_columns(root_path)
        super().__init__(
            root=str(root_path),
            tables=tables,
            dataset_name=dataset_name or "mimic_extract",
            config_path=config_path,
            **kwargs,
        )

    def preprocess_patients(self, df: nw.LazyFrame) -> nw.LazyFrame:
        """Casts the outcome and age columns to floats.

        Args:
            df: The ``patients`` table, with every column read as a string.

        Returns:
            The table with its numeric columns cast.
        """
        return df.with_columns(nw.col(*_PATIENTS_NUMERIC).cast(nw.Float64))

    def preprocess_vitals_labs(self, df: nw.LazyFrame) -> nw.LazyFrame:
        """Casts ``hours_in`` to an integer and the aggregates to floats.

        Args:
            df: The ``vitals_labs`` table, with every column read as a string.

        Returns:
            The table with its numeric columns cast.
        """
        return df.with_columns(
            nw.col("hours_in").cast(nw.Int64),
            nw.col(*AGGREGATES).cast(nw.Float64),
        )
