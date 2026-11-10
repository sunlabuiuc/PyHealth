"""Author: Tamjit Kumar, Yi Wang
NetID: tkumar70, ywang4715
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: eICU Dataset Adapter for Temporal Pointwise Convolution (TPC) Length of
             Stay Prediction and Multi-Modal Feature Extraction.
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pyhealth.datasets import eICUDataset

import numpy as np
import pandas as pd
import polars as pl

from pyhealth.tasks.length_of_stay_tpc import (
    LOS_FLOOR_DAYS,
    BaseLengthOfStayTPC,
    BucketBatchSampler,
    FastReservoirSampler,
    MapDataset,
    RawStayRecord,
    StayDataMap,
    compute_cache_fingerprint,
    compute_los_metrics,
    generate_tpc_tensors,
    length_of_stay_metrics_fn,
)

_compute_cache_fingerprint = compute_cache_fingerprint

logger = logging.getLogger(__name__)

# 6 Core Bedside Vital Signs extracted from physicalExam / vitalPeriodic
CORE_VITALS_MAP = {
    "heart_rate": "Heart Rate",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "respiration_rate": "Respiratory Rate",
    "spo2": "SpO2 (Oxygen Saturation)",
    "temperature": "Temperature",
}

# Substring matches for bedside vital sign paths in eICU physicalExam table
PHYSICAL_EXAM_VITALS_MAP = {
    "HR/HR Current": "heart_rate",
    "BP (systolic)/BP (systolic) Current": "systolic_bp",
    "BP (diastolic)/BP (diastolic) Current": "diastolic_bp",
    "Resp Rate/Resp Current": "respiration_rate",
    "O2 Sat%/O2 Sat% Current": "spo2",
    "Temperature": "temperature",
}

# Substring matches for baseline GCS components in eICU physicalExam table
BASELINE_GCS_MAP = {
    "Eyes Score": "eyes",
    "Motor Score": "motor",
    "Verbal Score": "verbal",
}

# Default clinical priors for 6 bedside vital signs (mean, std, p1, p99)
DEFAULT_VITALS_NORMALIZATION: dict[str, tuple[float, float, float, float]] = {
    "heart_rate": (75.0, 15.0, 40.0, 150.0),
    "systolic_bp": (120.0, 20.0, 70.0, 200.0),
    "diastolic_bp": (70.0, 15.0, 40.0, 110.0),
    "respiration_rate": (18.0, 5.0, 8.0, 40.0),
    "spo2": (98.0, 3.0, 85.0, 100.0),
    "temperature": (37.0, 1.0, 35.0, 39.5),
}

# Default clinical lab tests used if dataset is too small for statistical thresholding
DEFAULT_LAB_NAMES = [
    "potassium",
    "sodium",
    "chloride",
    "creatinine",
    "BUN",
    "glucose",
    "bicarbonate",
    "Hct",
    "Hgb",
    "platelets x 1000",
    "WBC x 1000",
    "calcium",
]


def select_qualifying_labs(
    dataset: eICUDataset | Any,
    train_pids: set[str],
    min_prevalence: float = 0.25,
    min_avg_obs: float = 3.0,
    max_labs: int = 45,
    min_stay_hours: float = 5.0,
    min_age: float = 18.0,
    max_stay_hours: int = 336,
) -> tuple[list[str], bool]:
    """Dynamically selects laboratory tests conforming to the paper's inclusion rule.

    Calculates stay prevalence (fraction of stays with >= 1 measurement) and average
    observations per stay across the training cohort. Labs meeting thresholds
    (prevalence >= min_prevalence and avg_obs > min_avg_obs) are sorted descending
    by prevalence and frequency up to max_labs.

    Args:
        dataset: eICUDataset instance or mock dataset.
        train_pids: Set of patient IDs belonging to the training split.
        min_prevalence: Minimum stay prevalence threshold (default: 0.25).
        min_avg_obs: Minimum average observations per stay threshold (default: 3.0).
        max_labs: Maximum number of lab tests to retain (default: 45).
        min_stay_hours: Minimum ICU length of stay in hours (default: 5.0).
        min_age: Minimum patient age in years (default: 18.0).
        max_stay_hours: Truncation threshold in hours (default: 336).

    Returns:
        tuple[list[str], bool]: Selected lab names and whether dynamic thresholding
            succeeded (returns DEFAULT_LAB_NAMES and False if under 5 labs qualify).

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import (
        ...     select_qualifying_labs,
        ... )
        >>> labs, ok = select_qualifying_labs(None, {"stay_1"})  # doctest: +SKIP
    """
    min_stay_offset = float(min_stay_hours * 60.0)
    max_stay_offset = float(max_stay_hours * 60.0)
    global_df = getattr(dataset, "global_event_df", None)
    if global_df is not None:
        try:
            train_pids_list = list(train_pids)
            lf_p = (
                global_df.filter(
                    (pl.col("event_type") == "patient")
                    & (pl.col("patient_id").is_in(train_pids_list))
                )
                .select(
                    [
                        "patient_id",
                        pl.col("patient/patientunitstayid").alias("sid"),
                        pl.col("patient/unitdischargeoffset")
                        .cast(pl.Float64, strict=False)
                        .alias("offset"),
                        pl.col("patient/age").str.strip_chars().alias("age_raw"),
                    ]
                )
                .with_columns(
                    pl.when(
                        (pl.col("age_raw").is_not_null())
                        & (pl.col("age_raw") != "")
                        & (pl.col("age_raw") != "None")
                    )
                    .then(
                        pl.col("age_raw")
                        .str.replace(">", "")
                        .str.strip_chars()
                        .cast(pl.Float64, strict=False)
                    )
                    .otherwise(None)
                    .alias("age")
                )
                .filter(
                    (pl.col("offset").is_not_null())
                    & (pl.col("offset") >= min_stay_offset)
                    & (pl.col("age").is_not_null())
                    & (pl.col("age") > min_age)
                )
                .with_columns(
                    pl.min_horizontal(pl.col("offset"), pl.lit(max_stay_offset)).alias(
                        "stay_end"
                    )
                )
            )
            df_p = lf_p.select(["sid", "stay_end"]).collect()
            total_train_stays = len(df_p)
            if total_train_stays == 0:
                return DEFAULT_LAB_NAMES, False

            lf_l = (
                global_df.filter(pl.col("event_type") == "lab")
                .select(
                    [
                        pl.col("lab/patientunitstayid").alias("sid"),
                        pl.col("lab/labresultoffset")
                        .cast(pl.Float64, strict=False)
                        .alias("loff"),
                        pl.col("lab/labname").str.strip_chars().alias("lab_name"),
                    ]
                )
                .filter(
                    pl.col("loff").is_not_null()
                    & (pl.col("loff") >= 0.0)
                    & pl.col("lab_name").is_not_null()
                    & (pl.col("lab_name") != "None")
                )
            )

            lab_agg = (
                lf_l.join(df_p.lazy(), on="sid", how="inner")
                .filter(pl.col("loff") < pl.col("stay_end"))
                .group_by("lab_name")
                .agg(
                    [
                        pl.len().alias("obs_count"),
                        pl.col("sid").n_unique().alias("stay_count"),
                    ]
                )
                .collect()
            )
            del df_p

            qualifying = []
            for row in lab_agg.iter_rows(named=True):
                prevalence = row["stay_count"] / float(total_train_stays)
                avg_obs = row["obs_count"] / float(total_train_stays)
                if prevalence >= min_prevalence and avg_obs > min_avg_obs:
                    qualifying.append((prevalence, avg_obs, row["lab_name"]))
            qualifying.sort(key=lambda item: (item[0], item[1]), reverse=True)
            selected = [lab_name for _, _, lab_name in qualifying[:max_labs]]
            del lab_agg
            gc.collect()

            if len(selected) < 5:
                return DEFAULT_LAB_NAMES, False
            return selected, True
        except (
            pl.exceptions.PolarsError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as err:
            logger.debug("Falling back to iter_patients for lab selection: %s", err)

    total_train_stays = 0
    obs_per_lab: dict[str, int] = {}
    stays_per_lab: dict[str, int] = {}

    for patient in dataset.iter_patients():
        if str(patient.patient_id) not in train_pids:
            continue

        patient_stays = patient.get_events("patient")
        patient_labs = patient.get_events("lab")

        labs_by_sid: dict[str, list[Any]] = {}
        for lab in patient_labs:
            sid_lab = str(getattr(lab, "patientunitstayid", ""))
            labs_by_sid.setdefault(sid_lab, []).append(lab)

        for stay in patient_stays:
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                discharge_offset = float(raw_offset)
            except (ValueError, TypeError):
                continue

            if discharge_offset < min_stay_offset:
                continue

            age_raw = getattr(stay, "age", None)
            if age_raw is None or str(age_raw).strip() == "":
                continue
            age_str = str(age_raw).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                continue
            if age <= min_age:
                continue

            sid = str(getattr(stay, "patientunitstayid", ""))
            stay_end = min(discharge_offset, max_stay_offset)
            total_train_stays += 1
            stay_labs: set[str] = set()

            for lab in labs_by_sid.get(sid, []):
                loff_raw = getattr(lab, "labresultoffset", None)
                if loff_raw is None:
                    continue
                try:
                    loff = float(loff_raw)
                except (ValueError, TypeError):
                    continue
                if 0.0 <= loff < stay_end:
                    lab_name_raw = getattr(lab, "labname", None)
                    if lab_name_raw is not None:
                        lab_name = str(lab_name_raw).strip()
                        obs_per_lab[lab_name] = obs_per_lab.get(lab_name, 0) + 1
                        stay_labs.add(lab_name)

            for lab_name in stay_labs:
                stays_per_lab[lab_name] = stays_per_lab.get(lab_name, 0) + 1

    if total_train_stays == 0:
        return DEFAULT_LAB_NAMES, False

    qualifying: list[tuple[float, float, str]] = []
    for lab_name, obs_count in obs_per_lab.items():
        prevalence = stays_per_lab.get(lab_name, 0) / float(total_train_stays)
        avg_obs = obs_count / float(total_train_stays)
        if prevalence >= min_prevalence and avg_obs > min_avg_obs:
            qualifying.append((prevalence, avg_obs, lab_name))

    qualifying.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [lab_name for _, _, lab_name in qualifying[:max_labs]]

    if len(selected) < 5:
        return DEFAULT_LAB_NAMES, False
    return selected, True


def get_valid_eicu_stay_ids(
    dataset: Any,
    min_stay_hours: float = 5.0,
    min_age: float = 18.0,
) -> set[str]:
    """Extracts stay IDs matching adult & minimum LoS criteria from eICUDataset.

    Args:
        dataset: eICUDataset instance or mock dataset.
        min_stay_hours: Minimum ICU length of stay in hours (default: 5.0).
        min_age: Minimum patient age in years (default: 18.0).

    Returns:
        set[str]: Set of valid patientunitstayid strings.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import get_valid_eicu_stay_ids
        >>> valid_ids = get_valid_eicu_stay_ids(None)  # doctest: +SKIP
    """
    min_offset = min_stay_hours * 60.0
    global_df = getattr(dataset, "global_event_df", None)
    if global_df is not None:
        try:
            df_p = (
                global_df.filter(pl.col("event_type") == "patient")
                .select(
                    [
                        pl.col("patient/patientunitstayid").alias("sid"),
                        pl.col("patient/unitdischargeoffset")
                        .cast(pl.Float64, strict=False)
                        .alias("offset"),
                        pl.col("patient/age").str.strip_chars().alias("age_raw"),
                    ]
                )
                .with_columns(
                    pl.when(
                        (pl.col("age_raw").is_not_null())
                        & (pl.col("age_raw") != "")
                        & (pl.col("age_raw") != "None")
                    )
                    .then(
                        pl.col("age_raw")
                        .str.replace(">", "")
                        .str.strip_chars()
                        .cast(pl.Float64, strict=False)
                    )
                    .otherwise(None)
                    .alias("age")
                )
                .filter(
                    (pl.col("offset").is_not_null())
                    & (pl.col("offset") >= min_offset)
                    & (pl.col("age").is_not_null())
                    & (pl.col("age") > min_age)
                )
                .collect()
            )
            return set(df_p["sid"].drop_nulls().to_list())
        except (
            pl.exceptions.PolarsError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as err:
            logger.debug("Falling back to iter_patients for stay filtering: %s", err)

    valid_ids: set[str] = set()
    for patient in dataset.iter_patients():
        for stay in patient.get_events(event_type="patient"):
            sid = str(getattr(stay, "patientunitstayid", ""))
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                offset = float(raw_offset)
            except (ValueError, TypeError):
                continue
            if offset < min_offset:
                continue
            age_raw = getattr(stay, "age", None)
            if age_raw is None or str(age_raw).strip() == "":
                continue
            age_str = str(age_raw).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                continue
            if age <= min_age:
                continue
            valid_ids.add(sid)
    return valid_ids


def load_eicu_vitals_cache(
    eicu_root: str,
    dataset: eICUDataset | None = None,
    chunksize: int = 250_000,
    valid_stay_ids: set[str] | None = None,
    output_path: str | Path | None = None,
    max_stay_hours: int = 336,
) -> tuple[dict[str, list[tuple[int, str, float]]], dict[str, dict[str, float | None]]]:
    """Loads bedside vital signs and admission-window baseline measurements for eICU.

    Parses bedside vital signs from physicalExam and periodic vitals tables, and
    baseline physical exam measurements (height, weight, GCS eyes/motor/verbal)
    within the admission window [-24h, +5h].

    Args:
        eicu_root: Path to directory containing eICU CSV files.
        dataset: Optional eICUDataset instance for stay filtering.
        chunksize: Number of CSV rows per chunk during reading (default: 250,000).
        valid_stay_ids: Optional set of stay IDs to filter during ingestion.
        output_path: Optional file path to save/load pickled cache tuple.
        max_stay_hours: Maximum sequence cutoff in hours (default: 336).

    Returns:
        tuple: (vitals_by_stay, baseline_static_by_stay) where:
            vitals_by_stay: Dict mapping stay ID to list of (hour, vital_name, val).
            baseline_static_by_stay: Dict mapping stay ID to baseline measurements.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import load_eicu_vitals_cache
        >>> vitals, baseline = load_eicu_vitals_cache("./data")  # doctest: +SKIP
    """
    if output_path is not None and os.path.exists(output_path):
        try:
            with open(output_path, "rb") as f:
                cached = pickle.load(f)
                if isinstance(cached, tuple) and len(cached) == 2:
                    return cached
        except (EOFError, pickle.UnpicklingError, OSError) as err:
            logger.warning("Cache corrupt (%s). Recomputing...", err)
            try:
                os.remove(output_path)
            except OSError:
                pass

    stay_filter = set(valid_stay_ids) if valid_stay_ids is not None else None
    max_offset = float(max_stay_hours * 60)
    vitals_sparse: dict[str, list[tuple[float, int, str, float]]] = {}
    vitals_periodic: dict[str, dict[tuple[int, str], tuple[float, float]]] = {}
    baseline_best: dict[str, dict[str, tuple[float, float]]] = {}

    pe_csv = os.path.join(eicu_root, "physicalExam.csv")
    pe_gz = os.path.join(eicu_root, "physicalExam.csv.gz")
    pe_path = (
        pe_csv if os.path.exists(pe_csv) else (pe_gz if os.path.exists(pe_gz) else None)
    )

    if pe_path is not None:
        pe_cols_set = {
            "patientunitstayid",
            "physicalexamoffset",
            "physicalexampath",
            "physicalexamtext",
            "physicalexamvalue",
        }
        for chunk in pd.read_csv(
            pe_path,
            usecols=lambda c, s=pe_cols_set: str(c).lower() in s,
            chunksize=chunksize,
        ):
            chunk.columns = chunk.columns.str.lower()
            has_text = (
                chunk["physicalexamtext"].notnull()
                if "physicalexamtext" in chunk.columns
                else pd.Series(False, index=chunk.index)
            )
            has_val = (
                chunk["physicalexamvalue"].notnull()
                if "physicalexamvalue" in chunk.columns
                else pd.Series(False, index=chunk.index)
            )
            chunk = chunk[has_text | has_val]
            chunk["offset"] = pd.to_numeric(
                chunk["physicalexamoffset"], errors="coerce"
            )
            chunk = chunk[chunk["offset"].notnull()]
            if stay_filter is not None:
                chunk = chunk[chunk["patientunitstayid"].astype(str).isin(stay_filter)]

            # Bedside vitals
            df_v = chunk[(chunk["offset"] >= 0) & (chunk["offset"] < max_offset)]
            for row in df_v.itertuples(index=False):
                sid = str(row.patientunitstayid)
                path = str(row.physicalexampath)
                raw_cand = getattr(row, "physicalexamtext", None)
                val_cand = getattr(row, "physicalexamvalue", None)
                num_val = None
                for cand in (raw_cand, val_cand):
                    if cand is not None:
                        try:
                            num_val = float(cand)
                            break
                        except (ValueError, TypeError):
                            continue
                if num_val is None:
                    continue

                off = float(row.offset)
                hour = int(off // 60)
                for sub, vkey in PHYSICAL_EXAM_VITALS_MAP.items():
                    if sub in path:
                        val = num_val
                        if vkey == "temperature" and val > 50.0:
                            val = (val - 32.0) * 5.0 / 9.0
                        vitals_sparse.setdefault(sid, []).append((off, hour, vkey, val))
                        break

            # Baseline measurements in window [-24h, +5h]
            df_w = chunk[(chunk["offset"] >= -1440.0) & (chunk["offset"] <= 300.0)]
            for row in df_w.itertuples(index=False):
                sid = str(row.patientunitstayid)
                path = str(row.physicalexampath)
                raw_cand = getattr(row, "physicalexamtext", None)
                val_cand = getattr(row, "physicalexamvalue", None)
                num_val = None
                for cand in (raw_cand, val_cand):
                    if cand is not None:
                        try:
                            num_val = float(cand)
                            break
                        except (ValueError, TypeError):
                            continue
                if num_val is None:
                    continue

                abs_off = abs(float(row.offset))
                for sub, gkey in BASELINE_GCS_MAP.items():
                    if sub in path:
                        val = num_val
                        sid_base = baseline_best.setdefault(sid, {})
                        if gkey not in sid_base or abs_off < sid_base[gkey][0]:
                            sid_base[gkey] = (abs_off, val)
                        break

    tables_config = [
        (
            "vitalPeriodic",
            {
                "heartrate": "heart_rate",
                "systemicsystolic": "systolic_bp",
                "systemicdiastolic": "diastolic_bp",
                "respiration": "respiration_rate",
                "sao2": "spo2",
                "temperature": "temperature",
            },
            True,
        ),
        (
            "vitalAperiodic",
            {
                "noninvasivesystolic": "systolic_bp",
                "noninvasivediastolic": "diastolic_bp",
                "temperature": "temperature",
            },
            False,
        ),
    ]

    for table_base, col_map, dedup_hourly in tables_config:
        tbl_csv = os.path.join(eicu_root, f"{table_base}.csv")
        tbl_gz = os.path.join(eicu_root, f"{table_base}.csv.gz")
        tbl_path = (
            tbl_csv
            if os.path.exists(tbl_csv)
            else (tbl_gz if os.path.exists(tbl_gz) else None)
        )
        if tbl_path is None:
            continue

        try:
            header_cols = [c.lower() for c in pd.read_csv(tbl_path, nrows=0).columns]
            vital_cols = [c for c in col_map if c in header_cols]
            if (
                "patientunitstayid" in header_cols
                and "observationoffset" in header_cols
                and vital_cols
            ):
                available_set = {
                    "patientunitstayid",
                    "observationoffset",
                    *vital_cols,
                }
                for chunk in pd.read_csv(
                    tbl_path,
                    usecols=lambda c, s=available_set: str(c).lower() in s,
                    chunksize=chunksize,
                ):
                    chunk.columns = chunk.columns.str.lower()
                    chunk["observationoffset"] = pd.to_numeric(
                        chunk["observationoffset"], errors="coerce"
                    )
                    chunk = chunk[
                        chunk["observationoffset"].notnull()
                        & (chunk["observationoffset"] >= 0)
                        & (chunk["observationoffset"] < max_offset)
                    ]
                    if chunk.empty:
                        continue
                    chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(str)
                    if stay_filter is not None:
                        chunk = chunk[chunk["patientunitstayid"].isin(stay_filter)]
                        if chunk.empty:
                            continue
                    chunk["hour"] = (chunk["observationoffset"] // 60).astype(np.int32)
                    if dedup_hourly:
                        chunk = chunk.sort_values("observationoffset", kind="mergesort")
                    for c_name in vital_cols:
                        vkey = col_map[c_name]
                        sub = chunk[chunk[c_name].notnull()]
                        if sub.empty:
                            continue
                        vals = cast(
                            pd.Series, pd.to_numeric(sub[c_name], errors="coerce")
                        )
                        valid_mask = vals.notnull()
                        if not valid_mask.any():
                            continue
                        sub = sub.loc[valid_mask]
                        vals = vals[valid_mask].astype(float)
                        if vkey == "temperature":
                            vals = np.where(
                                vals > 50.0, (vals - 32.0) * 5.0 / 9.0, vals
                            )
                        if dedup_hourly:
                            dedup_idx = ~sub.duplicated(
                                subset=["patientunitstayid", "hour"],
                                keep="last",
                            )
                            sids = sub.loc[dedup_idx, "patientunitstayid"].tolist()
                            hours = sub.loc[dedup_idx, "hour"].tolist()
                            offs = sub.loc[dedup_idx, "observationoffset"].tolist()
                            v_list = (
                                vals[dedup_idx.to_numpy()].tolist()
                                if isinstance(vals, np.ndarray)
                                else vals.loc[dedup_idx].tolist()
                            )
                            for sid, hr, off, val in zip(sids, hours, offs, v_list):
                                stay_dict = vitals_periodic.setdefault(sid, {})
                                hk = (int(hr), vkey)
                                prev = stay_dict.get(hk)
                                if prev is None or off >= prev[0]:
                                    stay_dict[hk] = (float(off), float(val))
                        else:
                            sids = sub["patientunitstayid"].tolist()
                            hours = sub["hour"].tolist()
                            offs = sub["observationoffset"].tolist()
                            v_list = vals.tolist()
                            for sid, hr, off, val in zip(sids, hours, offs, v_list):
                                vitals_sparse.setdefault(sid, []).append(
                                    (float(off), int(hr), vkey, float(val))
                                )
        except (OSError, ValueError) as err:
            logger.warning("Error processing %s: %s", table_base, err)

    all_sids = set(vitals_sparse.keys()) | set(vitals_periodic.keys())
    vitals_by_stay: dict[str, list[tuple[int, str, float]]] = {}
    for sid in all_sids:
        records = list(vitals_sparse.get(sid, []))
        if sid in vitals_periodic:
            records.extend(
                (off, hr, vkey, val)
                for (hr, vkey), (off, val) in vitals_periodic[sid].items()
            )
        records.sort(key=lambda r: r[0])
        vitals_by_stay[sid] = [(r[1], r[2], r[3]) for r in records]

    baseline_static_by_stay: dict[str, dict[str, float | None]] = {
        sid: {gkey: best[1] for gkey, best in feats.items()}
        for sid, feats in baseline_best.items()
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_suffix(".tmp")
        with open(tmp_p, "wb") as f:
            pickle.dump((vitals_by_stay, baseline_static_by_stay), f)
        tmp_p.replace(out_p)

    return vitals_by_stay, baseline_static_by_stay


def compute_empirical_training_stats(
    dataset: eICUDataset | Any,
    train_pids: set[str],
    lab_names: list[str],
    vitals_by_stay: (
        Mapping[str, Sequence[tuple[int, str, float]]]
        | dict[str, list[tuple[int, str, float]]]
        | None
    ) = None,
    baseline_static_by_stay: (
        Mapping[str, Mapping[str, float | None]]
        | dict[str, dict[str, float]]
        | dict[str, dict[str, float | None]]
        | None
    ) = None,
    min_stay_hours: float = 5.0,
    min_age: float = 18.0,
    max_stay_hours: int = 336,
    max_stat_samples: int = 100_000,
    seed: int = 42,
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    dict[str, int],
    dict[str, tuple[float, float, float]],
    dict[str, list[str]],
]:
    """Computes empirical training statistics and builds fixed categorical vocabs.

    Strictly isolated to training split and cohort filters:
    - Time-series: empirical mean and std computed on 1st/99th percentile
      clipped values.
    - Continuous static: empirical mean, std, and median (for imputation).
    - Categorical static: deterministic sorted vocabulary for one-hot encoding.

    Args:
        dataset: eICUDataset instance.
        train_pids: Set of patient IDs belonging to training split.
        lab_names: List of laboratory test names to include.
        vitals_by_stay: Optional bedside vitals cache dictionary.
        baseline_static_by_stay: Optional baseline measurements dictionary.
        min_stay_hours: Minimum ICU length of stay in hours (default: 5.0).
        min_age: Minimum patient age in years (default: 18.0).
        max_stay_hours: Sequence truncation cutoff in hours (default: 336).
        max_stat_samples: Reservoir sampling capacity per feature to bound
            memory while guaranteeing unbiased uniform sampling across all ICU stays
            (default: 100,000; set <= 0 for unlimited).
        seed: Random seed for reservoir sampling (default: 42).

    Returns:
        tuple: (ts_stats, ts_counts, static_stats, cat_vocabs) where:
            ts_stats: Dict mapping time-series item ID to (mean, std, p1, p99).
            ts_counts: Dict mapping time-series item ID to observation count.
            static_stats: Dict mapping continuous static features to
                (mean, std, median).
            cat_vocabs: Dict mapping categorical static features to sorted
                category lists.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import (
        ...     compute_empirical_training_stats,
        ... )
        >>> stats, counts, s_stats, vocabs = compute_empirical_training_stats(
        ...     None, {"p1"}, ["potassium"]
        ... )  # doctest: +SKIP
    """
    min_stay_offset = float(min_stay_hours * 60.0)
    max_stay_offset = float(max_stay_hours * 60.0)
    ts_stats: dict[str, tuple[float, float, float, float]] = {}
    ts_counts: dict[str, int] = {}
    ts_samplers: dict[str, FastReservoirSampler] = {
        vid: FastReservoirSampler(max_stat_samples, seed=seed)
        for vid in CORE_VITALS_MAP
    }
    for name in lab_names:
        ts_samplers[name] = FastReservoirSampler(max_stat_samples, seed=seed)

    static_samplers: dict[str, FastReservoirSampler] = {
        "age": FastReservoirSampler(max_stat_samples, seed=seed),
        "height": FastReservoirSampler(max_stat_samples, seed=seed),
        "weight": FastReservoirSampler(max_stat_samples, seed=seed),
        "eyes": FastReservoirSampler(max_stat_samples, seed=seed),
        "motor": FastReservoirSampler(max_stat_samples, seed=seed),
        "verbal": FastReservoirSampler(max_stat_samples, seed=seed),
    }
    cat_sets: dict[str, set[str]] = {
        "unittype": set(),
        "unitadmitsource": set(),
        "unitstaytype": set(),
        "ethnicity": set(),
    }

    train_stay_max_hours: dict[str, int] = {}

    global_df = getattr(dataset, "global_event_df", None)
    if global_df is not None:
        try:
            train_pids_list = list(train_pids)
            lf_p = (
                global_df.filter(
                    (pl.col("event_type") == "patient")
                    & (pl.col("patient_id").is_in(train_pids_list))
                )
                .select(
                    [
                        "patient_id",
                        pl.col("patient/patientunitstayid").alias("sid"),
                        pl.col("patient/unitdischargeoffset")
                        .cast(pl.Float64, strict=False)
                        .alias("offset"),
                        pl.col("patient/age").str.strip_chars().alias("age_raw"),
                        pl.col("patient/unittype").fill_null("None").alias("unittype"),
                        pl.col("patient/unitadmitsource")
                        .fill_null("None")
                        .alias("unitadmitsource"),
                        pl.col("patient/unitstaytype")
                        .fill_null("None")
                        .alias("unitstaytype"),
                        pl.col("patient/ethnicity")
                        .fill_null("None")
                        .alias("ethnicity"),
                        pl.col("patient/admissionheight")
                        .cast(pl.Float64, strict=False)
                        .alias("height"),
                        pl.col("patient/admissionweight")
                        .cast(pl.Float64, strict=False)
                        .alias("weight"),
                    ]
                )
                .with_columns(
                    pl.when(
                        (pl.col("age_raw").is_not_null())
                        & (pl.col("age_raw") != "")
                        & (pl.col("age_raw") != "None")
                    )
                    .then(
                        pl.col("age_raw")
                        .str.replace(">", "")
                        .str.strip_chars()
                        .cast(pl.Float64, strict=False)
                    )
                    .otherwise(None)
                    .alias("age")
                )
                .filter(
                    (pl.col("offset").is_not_null())
                    & (pl.col("offset") >= min_stay_offset)
                    & (pl.col("age").is_not_null())
                    & (pl.col("age") > min_age)
                )
                .with_columns(
                    pl.min_horizontal(pl.col("offset"), pl.lit(max_stay_offset)).alias(
                        "stay_end"
                    )
                )
            )

            df_p = lf_p.collect()
            for row in df_p.select(["sid", "stay_end"]).iter_rows():
                train_stay_max_hours[str(row[0])] = int(float(row[1]) // 60)

            for col_name in ["age", "height", "weight"]:
                vals = df_p.filter(
                    pl.col(col_name).is_not_null() & (pl.col(col_name) > 0)
                )[col_name].to_list()
                for v in vals:
                    static_samplers[col_name].add(v)

            for col_name in [
                "unittype",
                "unitadmitsource",
                "unitstaytype",
                "ethnicity",
            ]:
                cat_sets[col_name].update(df_p[col_name].unique().to_list())

            if baseline_static_by_stay:
                for sid in train_stay_max_hours:
                    if sid in baseline_static_by_stay:
                        base = baseline_static_by_stay[sid]
                        for k in ["height", "weight", "eyes", "motor", "verbal"]:
                            val = base.get(k)
                            if val is not None and (
                                k not in ("height", "weight") or val > 0
                            ):
                                static_samplers[k].add(val)

            lf_l = (
                global_df.filter(pl.col("event_type") == "lab")
                .select(
                    [
                        pl.col("lab/patientunitstayid").alias("sid"),
                        pl.col("lab/labresultoffset")
                        .cast(pl.Float64, strict=False)
                        .alias("loff"),
                        pl.col("lab/labname").str.strip_chars().alias("lab_name"),
                        pl.col("lab/labresult")
                        .cast(pl.Float64, strict=False)
                        .alias("lab_val"),
                    ]
                )
                .filter(
                    pl.col("loff").is_not_null()
                    & (pl.col("loff") >= 0.0)
                    & pl.col("lab_name").is_not_null()
                    & pl.col("lab_val").is_not_null()
                    & pl.col("lab_name").is_in(lab_names)
                )
            )

            df_p_sub = df_p.select(["sid", "stay_end"])
            del df_p

            lf_valid_labs = (
                lf_l.join(df_p_sub.lazy(), on="sid", how="inner")
                .filter(pl.col("loff") < pl.col("stay_end"))
                .select(["lab_name", "lab_val"])
            )

            lf_quantiles = lf_valid_labs.group_by("lab_name").agg(
                [
                    pl.col("lab_val")
                    .quantile(0.01, interpolation="linear")
                    .alias("p1"),
                    pl.col("lab_val")
                    .quantile(0.99, interpolation="linear")
                    .alias("p99"),
                    pl.len().alias("count"),
                ]
            )

            lab_stats_df = (
                lf_valid_labs.join(lf_quantiles, on="lab_name", how="inner")
                .with_columns(
                    pl.col("lab_val")
                    .clip(pl.col("p1"), pl.col("p99"))
                    .alias("clipped_val")
                )
                .group_by("lab_name")
                .agg(
                    [
                        pl.col("p1").first().alias("p1"),
                        pl.col("p99").first().alias("p99"),
                        pl.col("clipped_val").mean().alias("mean"),
                        pl.col("clipped_val").std().alias("std"),
                        pl.col("count").first().alias("count"),
                    ]
                )
                .collect()
            )
            del df_p_sub

            for row in lab_stats_df.iter_rows(named=True):
                lname = str(row["lab_name"])
                m = float(row["mean"]) if row["mean"] is not None else 0.0
                s = float(row["std"]) if row["std"] is not None else 1.0
                p1 = float(row["p1"]) if row["p1"] is not None else float("-inf")
                p99 = float(row["p99"]) if row["p99"] is not None else float("inf")
                ts_stats[lname] = (m, s if s > 1e-4 else 1.0, p1, p99)
                ts_counts[lname] = int(row["count"])

            del lab_stats_df
            gc.collect()

        except (
            pl.exceptions.PolarsError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as err:
            logger.debug("Falling back to iter_patients for stats: %s", err)

    if not train_stay_max_hours:
        for patient in dataset.iter_patients():
            if str(patient.patient_id) not in train_pids:
                continue

            patient_stays = patient.get_events("patient")
            patient_labs = patient.get_events("lab")

            labs_by_sid: dict[str, list[Any]] = {}
            for lab in patient_labs:
                sid_lab = str(getattr(lab, "patientunitstayid", ""))
                labs_by_sid.setdefault(sid_lab, []).append(lab)

            for stay in patient_stays:
                raw_offset = getattr(stay, "unitdischargeoffset", None)
                if raw_offset is None:
                    continue
                try:
                    discharge_offset = float(raw_offset)
                except (ValueError, TypeError):
                    continue

                if discharge_offset < min_stay_offset:
                    continue

                age_raw = getattr(stay, "age", None)
                if age_raw is None or str(age_raw).strip() == "":
                    continue
                age_str = str(age_raw).replace(">", "").strip()
                try:
                    age = float(age_str)
                except (ValueError, TypeError):
                    continue
                if age <= min_age:
                    continue

                sid = str(getattr(stay, "patientunitstayid", ""))
                stay_end = min(discharge_offset, max_stay_offset)
                train_stay_max_hours[sid] = int(stay_end // 60)

                static_samplers["age"].add(age)
                cat_sets["unittype"].add(str(getattr(stay, "unittype", "")))
                cat_sets["unitadmitsource"].add(
                    str(getattr(stay, "unitadmitsource", ""))
                )
                cat_sets["unitstaytype"].add(str(getattr(stay, "unitstaytype", "")))
                cat_sets["ethnicity"].add(str(getattr(stay, "ethnicity", "")))

                raw_h = getattr(stay, "admissionheight", None)
                raw_w = getattr(stay, "admissionweight", None)
                if raw_h is not None:
                    try:
                        h_val = float(raw_h)
                        if h_val > 0:
                            static_samplers["height"].add(h_val)
                    except (ValueError, TypeError):
                        pass
                if raw_w is not None:
                    try:
                        w_val = float(raw_w)
                        if w_val > 0:
                            static_samplers["weight"].add(w_val)
                    except (ValueError, TypeError):
                        pass

                if baseline_static_by_stay and sid in baseline_static_by_stay:
                    base = baseline_static_by_stay[sid]
                    for k in ["height", "weight", "eyes", "motor", "verbal"]:
                        val = base.get(k)
                        if val is not None and (
                            k not in ("height", "weight") or val > 0
                        ):
                            static_samplers[k].add(val)

                for lab in labs_by_sid.get(sid, []):
                    loff_raw = getattr(lab, "labresultoffset", None)
                    if loff_raw is None:
                        continue
                    try:
                        loff = float(loff_raw)
                    except (ValueError, TypeError):
                        continue
                    if not (0.0 <= loff < stay_end):
                        continue

                    name_raw = getattr(lab, "labname", None)
                    val_raw = getattr(lab, "labresult", None)
                    if name_raw is not None and val_raw is not None:
                        lab_name = str(name_raw).strip()
                        if lab_name in ts_samplers:
                            try:
                                ts_samplers[lab_name].add(float(val_raw))
                            except (ValueError, TypeError):
                                pass

    if vitals_by_stay:
        for sid, max_h in train_stay_max_hours.items():
            for hour, vkey, val in vitals_by_stay.get(sid, []):
                if 0 <= hour < max_h and vkey in ts_samplers:
                    ts_samplers[vkey].add(val)
    elif dataset is not None and hasattr(dataset, "iter_patients"):
        for patient in dataset.iter_patients():
            if str(patient.patient_id) not in train_pids:
                continue
            pe_events = cast(list[Any], patient.get_events("physicalexam"))
            if pe_events is None or len(pe_events) == 0:
                continue
            for ev in pe_events:
                sid = str(getattr(ev, "patientunitstayid", ""))
                max_h = train_stay_max_hours.get(sid)
                if max_h is None:
                    continue
                off_raw = getattr(ev, "physicalexamoffset", None)
                if off_raw is None:
                    continue
                try:
                    off = float(off_raw)
                except (ValueError, TypeError):
                    continue
                if not (0.0 <= off < max_h * 60.0):
                    continue
                path = str(getattr(ev, "physicalexampath", ""))
                raw_cand = getattr(ev, "physicalexamtext", None)
                val_cand = getattr(ev, "physicalexamvalue", None)
                num_val = None
                for cand in (raw_cand, val_cand):
                    if cand is not None:
                        try:
                            num_val = float(cand)
                            break
                        except (ValueError, TypeError):
                            continue
                if num_val is None:
                    continue
                for sub, vkey in PHYSICAL_EXAM_VITALS_MAP.items():
                    if sub in path:
                        val = num_val
                        if vkey == "temperature" and val > 50.0:
                            val = (val - 32.0) * 5.0 / 9.0
                        if vkey in ts_samplers:
                            ts_samplers[vkey].add(val)
                        break

            if not baseline_static_by_stay:
                best_gcs_by_stay: dict[str, dict[str, tuple[float, float]]] = {}
                for ev in pe_events:
                    sid = str(getattr(ev, "patientunitstayid", ""))
                    if sid not in train_stay_max_hours:
                        continue
                    off_raw = getattr(ev, "physicalexamoffset", None)
                    if off_raw is None:
                        continue
                    try:
                        off = float(off_raw)
                    except (ValueError, TypeError):
                        continue
                    if -1440.0 <= off <= 300.0:
                        path = str(getattr(ev, "physicalexampath", ""))
                        raw_cand = getattr(ev, "physicalexamtext", None)
                        val_cand = getattr(ev, "physicalexamvalue", None)
                        num_val = None
                        for cand in (raw_cand, val_cand):
                            if cand is not None:
                                try:
                                    num_val = float(cand)
                                    break
                                except (ValueError, TypeError):
                                    continue
                        if num_val is None:
                            continue
                        abs_off = abs(off)
                        for sub, gkey in BASELINE_GCS_MAP.items():
                            if sub in path:
                                val = num_val
                                stay_gcs = best_gcs_by_stay.setdefault(sid, {})
                                if gkey not in stay_gcs or abs_off < stay_gcs[gkey][0]:
                                    stay_gcs[gkey] = (abs_off, val)
                                break
                for stay_gcs in best_gcs_by_stay.values():
                    for gkey, (_, val) in stay_gcs.items():
                        if gkey in static_samplers:
                            static_samplers[gkey].add(val)

    all_keys = list(CORE_VITALS_MAP.keys()) + list(lab_names)

    for item_id in all_keys:
        if item_id in ts_stats:
            continue
        sampler = ts_samplers.get(item_id)
        if sampler is not None:
            vals = sampler.samples
            ts_counts[item_id] = sampler.total_count
        else:
            vals = []
            ts_counts[item_id] = 0

        if len(vals) > 0:
            arr: Any = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                p1 = float(np.percentile(arr, 1.0))
                p99 = float(np.percentile(arr, 99.0))
                clipped: Any = np.clip(arr, p1, p99)
                mean = float(np.mean(clipped))
                std = (
                    float(np.std(clipped, ddof=1))
                    if len(clipped) > 1
                    else float(np.std(clipped))
                )
                ts_stats[item_id] = (
                    mean,
                    std if std > 1e-4 else 1.0,
                    p1,
                    p99,
                )
            else:
                ts_stats[item_id] = DEFAULT_VITALS_NORMALIZATION.get(
                    item_id, (0.0, 1.0, float("-inf"), float("inf"))
                )
        else:
            ts_stats[item_id] = DEFAULT_VITALS_NORMALIZATION.get(
                item_id, (0.0, 1.0, float("-inf"), float("inf"))
            )

    vitals_total_obs = sum(ts_counts.get(k, 0) for k in CORE_VITALS_MAP)
    if vitals_total_obs == 0:
        logger.warning(
            "No bedside vital signs found in the training cohort. "
            "Normalization statistics for vitals are falling back to "
            "hardcoded DEFAULT_VITALS_NORMALIZATION priors."
        )

    static_stats: dict[str, tuple[float, float, float]] = {}
    default_priors = {
        "age": (63.0, 15.0, 63.0),
        "height": (170.0, 10.0, 170.0),
        "weight": (80.0, 20.0, 80.0),
        "eyes": (3.5, 0.8, 4.0),
        "motor": (5.5, 1.0, 6.0),
        "verbal": (4.5, 1.0, 5.0),
    }
    for k, prior in default_priors.items():
        sampler = static_samplers.get(k)
        vals = sampler.samples if sampler is not None else []
        if len(vals) > 0:
            arr_s: Any = np.asarray(vals, dtype=np.float64)
            arr_s = arr_s[np.isfinite(arr_s)]
            if len(arr_s) > 0:
                mean = float(np.mean(arr_s))
                std = (
                    float(np.std(arr_s, ddof=1))
                    if len(arr_s) > 1
                    else float(np.std(arr_s))
                )
                median = float(np.median(arr_s))
                static_stats[k] = (mean, std if std > 1e-4 else prior[1], median)
            else:
                static_stats[k] = prior
        else:
            static_stats[k] = prior

    cat_vocabs = {k: sorted(v) for k, v in cat_sets.items()}
    return ts_stats, ts_counts, static_stats, cat_vocabs


class LengthOfStayTPCeICU(BaseLengthOfStayTPC):
    """eICU Clinical Database Adapter for TPC Length of Stay Prediction.

    Inherits from ``BaseLengthOfStayTPC`` and implements ``extract_patient_events``
    to query core eICU tables (patient, vitalPeriodic, vitalAperiodic, lab,
    physicalExam).

    Reproduces the multi-modal dynamic time series and static feature pipeline from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit'
    (Emma Rocheteau, Pietro Liò, Stephanie Hyland, ACM CHIL 2021).

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import LengthOfStayTPCeICU
        >>> dataset = eICUDataset(
        ...     root="./test-resources/core/eicudemo", tables=["patient", "lab"]
        ... )  # doctest: +SKIP
        >>> task = LengthOfStayTPCeICU()  # doctest: +SKIP
        >>> sample_dataset = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "eICU Remaining Length of Stay Prediction (TPC)"
    default_priors: dict[str, tuple[float, ...]] = dict(DEFAULT_VITALS_NORMALIZATION)  # noqa: RUF012

    @classmethod
    def fit(
        cls,
        dataset: Any,
        train_pids: set[str] | Sequence[str],
        vitals_by_stay: (
            Mapping[str, Sequence[tuple[int, str, float]]]
            | dict[str, list[tuple[int, str, float]]]
            | StayDataMap
            | str
            | Path
            | None
        ) = None,
        baseline_static_by_stay: (
            Mapping[str, Mapping[str, float | None]]
            | dict[str, dict[str, float]]
            | dict[str, dict[str, float | None]]
            | StayDataMap
            | str
            | Path
            | None
        ) = None,
        min_prevalence: float = 0.25,
        min_avg_obs: float = 3.0,
        max_labs: int = 45,
        max_stay_hours: int = 336,
        min_stay_hours: float = 5.0,
        min_age: float = 18.0,
        floor_days: float = LOS_FLOOR_DAYS,
        decay_rate: float = 0.75,
        include_mortality: bool = True,
        include_time_features: bool = True,
        max_stat_samples: int = 100_000,
        seed: int = 42,
    ) -> LengthOfStayTPCeICU:
        """Fits task lab selection and empirical statistics on training cohort.

        Args:
            dataset: eICUDataset instance.
            train_pids: Sequence or set of training patient IDs.
            vitals_by_stay: Optional bedside vitals dictionary or cache file path.
            baseline_static_by_stay: Optional baseline physical exam dictionary.
            min_prevalence: Minimum stay prevalence for lab inclusion (default: 0.25).
            min_avg_obs: Minimum average observations per stay for lab inclusion
                (default: 3.0).
            max_labs: Maximum number of lab tests to retain (default: 45).
            max_stay_hours: Sequence truncation cutoff in hours (default: 336).
            min_stay_hours: Minimum ICU length of stay in hours (default: 5.0).
            min_age: Minimum patient age in years (default: 18.0).
            floor_days: Minimum remaining LoS floor threshold in days (default: 1/48).
            decay_rate: Exponential decay base rate (0.75^j). Default: 0.75.
            include_mortality: Whether to extract in-hospital mortality target.
                Default: True.
            include_time_features: Whether to extract non-convolved temporal context
                features. Default: True.
            max_stat_samples: Reservoir sampling capacity per feature
                (default: 100,000).
            seed: Random seed for sampling reproducibility (default: 42).

        Returns:
            LengthOfStayTPCeICU: Fully fitted task instance ready for
                dataset.set_task().
        """
        pids_set = set(train_pids)
        qualifying_labs, _ = select_qualifying_labs(
            dataset=dataset,
            train_pids=pids_set,
            min_prevalence=min_prevalence,
            min_avg_obs=min_avg_obs,
            max_labs=max_labs,
            min_stay_hours=min_stay_hours,
            min_age=min_age,
            max_stay_hours=max_stay_hours,
        )
        raw_vitals = (
            vitals_by_stay._data
            if isinstance(vitals_by_stay, StayDataMap)
            else vitals_by_stay
        )
        raw_base = (
            baseline_static_by_stay._data
            if isinstance(baseline_static_by_stay, StayDataMap)
            else baseline_static_by_stay
        )
        (
            ts_stats,
            ts_counts,
            static_stats,
            cat_vocabs,
        ) = compute_empirical_training_stats(
            dataset=dataset,
            train_pids=pids_set,
            lab_names=qualifying_labs,
            vitals_by_stay=raw_vitals
            if isinstance(raw_vitals, (dict, Mapping))
            else None,
            baseline_static_by_stay=raw_base
            if isinstance(raw_base, (dict, Mapping))
            else None,
            min_stay_hours=min_stay_hours,
            min_age=min_age,
            max_stay_hours=max_stay_hours,
            max_stat_samples=max_stat_samples,
            seed=seed,
        )
        return cls(
            lab_names=qualifying_labs,
            norm_stats=ts_stats,
            vitals_by_stay=vitals_by_stay,
            baseline_static_by_stay=baseline_static_by_stay,
            static_stats=static_stats,
            cat_vocabs=cat_vocabs,
            max_stay_hours=max_stay_hours,
            min_stay_hours=min_stay_hours,
            min_age=min_age,
            floor_days=floor_days,
            decay_rate=decay_rate,
            include_mortality=include_mortality,
            include_time_features=include_time_features,
            feat_counts=ts_counts,
        )

    def __init__(
        self,
        lab_names: list[str] | None = None,
        norm_stats: Mapping[str, tuple[float, ...]] | None = None,
        vitals_by_stay: (
            Mapping[str, Sequence[tuple[int, str, float]]]
            | dict[str, list[tuple[int, str, float]]]
            | StayDataMap
            | str
            | Path
            | None
        ) = None,
        baseline_static_by_stay: (
            Mapping[str, Mapping[str, float | None]]
            | dict[str, dict[str, float]]
            | dict[str, dict[str, float | None]]
            | StayDataMap
            | str
            | Path
            | None
        ) = None,
        static_stats: Mapping[str, tuple[float, float, float]] | None = None,
        cat_vocabs: Mapping[str, list[str]] | None = None,
        max_stay_hours: int = 336,
        min_stay_hours: float = 5.0,
        min_age: float = 18.0,
        floor_days: float = LOS_FLOOR_DAYS,
        decay_rate: float = 0.75,
        include_mortality: bool = True,
        include_time_features: bool = True,
        feat_counts: Mapping[str, int] | None = None,
    ):
        """Initializes the eICU length of stay TPC task.

        Args:
            lab_names: List of laboratory test names to include in dynamic time-series.
            norm_stats: Dict of empirical normalization statistics per feature.
            vitals_by_stay: Optional pre-loaded vitals dictionary or cache file path.
            baseline_static_by_stay: Optional baseline physical exam dictionary or
                cache path.
            static_stats: Continuous static statistics mapping key to
                (mean, std, median).
            cat_vocabs: Categorical vocabulary mapping feature name to valid
                category list.
            max_stay_hours: Maximum length of stay in hours for sequence truncation
                (default: 336).
            min_stay_hours: Minimum length of stay in hours for cohort inclusion
                (default: 5.0).
            min_age: Minimum patient age in years for adult cohort inclusion
                (default: 18.0).
            floor_days: Minimum remaining LoS threshold in days
                (default: 1/48, ~30 mins).
            decay_rate: Exponential decay base rate (0.75^j). Default: 0.75.
            include_mortality: Whether to include in-hospital mortality target.
                Default: True.
            include_time_features: Whether to include non-convolved temporal context
                features. Default: True.
            feat_counts: Optional mapping of feature names to observation counts.
        """
        self.lab_names = lab_names if lab_names is not None else DEFAULT_LAB_NAMES
        self.vital_keys = list(CORE_VITALS_MAP.keys())
        all_features = self.vital_keys + self.lab_names

        super().__init__(
            time_series_features=all_features,
            norm_stats=norm_stats,
            static_stats=static_stats,
            cat_vocabs=cat_vocabs,
            max_stay_hours=max_stay_hours,
            min_stay_hours=min_stay_hours,
            min_age=min_age,
            floor_days=floor_days,
            decay_rate=decay_rate,
            include_mortality=include_mortality,
            include_time_features=include_time_features,
            feat_counts=feat_counts,
        )

        # Decouple large stay dictionaries using StayDataMap
        if isinstance(vitals_by_stay, (str, Path)):
            self.vitals_cache_path: str | None = str(vitals_by_stay)
            self._vitals_data = StayDataMap()
            self.vitals_hash = compute_cache_fingerprint(vitals_by_stay)
            self.vitals_count = -1
        else:
            v_dict = (
                vitals_by_stay._data
                if isinstance(vitals_by_stay, StayDataMap)
                else (vitals_by_stay or {})
            )
            self.vitals_cache_path = None
            self._vitals_data = StayDataMap(v_dict)
            self.vitals_hash = compute_cache_fingerprint(v_dict)
            self.vitals_count = len(v_dict)

        if isinstance(baseline_static_by_stay, (str, Path)):
            self.baseline_cache_path: str | None = str(baseline_static_by_stay)
            self._baseline_data = StayDataMap()
            self.baseline_hash = compute_cache_fingerprint(baseline_static_by_stay)
            self.baseline_count = -1
        else:
            b_dict = (
                baseline_static_by_stay._data
                if isinstance(baseline_static_by_stay, StayDataMap)
                else (baseline_static_by_stay or {})
            )
            self.baseline_cache_path = None
            self._baseline_data = StayDataMap(b_dict)
            self.baseline_hash = compute_cache_fingerprint(b_dict)
            self.baseline_count = len(b_dict)

    @property
    def vitals_by_stay(self) -> StayDataMap:
        if len(self._vitals_data) > 0:
            return self._vitals_data
        if self.vitals_cache_path and os.path.exists(self.vitals_cache_path):
            with open(self.vitals_cache_path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    self._vitals_data = StayDataMap(cast(dict[str, Any], loaded[0]))
                    if len(self._baseline_data) == 0:
                        self._baseline_data = StayDataMap(
                            cast(dict[str, Any], loaded[1])
                        )
                else:
                    self._vitals_data = StayDataMap(cast(dict[str, Any], loaded))
                return self._vitals_data
        return self._vitals_data

    @property
    def baseline_static_by_stay(self) -> StayDataMap:
        if len(self._baseline_data) > 0:
            return self._baseline_data
        if self.baseline_cache_path and os.path.exists(self.baseline_cache_path):
            with open(self.baseline_cache_path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    if len(self._vitals_data) == 0:
                        self._vitals_data = StayDataMap(cast(dict[str, Any], loaded[0]))
                    self._baseline_data = StayDataMap(cast(dict[str, Any], loaded[1]))
                else:
                    self._baseline_data = StayDataMap(cast(dict[str, Any], loaded))
                return self._baseline_data
        return self._baseline_data

    def extract_patient_events(self, patient: Any) -> list[RawStayRecord]:
        """eICU Implementation: Extracts raw stay records from eICU events."""
        patient_stays = patient.get_events(event_type="patient")
        if not patient_stays:
            return []

        patient_labs = patient.get_events(event_type="lab")
        filtered_labs: list[tuple[float, Any]] = []
        for lab in patient_labs:
            lname = str(getattr(lab, "labname", "")).strip()
            if lname not in self.feature_map:
                continue
            loff_raw = getattr(lab, "labresultoffset", None)
            if loff_raw is None:
                continue
            try:
                loff = float(loff_raw)
            except (ValueError, TypeError):
                continue
            filtered_labs.append((loff, lab))
        filtered_labs.sort(key=lambda item: item[0])

        labs_by_sid: dict[str, list[tuple[float, Any]]] = {}
        for loff, lab in filtered_labs:
            sid_lab = str(getattr(lab, "patientunitstayid", ""))
            labs_by_sid.setdefault(sid_lab, []).append((loff, lab))

        patient_pe = patient.get_events(event_type="physicalexam")
        pe_by_sid: dict[str, list[Any]] = {}
        for ev in patient_pe:
            sid_pe = str(getattr(ev, "patientunitstayid", ""))
            pe_by_sid.setdefault(sid_pe, []).append(ev)

        records = []
        for stay in patient_stays:
            sid = str(getattr(stay, "patientunitstayid", ""))
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                discharge_offset = float(raw_offset)
            except (ValueError, TypeError):
                continue

            if discharge_offset < self.min_stay_offset:
                continue

            age_raw = getattr(stay, "age", None)
            if age_raw is None or str(age_raw).strip() == "":
                continue
            age_str = str(age_raw).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                continue
            if age <= self.min_age:
                continue

            stay_hours = int(discharge_offset // 60)
            T = min(stay_hours, self.max_stay_hours)
            if T <= 0:
                continue

            stay_pe_events = pe_by_sid.get(sid, [])

            # 1. Bedside vitals
            stay_vitals: list[tuple[int, str, float]] = []
            if self.vitals_by_stay and sid in self.vitals_by_stay:
                stay_vitals = list(self.vitals_by_stay[sid])
            elif stay_pe_events:
                pe_vitals_raw: list[tuple[float, int, str, float]] = []
                for ev in stay_pe_events:
                    off_raw = getattr(ev, "physicalexamoffset", None)
                    if off_raw is None:
                        continue
                    try:
                        off = float(off_raw)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= off < (T * 60):
                        path = str(getattr(ev, "physicalexampath", ""))
                        raw_cand = getattr(ev, "physicalexamtext", None)
                        val_cand = getattr(ev, "physicalexamvalue", None)
                        num_val = None
                        for cand in (raw_cand, val_cand):
                            if cand is not None:
                                try:
                                    num_val = float(cand)
                                    break
                                except (ValueError, TypeError):
                                    continue
                        if num_val is None:
                            continue
                        hour = int(off // 60)
                        for sub, vkey in PHYSICAL_EXAM_VITALS_MAP.items():
                            if sub in path:
                                val = num_val
                                if vkey == "temperature" and val > 50.0:
                                    val = (val - 32.0) * 5.0 / 9.0
                                pe_vitals_raw.append((off, hour, vkey, val))
                                break
                pe_vitals_raw.sort(key=lambda item: item[0])
                stay_vitals = [(h, k, v) for _, h, k, v in pe_vitals_raw]

            # 2. Lab observations
            for loff, lab in labs_by_sid.get(sid, []):
                hour = int(loff // 60)
                if 0 <= hour < T:
                    lab_name = str(getattr(lab, "labname", "")).strip()
                    val_raw = getattr(lab, "labresult", None)
                    if val_raw is not None and lab_name in self.feature_map:
                        try:
                            stay_vitals.append((hour, lab_name, float(val_raw)))
                        except (ValueError, TypeError):
                            pass

            # 3. Static baselines
            base_s = dict(self.baseline_static_by_stay.get(sid, {}))
            if not base_s and stay_pe_events:
                best_gcs: dict[str, tuple[float, float]] = {}
                for ev in stay_pe_events:
                    off_raw = getattr(ev, "physicalexamoffset", None)
                    if off_raw is None:
                        continue
                    try:
                        off = float(off_raw)
                    except (ValueError, TypeError):
                        continue
                    if -1440.0 <= off <= 300.0:
                        path = str(getattr(ev, "physicalexampath", ""))
                        raw_cand = getattr(ev, "physicalexamtext", None)
                        val_cand = getattr(ev, "physicalexamvalue", None)
                        num_val = None
                        for cand in (raw_cand, val_cand):
                            if cand is not None:
                                try:
                                    num_val = float(cand)
                                    break
                                except (ValueError, TypeError):
                                    continue
                        if num_val is None:
                            continue
                        abs_off = abs(off)
                        for sub, gkey in BASELINE_GCS_MAP.items():
                            if sub in path:
                                val = num_val
                                if gkey not in best_gcs or abs_off < best_gcs[gkey][0]:
                                    best_gcs[gkey] = (abs_off, val)
                                break
                base_s = {k: v[1] for k, v in best_gcs.items()}

            raw_h = getattr(stay, "admissionheight", None)
            parsed_h = None
            if raw_h is not None and str(raw_h).strip() != "":
                try:
                    h_cand = float(raw_h)
                    if np.isfinite(h_cand) and h_cand > 0:
                        parsed_h = h_cand
                except (ValueError, TypeError):
                    pass
            if (
                parsed_h is None
                and base_s.get("height") is not None
                and base_s["height"] > 0
            ):
                parsed_h = base_s["height"]

            raw_w = getattr(stay, "admissionweight", None)
            parsed_w = None
            if raw_w is not None and str(raw_w).strip() != "":
                try:
                    w_cand = float(raw_w)
                    if np.isfinite(w_cand) and w_cand > 0:
                        parsed_w = w_cand
                except (ValueError, TypeError):
                    pass
            if (
                parsed_w is None
                and base_s.get("weight") is not None
                and base_s["weight"] > 0
            ):
                parsed_w = base_s["weight"]

            adm_time_str = str(getattr(stay, "unitadmittime24", "12:00:00"))
            try:
                adm_hour = float(adm_time_str.split(":")[0])
            except (ValueError, IndexError):
                adm_hour = 12.0

            gender_str = str(getattr(stay, "gender", "Female")).lower()
            gender_val = 1.0 if gender_str in ("male", "m") else 0.0

            mort_str = str(getattr(stay, "hospitaldischargestatus", "")).lower()
            unit_mort_str = str(getattr(stay, "unitdischargestatus", "")).lower()
            is_expired = (mort_str == "expired") or (unit_mort_str == "expired")

            records.append(
                RawStayRecord(
                    stay_id=sid,
                    patient_id=str(patient.patient_id),
                    stay_hours=stay_hours,
                    discharge_offset_minutes=discharge_offset,
                    admission_hour=adm_hour,
                    hourly_vitals=stay_vitals,
                    static_continuous={
                        "age": age,
                        "gender": gender_val,
                        "height": parsed_h,
                        "weight": parsed_w,
                        "eyes": base_s.get("eyes"),
                        "motor": base_s.get("motor"),
                        "verbal": base_s.get("verbal"),
                    },
                    static_categorical={
                        "unittype": str(getattr(stay, "unittype", "")),
                        "unitadmitsource": str(getattr(stay, "unitadmitsource", "")),
                        "unitstaytype": str(getattr(stay, "unitstaytype", "")),
                        "ethnicity": str(getattr(stay, "ethnicity", "")),
                    },
                    is_expired=is_expired,
                )
            )
        return records


def length_of_stay_tpc_eicu_fn(patient: Any) -> list[dict[str, Any]]:
    """Functional task wrapper for eICU length of stay prediction with TPC.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import (
        ...     length_of_stay_tpc_eicu_fn,
        ... )
        >>> samples = length_of_stay_tpc_eicu_fn(patient)  # doctest: +SKIP
    """
    return LengthOfStayTPCeICU()(patient)


LengthOfStayPredictionTPCeICU = LengthOfStayTPCeICU
LengthOfStayTPCEICU = LengthOfStayTPCeICU

__all__ = [
    "BASELINE_GCS_MAP",
    "CORE_VITALS_MAP",
    "DEFAULT_LAB_NAMES",
    "DEFAULT_VITALS_NORMALIZATION",
    "LOS_FLOOR_DAYS",
    "PHYSICAL_EXAM_VITALS_MAP",
    "BaseLengthOfStayTPC",
    "BucketBatchSampler",
    "FastReservoirSampler",
    "LengthOfStayPredictionTPCeICU",
    "LengthOfStayTPCEICU",
    "LengthOfStayTPCeICU",
    "MapDataset",
    "RawStayRecord",
    "StayDataMap",
    "compute_cache_fingerprint",
    "compute_empirical_training_stats",
    "compute_los_metrics",
    "generate_tpc_tensors",
    "get_valid_eicu_stay_ids",
    "length_of_stay_metrics_fn",
    "length_of_stay_tpc_eicu_fn",
    "load_eicu_vitals_cache",
    "select_qualifying_labs",
]
