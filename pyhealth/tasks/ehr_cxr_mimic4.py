"""ICU time series paired with chest X-rays on MIMIC-IV + MIMIC-CXR.

Builds the inputs of :class:`~pyhealth.models.MedFuse` (``ehr``, ``cxr`` and
``cxr_mask``) for the two tasks of its paper: in-hospital mortality after 48
hours in the ICU, and phenotyping of the whole ICU stay.

Paper:
    Hayat, N., Geras, K. J., & Shamout, F. E. (2022). MedFuse: Multi-modal
    fusion with clinical time-series data and chest X-ray images. MLHC 2022.
    https://arxiv.org/abs/2207.07027

The stay exclusions and the pairing of AP images with stays are adapted from
DiPro (commit 64b12ec, ``mimic_data_extract/mimic3benchmark``,
https://github.com/Chenliu-svg/DiPro), MIT License,
Copyright (c) 2025 LIU Chen @ The Hong Kong Polytechnic University, which in
turn builds on mimic3-benchmarks, MIT License, Copyright (c) 2017 YerevaNN.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from functools import lru_cache
from math import ceil
from typing import Any, ClassVar, NamedTuple

import numpy as np
import polars as pl
import torch

from pyhealth.medcode import CrossMap
from pyhealth.processors import ImageProcessor
from pyhealth.processors.base_processor import FeatureProcessor

from .base_task import BaseTask


class _Item(NamedTuple):
    """Rows of one MIMIC-IV table that feed a feature.

    ``scale``/``offset`` convert a value as ``value * scale + offset``, applied
    only to rows whose unit matches ``unit_regex`` (case-insensitive).
    """

    table: str
    itemids: tuple[int, ...]
    unit_regex: str | None = None
    scale: float = 1.0
    offset: float = 0.0


def _lab(*itemids: int, **kwargs: Any) -> _Item:
    return _Item("labevents", itemids, **kwargs)


def _chart(*itemids: int, **kwargs: Any) -> _Item:
    return _Item("chartevents", itemids, **kwargs)


# feature: (min, max, items). Out-of-range values are dropped. The item IDs,
# units and bounds of the 12 continuous variables are copied from the YAIB
# MIMIC-IV definitions (pyhealth/tasks/yaib_mimic4.py, ported from ricu), so
# this task stays self-contained; the Glasgow scales are ordinal scores from
# ``valuenum``.
_MEASURED_FEATURES: dict[str, tuple[float | None, float | None, list[_Item]]] = {
    "crr": (0, 1, [_chart(223951, 224308)]),
    "gcs_eye": (1, 4, [_chart(220739)]),
    "gcs_motor": (1, 6, [_chart(223901)]),
    "gcs_verbal": (1, 5, [_chart(223900)]),
    "dbp": (0, 200, [_chart(220051, 220180)]),
    "fio2": (21, 100, [_chart(223835), _lab(50816)]),
    "glu": (0, 1000, [_lab(50809, 50931)]),
    "hr": (0, 300, [_chart(220045)]),
    "map": (0, 250, [_chart(220052, 220181, 225312)]),
    "o2sat": (50, 100, [_chart(220277, 226253, 220227), _lab(50817)]),
    "resp": (0, 120, [_chart(220210, 224688, 224689, 224690)]),
    "sbp": (0, 300, [_chart(220050, 220179)]),
    "temp": (
        32,
        42,
        [
            _chart(223762),
            _chart(223761, 224027, unit_regex="f", scale=5 / 9, offset=-160 / 9),
        ],
    ),
    "ph": (6.8, 8, [_lab(50820)]),
    "height": (10, 230, [_chart(226707, unit_regex="^in", scale=2.54)]),
    "weight": (1, 500, [_chart(226512)]),
}

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _item_lookup(
    features: dict[str, tuple[float | None, float | None, list[_Item]]],
) -> pl.DataFrame:
    """Flattens feature definitions into one row per (table, itemid)."""
    rows = [
        {
            "table": item.table,
            "itemid": itemid,
            "feature": index,
            "unit_regex": (
                None if item.unit_regex is None else "(?i)" + item.unit_regex
            ),
            "scale": item.scale,
            "offset": item.offset,
            "lower": None if lower is None else float(lower),
            "upper": None if upper is None else float(upper),
        }
        for index, (lower, upper, items) in enumerate(features.values())
        for item in items
        for itemid in item.itemids
    ]
    return pl.DataFrame(
        rows,
        schema={
            "table": pl.String,
            "itemid": pl.Int64,
            "feature": pl.Int64,
            "unit_regex": pl.String,
            "scale": pl.Float64,
            "offset": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
        },
    )


def _parse_time(column: str) -> pl.Expr:
    return pl.col(column).str.strptime(pl.Datetime("ms"), _TIME_FORMAT, strict=False)


_MEASURED_NAMES = list(_MEASURED_FEATURES)
_ITEMS = _item_lookup(_MEASURED_FEATURES)
# Capillary refill is charted as text only.
_CRR_ITEMIDS = _MEASURED_FEATURES["crr"][2][0].itemids
_EVENT_TABLES = ("chartevents", "labevents")

# The 25 phenotypes of Harutyunyan et al. (2019) and their CCS category.
_PHENOTYPE_CCS = {
    "2": "Septicemia (except in labor)",
    "49": "Diabetes mellitus without complication",
    "50": "Diabetes mellitus with complications",
    "53": "Disorders of lipid metabolism",
    "55": "Fluid and electrolyte disorders",
    "98": "Essential hypertension",
    "99": "Hypertension with complications and secondary hypertension",
    "100": "Acute myocardial infarction",
    "101": "Coronary atherosclerosis and other heart disease",
    "105": "Conduction disorders",
    "106": "Cardiac dysrhythmias",
    "108": "Congestive heart failure; nonhypertensive",
    "109": "Acute cerebrovascular disease",
    "122": (
        "Pneumonia (except that caused by tuberculosis or sexually "
        "transmitted disease)"
    ),
    "127": "Chronic obstructive pulmonary disease and bronchiectasis",
    "130": "Pleurisy; pneumothorax; pulmonary collapse",
    "131": "Respiratory failure; insufficiency; arrest (adult)",
    "133": "Other lower respiratory disease",
    "134": "Other upper respiratory disease",
    "151": "Other liver diseases",
    "153": "Gastrointestinal hemorrhage",
    "157": "Acute and unspecified renal failure",
    "158": "Chronic kidney disease",
    "238": "Complications of surgical procedures or medical care",
    "249": "Shock",
}


class _OptionalImageProcessor(FeatureProcessor):
    """An image processor for the ``cxr`` input that accepts a missing image.

    A stay without a paired image has ``cxr=None`` and ``cxr_mask=0``. The
    missing image becomes a zero image of the output shape, as MedFuse expects
    a tensor for every sample and ignores it where the mask is 0. A present
    image goes through :class:`~pyhealth.processors.ImageProcessor` unchanged.

    Args:
        image_size: Output height and width.
        mode: PIL mode to convert images to: 'L', 'RGB' or 'RGBA'.
        **kwargs: Other :class:`~pyhealth.processors.ImageProcessor` options.

    Examples:
        >>> p = _OptionalImageProcessor(image_size=8, mode="L")
        >>> p.process(None).shape
        torch.Size([1, 8, 8])
    """

    stores_tensor = True
    _CHANNELS: ClassVar[dict[str, int]] = {"L": 1, "RGB": 3, "RGBA": 4}

    def __init__(self, image_size: int = 224, mode: str = "RGB", **kwargs: Any):
        if mode not in self._CHANNELS:
            raise ValueError(f"mode must be one of {list(self._CHANNELS)}.")
        self.image = ImageProcessor(image_size=image_size, mode=mode, **kwargs)
        if not self.image.stores_tensor:
            raise ValueError("to_tensor must be True.")
        self.shape = (self._CHANNELS[mode], image_size, image_size)

    def process(self, value: Any) -> torch.Tensor:
        if value is None:
            return torch.zeros(self.shape)
        return self.image.process(value)

    def is_token(self) -> bool:
        return False

    def dim(self) -> tuple[int, ...]:
        return self.image.dim()

    def spatial(self) -> tuple[bool, ...]:
        return self.image.spatial()

    def __repr__(self) -> str:
        return f"_OptionalImageProcessor({self.image!r})"


def _input_schema(image_size: int) -> dict[str, Any]:
    """Schema of the MedFuse inputs.

    Values are standardized on the training split, and values still missing
    (before a feature's first measurement) become 0, the training mean. The
    image is replicated to three channels, with ImageNet statistics as the
    CXR backbone is an ImageNet ResNet.
    """
    return {
        "ehr": (
            "tensor",
            {"normalization": "zscore", "impute_strategy": "zero"},
        ),
        "cxr": (
            _OptionalImageProcessor,
            {
                "image_size": image_size,
                "mode": "RGB",
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ),
        "cxr_mask": "tensor",
    }


def _pre_filter(
    df: pl.LazyFrame, task_name: str, tables: Sequence[str]
) -> pl.LazyFrame:
    """Checks the required tables and drops unused measurement rows."""
    columns = df.collect_schema().names()
    missing = [
        table
        for table in tables
        if not any(column.startswith(f"{table}/") for column in columns)
    ]
    if missing:
        raise ValueError(
            f"{task_name} requires the MIMIC-IV tables {missing}; add them to "
            "the dataset's tables."
        )
    keep = ~pl.col("event_type").is_in(list(_EVENT_TABLES))
    for table in _EVENT_TABLES:
        itemids = _ITEMS.filter(pl.col("table") == table)["itemid"].to_list()
        keep = keep | (
            (pl.col("event_type") == table)
            & pl.col(f"{table}/itemid").is_in([str(i) for i in sorted(set(itemids))])
        )
    return df.filter(keep)


def _eligible_stays(patient: Any) -> pl.DataFrame:
    """Applies the benchmark's stay exclusions, in the benchmark's order.

    Removes stays that changed care unit, then stays of admissions with more
    than one remaining stay, then stays of patients younger than 18.
    """
    demographics = patient.get_events(event_type="patients")
    if not demographics:
        return pl.DataFrame()
    demographics = demographics[0]
    if demographics.anchor_age is None or demographics.anchor_year is None:
        return pl.DataFrame()
    birth_offset = int(demographics.anchor_age) - int(demographics.anchor_year)

    stays = patient.get_events(event_type="icustays", return_df=True)
    if stays.height == 0:
        return pl.DataFrame()
    return (
        stays.select(
            pl.col("icustays/stay_id").cast(pl.Int64).alias("stay_id"),
            pl.col("icustays/hadm_id").cast(pl.Int64).alias("hadm_id"),
            pl.col("timestamp").alias("intime"),
            _parse_time("icustays/outtime").alias("outtime"),
            pl.col("icustays/first_careunit").alias("first_careunit"),
            pl.col("icustays/last_careunit").alias("last_careunit"),
        )
        .drop_nulls(["stay_id", "hadm_id", "intime", "outtime"])
        .filter(
            (pl.col("outtime") > pl.col("intime"))
            & (pl.col("first_careunit") == pl.col("last_careunit"))
        )
        .filter(pl.len().over("hadm_id") == 1)
        .filter(pl.col("intime").dt.year() + birth_offset >= 18)
        .select("stay_id", "hadm_id", "intime", "outtime")
        .sort("intime")
    )


def _measurements(patient: Any) -> pl.DataFrame:
    """Returns measurements as (timestamp, stay_id, feature, value)."""
    frames = []
    for table in _EVENT_TABLES:
        items = _ITEMS.filter(pl.col("table") == table)
        events = patient.get_events(event_type=table, return_df=True)
        if events.height == 0:
            continue
        frames.append(
            events.select(
                pl.col("timestamp"),
                (
                    pl.col("chartevents/stay_id").cast(pl.Int64)
                    if table == "chartevents"
                    else pl.lit(None, dtype=pl.Int64)
                ).alias("stay_id"),
                pl.col(f"{table}/itemid").cast(pl.Int64).alias("itemid"),
                pl.col(f"{table}/valuenum")
                .cast(pl.Float64, strict=False)
                .alias("value"),
                pl.col(f"{table}/value").alias("text"),
                pl.col(f"{table}/valueuom").alias("unit"),
            ).join(items, on="itemid", how="inner")
        )
    schema = {
        "timestamp": pl.Datetime("ms"),
        "stay_id": pl.Int64,
        "feature": pl.Int64,
        "value": pl.Float64,
    }
    if not frames:
        return pl.DataFrame(schema=schema)

    converted = pl.col("unit").str.contains(pl.col("unit_regex")).fill_null(False)
    capillary_refill = (
        pl.when(pl.col("text").str.starts_with("Abnormal"))
        .then(1.0)
        .when(pl.col("text").str.starts_with("Normal"))
        .then(0.0)
    )
    return (
        pl.concat(frames)
        .with_columns(
            pl.when(pl.col("itemid").is_in(list(_CRR_ITEMIDS)))
            .then(capillary_refill)
            .when(converted)
            .then(pl.col("value") * pl.col("scale") + pl.col("offset"))
            .otherwise(pl.col("value"))
            .alias("value")
        )
        .filter(
            pl.col("value").is_not_null()
            & pl.col("value").is_not_nan()
            & (pl.col("lower").is_null() | (pl.col("value") >= pl.col("lower")))
            & (pl.col("upper").is_null() | (pl.col("value") <= pl.col("upper")))
        )
        .select(list(schema))
    )


def _time_series(
    measurements: pl.DataFrame,
    stay_id: int,
    start: datetime,
    end: datetime,
    bin_hours: int,
) -> np.ndarray | None:
    """Builds the time series of measurements in ``[start, end)``.

    Chart rows of the stay timed before ``start`` count towards the first bin:
    MIMIC-IV charts admission height and weight just before ``intime``.

    Each bin holds the last measurement of each feature, as in the benchmark's
    discretizer. Features are forward-filled, staying NaN before their first
    measurement, and followed by a mask of the values measured in that bin.

    Args:
        measurements: Output of ``_measurements``.
        stay_id: The ICU stay.
        start: Start of the first bin.
        end: End of the observation window.
        bin_hours: Width of a bin in hours.

    Returns:
        A ``[bins, 2 * 17]`` array, or None without any measurement.
    """
    rows = measurements.filter(
        (pl.col("timestamp") < end)
        & ((pl.col("timestamp") >= start) | (pl.col("stay_id") == stay_id))
    )
    if rows.height == 0:
        return None
    bin_seconds = bin_hours * 3600
    bins = ceil((end - start).total_seconds() / bin_seconds)
    last = (
        rows.sort("timestamp")
        .with_columns(
            (
                (pl.col("timestamp") - start).dt.total_seconds() // bin_seconds
            ).clip(lower_bound=0).alias("bin")
        )
        .group_by("bin", "feature")
        .agg(pl.col("value").last())
    )
    values = np.full((bins, len(_MEASURED_NAMES) + 1), np.nan)
    values[last["bin"].to_numpy(), last["feature"].to_numpy()] = (
        last["value"].to_numpy()
    )
    scales = [_MEASURED_NAMES.index(f"gcs_{s}") for s in ("eye", "motor", "verbal")]
    values[:, -1] = values[:, scales].sum(axis=1)  # NaN unless all three are charted

    observed = ~np.isnan(values)
    latest = np.maximum.accumulate(
        np.where(observed, np.arange(bins)[:, None], 0), axis=0
    )
    filled = np.take_along_axis(values, latest, axis=0)
    return np.hstack([filled, observed]).astype(np.float32)


def _last_cxr(
    cxrs: pl.DataFrame,
    start: datetime,
    end: datetime,
    view_positions: Sequence[str],
) -> str | None:
    """Returns the path of the last image taken in ``[start, end]``."""
    if cxrs.height == 0:
        return None
    selected = cxrs.filter(
        (pl.col("timestamp") >= start)
        & (pl.col("timestamp") <= end)
        & pl.col("metadata/viewposition").is_in(list(view_positions))
    ).sort("timestamp", "metadata/dicom_id")
    return selected["metadata/image_path"][-1] if selected.height else None


@lru_cache(maxsize=1)
def _ccs_maps() -> dict[str, CrossMap]:
    """ICD to CCS maps by ``icd_version``, downloaded once by medcode."""
    return {
        "9": CrossMap.load("ICD9CM", "CCSCM"),
        "10": CrossMap.load("ICD10CM", "CCSCM"),
    }


class EHRCXRMortalityMIMIC4(BaseTask):
    """In-hospital mortality from the first 48 ICU hours and a chest X-ray.

    One sample per ICU stay of at least 48 hours, following MedFuse (Hayat et
    al., 2022) and the benchmark of Harutyunyan et al. (2019), on the
    composite :class:`~pyhealth.datasets.MIMIC4Dataset`.

    Stays: stays that changed care unit, then admissions with several
    remaining stays, then patients under 18 are removed, as are stays without
    any measurement in the window.

    ``ehr``: ``[24, 34]`` for the default 2-hour bins of the first 48 hours,
    holding the 17 MedFuse variables (:attr:`FEATURES`), forward-filled, then
    their measured-in-this-bin masks. Chart rows of the stay timed just before
    admission, as admission height and weight are, count towards the first
    bin. Statistics are learned when the processors are fitted, so use
    ``set_task(task, split=...)``.

    ``cxr``: the last image with a view in ``view_positions`` taken between
    ICU admission and the prediction time (48 hours after admission,
    inclusive). Images taken later never enter the sample.

    ``cxr_mask``: 1 with such an image, else 0 with a zero ``cxr``. Stays
    without an image are kept (the paper's partial cohort), unless
    ``paired_only`` is set (its paired cohort).

    ``mortality``: ``hospital_expire_flag`` of the stay's admission.

    Requires the ``chartevents`` and ``labevents`` EHR tables and the
    ``metadata`` CXR table.

    Args:
        image_size: Side of the square image.
        view_positions: Accepted ``ViewPosition`` values.
        bin_hours: Width of a time bin in hours.
        paired_only: Whether to drop stays without an image.

    Examples:
        >>> from pyhealth.tasks import EHRCXRMortalityMIMIC4
        >>> task = EHRCXRMortalityMIMIC4(image_size=224)
        >>> len(task.FEATURES)
        17
        >>> sorted(task.input_schema), task.output_schema
        (['cxr', 'cxr_mask', 'ehr'], {'mortality': 'binary'})
        >>> from pyhealth.datasets import MIMIC4Dataset, PatientSplit
        >>> dataset = MIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     cxr_root="/path/to/mimic-cxr-jpg/2.0.0",
        ...     ehr_tables=["chartevents", "labevents"],
        ...     cxr_tables=["metadata"],
        ... )
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0)
        ... )
    """

    task_name: str = "EHRCXRMortalityMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"mortality": "binary"}

    FEATURES: ClassVar[list[str]] = [*_MEASURED_NAMES, "gcs_total"]
    OBSERVATION_HOURS: ClassVar[int] = 48

    def __init__(
        self,
        image_size: int = 224,
        view_positions: Sequence[str] = ("AP",),
        bin_hours: int = 2,
        paired_only: bool = False,
    ) -> None:
        self.input_schema = _input_schema(image_size)
        super().__init__()
        self.view_positions = list(view_positions)
        self.bin_hours = bin_hours
        self.paired_only = paired_only

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops measurement rows whose item ID is not used by the task."""
        return _pre_filter(df, self.task_name, [*_EVENT_TABLES, "metadata"])

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the samples of one patient.

        Args:
            patient: The patient.

        Returns:
            One sample per eligible stay, with ``patient_id``, ``visit_id``,
            ``stay_id``, ``ehr``, ``cxr`` (a path or None), ``cxr_mask`` and
            ``mortality``.
        """
        stays = _eligible_stays(patient)
        if stays.height == 0:
            return []
        measurements = _measurements(patient)
        cxrs = patient.get_events(event_type="metadata", return_df=True)
        expired = {
            int(event.hadm_id): int(event.hospital_expire_flag)
            for event in patient.get_events(event_type="admissions")
            if event.hospital_expire_flag is not None
        }

        window = timedelta(hours=self.OBSERVATION_HOURS)
        samples = []
        for stay_id, hadm_id, intime, outtime in stays.iter_rows():
            if outtime - intime < window or hadm_id not in expired:
                continue
            end = intime + window
            ehr = _time_series(measurements, stay_id, intime, end, self.bin_hours)
            cxr = _last_cxr(cxrs, intime, end, self.view_positions)
            if ehr is None or (self.paired_only and cxr is None):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(hadm_id),
                    "stay_id": stay_id,
                    "ehr": ehr,
                    "cxr": cxr,
                    "cxr_mask": float(cxr is not None),
                    "mortality": expired[hadm_id],
                }
            )
        return samples


class EHRCXRPhenotypingMIMIC4(BaseTask):
    """25-label phenotyping of an ICU stay from its time series and a chest X-ray.

    One sample per ICU stay, following MedFuse (Hayat et al., 2022) and the
    benchmark of Harutyunyan et al. (2019), on the composite
    :class:`~pyhealth.datasets.MIMIC4Dataset`.

    Stays, ``ehr``, ``cxr`` and ``cxr_mask`` are built as in
    :class:`EHRCXRMortalityMIMIC4`, over the whole ICU stay instead of its
    first 48 hours: ``ehr`` is ``[T, 34]`` with ``T`` depending on the stay,
    and ``cxr`` is the last image taken during the stay.

    ``phenotypes``: the :attr:`PHENOTYPES` among the ICD codes of the stay's
    admission, a list that the ``multilabel`` processor turns into 25 binary
    labels. Codes go to CCS categories through
    :class:`~pyhealth.medcode.CrossMap`, which downloads its mappings once.

    :class:`~pyhealth.models.MedFuse` feeds a batch to its LSTM without
    packing, so a stay shorter than the longest of its batch ends in zero
    padding. Keep this in mind when comparing with the paper's results.

    Requires the ``chartevents``, ``labevents`` and ``diagnoses_icd`` EHR
    tables and the ``metadata`` CXR table.

    Args:
        image_size: Side of the square image.
        view_positions: Accepted ``ViewPosition`` values.
        bin_hours: Width of a time bin in hours.
        paired_only: Whether to drop stays without an image.

    Examples:
        >>> from pyhealth.tasks import EHRCXRPhenotypingMIMIC4
        >>> task = EHRCXRPhenotypingMIMIC4(image_size=224)
        >>> len(task.PHENOTYPES)
        25
        >>> task.PHENOTYPES[0]
        'Septicemia (except in labor)'
        >>> from pyhealth.datasets import MIMIC4Dataset, PatientSplit
        >>> dataset = MIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     cxr_root="/path/to/mimic-cxr-jpg/2.0.0",
        ...     ehr_tables=["chartevents", "labevents", "diagnoses_icd"],
        ...     cxr_tables=["metadata"],
        ... )
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0)
        ... )
    """

    task_name: str = "EHRCXRPhenotypingMIMIC4"

    FEATURES: ClassVar[list[str]] = [*_MEASURED_NAMES, "gcs_total"]
    PHENOTYPES: ClassVar[list[str]] = list(_PHENOTYPE_CCS.values())
    output_schema: ClassVar[dict[str, Any]] = {
        "phenotypes": ("multilabel", {"labels": PHENOTYPES})
    }

    def __init__(
        self,
        image_size: int = 224,
        view_positions: Sequence[str] = ("AP",),
        bin_hours: int = 2,
        paired_only: bool = False,
    ) -> None:
        self.input_schema = _input_schema(image_size)
        super().__init__()
        self.view_positions = list(view_positions)
        self.bin_hours = bin_hours
        self.paired_only = paired_only

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops measurement rows whose item ID is not used by the task."""
        return _pre_filter(
            df, self.task_name, [*_EVENT_TABLES, "diagnoses_icd", "metadata"]
        )

    @staticmethod
    def _phenotypes(patient: Any, hadm_id: int) -> list[str]:
        """Returns the benchmark phenotypes coded for an admission."""
        maps = _ccs_maps()
        codes = (
            patient.get_events(event_type="diagnoses_icd", return_df=True)
            .filter(pl.col("diagnoses_icd/hadm_id").cast(pl.Int64) == hadm_id)
            .select("diagnoses_icd/icd_version", "diagnoses_icd/icd_code")
            .unique()
        )
        labels = set()
        for version, code in codes.iter_rows():
            for category in maps[version].map(code):
                if category in _PHENOTYPE_CCS:
                    labels.add(_PHENOTYPE_CCS[category])
        return sorted(labels)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the samples of one patient.

        Args:
            patient: The patient.

        Returns:
            One sample per eligible stay, with ``patient_id``, ``visit_id``,
            ``stay_id``, ``ehr``, ``cxr`` (a path or None), ``cxr_mask`` and
            ``phenotypes``.
        """
        stays = _eligible_stays(patient)
        if stays.height == 0:
            return []
        measurements = _measurements(patient)
        cxrs = patient.get_events(event_type="metadata", return_df=True)

        samples = []
        for stay_id, hadm_id, intime, outtime in stays.iter_rows():
            ehr = _time_series(
                measurements, stay_id, intime, outtime, self.bin_hours
            )
            cxr = _last_cxr(cxrs, intime, outtime, self.view_positions)
            if ehr is None or (self.paired_only and cxr is None):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(hadm_id),
                    "stay_id": stay_id,
                    "ehr": ehr,
                    "cxr": cxr,
                    "cxr_mask": float(cxr is not None),
                    "phenotypes": self._phenotypes(patient, hadm_id),
                }
            )
        return samples
