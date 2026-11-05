"""Disease progression between consecutive chest X-rays of an ICU stay.

Given two consecutive chest X-rays and the time series recorded around them,
identifies whether each of seven thoracic conditions worsened, improved or did
not change, using the region-level change labels of Chest ImaGenome.

Paper:
    Chen Liu, Wenfang Yao, Kejing Yin, William K. Cheung, and Jing Qin.
    "Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement
    and Multiscale Alignment." NeurIPS 2025. https://arxiv.org/abs/2510.11112

Image selection, label derivation and the EHR window are adapted from DiPro
(commit 64b12ec, ``mimic_data_extract/mimic3benchmark/scripts`` and
``dataset_module/dipro_dataset.py``, https://github.com/Chenliu-svg/DiPro),
MIT License, Copyright (c) 2025 LIU Chen @ The Hong Kong Polytechnic University.
"""

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from math import ceil, floor
from typing import Any, ClassVar

import numpy as np
import polars as pl

from pyhealth.data import Patient

from .base_task import BaseTask
from .ehr_cxr_mimic4 import (
    _EVENT_TABLES,
    _MEASURED_NAMES,
    _eligible_stays,
    _measurements,
    _pre_filter,
    _time_series,
)

_DISEASES = (
    "atelectasis",
    "enlarged cardiac silhouette",
    "consolidation",
    "pulmonary edema",
    "lung opacity",
    "pleural effusion",
    "pneumonia",
)
_REGIONS = (
    "left lung",
    "right lung",
    "left lower lung zone",
    "right lower lung zone",
    "left hilar structures",
    "right hilar structures",
    "left costophrenic angle",
    "right costophrenic angle",
    "cardiac silhouette",
    "mediastinum",
)
# DiPro's class order; "not mentioned" is the class that its loss ignores.
_STATUSES = ("worsened", "improved", "no change")
_NOT_MENTIONED = len(_STATUSES)
# Chest ImaGenome names the edema finding "pulmonary edema/hazy opacity".
_DISEASE_OF_FINDING = {
    **{disease: disease for disease in _DISEASES},
    "pulmonary edema/hazy opacity": "pulmonary edema",
}
_IMAGENOME_TABLES = ("silver_images", "silver_comparisons", "gold_comparisons")
_LIST_SEPARATOR = ";"


def _los_outlier_stays(df: pl.LazyFrame, min_los_hours: float) -> list[int]:
    """Returns the eligible stays outside the cohort's IQR length-of-stay limits.

    The limits are computed over all eligible stays longer than
    ``min_los_hours``, as in DiPro's ``get_cxr.py``, so they cannot be
    computed from one patient.
    """
    rows = (
        df.filter(pl.col("event_type").is_in(["patients", "icustays"]))
        .select(
            "patient_id",
            "event_type",
            "timestamp",
            pl.col("^(patients|icustays)/.*$"),
        )
        .collect(engine="streaming")
    )
    if rows.height == 0:
        return []
    stays = [
        _eligible_stays(Patient(patient_id=key[0], data_source=frame))
        for key, frame in rows.partition_by("patient_id", as_dict=True).items()
    ]
    stays = [frame for frame in stays if frame.height]
    if not stays:
        return []
    los = (
        pl.concat(stays)
        .with_columns(
            ((pl.col("outtime") - pl.col("intime")).dt.total_seconds() / 3600).alias(
                "los"
            )
        )
        .filter(pl.col("los") > min_los_hours)
    )
    if los.height == 0:
        return []
    first, third = np.quantile(los["los"].to_numpy(), [0.25, 0.75])
    spread = 1.5 * (third - first)
    return los.filter(
        (pl.col("los") <= first - spread) | (pl.col("los") >= third + spread)
    )["stay_id"].to_list()


def _latest_image_per_study(
    cxrs: pl.DataFrame, view_positions: Sequence[str]
) -> pl.DataFrame:
    """Returns the last image of each study with an accepted view."""
    if cxrs.height == 0:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ms"),
                "dicom_id": pl.String,
                "image_path": pl.String,
            }
        )
    return (
        cxrs.filter(
            pl.col("timestamp").is_not_null()
            & pl.col("metadata/viewposition").is_in(list(view_positions))
        )
        .sort("timestamp", "metadata/dicom_id")
        .group_by("metadata/study_id", maintain_order=True)
        .last()
        .select(
            "timestamp",
            pl.col("metadata/dicom_id").alias("dicom_id"),
            pl.col("metadata/image_path").alias("image_path"),
        )
    )


def _stay_images(
    images: pl.DataFrame,
    intime: datetime,
    outtime: datetime,
    scene_graphs: set[str],
    min_gap: timedelta,
    max_gap: timedelta,
    max_images: int,
) -> list[tuple[datetime, str, str]]:
    """Selects the images of a stay, as DiPro's ``align_cxr_with_stay`` does.

    Images need a scene graph, and a stay needs 2 to ``max_images`` of them. An
    image taken less than ``min_gap`` after the previous image of that list is
    removed, and a stay with a gap of ``max_gap`` or more is excluded. Gaps are
    not recomputed after a removal, and a stay may be left with one image.

    Returns:
        (timestamp, dicom_id, image_path) tuples in time order.
    """
    rows = (
        images.filter(
            pl.col("timestamp").is_between(intime, outtime)
            & pl.col("dicom_id").is_in(list(scene_graphs))
        )
        .sort("timestamp", "dicom_id")
        .rows()
    )
    if not 2 <= len(rows) <= max_images:
        return []
    gaps = [later[0] - earlier[0] for earlier, later in pairwise(rows)]
    if any(gap >= max_gap for gap in gaps):
        return []
    return rows[:1] + [row for row, gap in zip(rows[1:], gaps) if gap >= min_gap]


def _rows_by_pair(
    patient: Any, table: str, columns: Sequence[str]
) -> dict[tuple[str, str], list[tuple]]:
    """Groups a comparison table by (previous image, current image)."""
    rows = patient.get_events(event_type=table, return_df=True)
    pairs = defaultdict(list)
    if rows.height == 0:
        return pairs
    names = ["previous_dicom_id", "dicom_id", *columns]
    for previous, current, *values in rows.select(
        [f"{table}/{name}" for name in names]
    ).iter_rows():
        pairs[(previous, current)].append(tuple(values))
    return pairs


def _agreed(votes: dict[Hashable, set[str]]) -> dict[Hashable, str]:
    """Keeps the keys with a single status; conflicts count as not mentioned."""
    return {key: next(iter(v)) for key, v in votes.items() if len(v) == 1}


def _vote(
    votes: dict[str, set[str]],
    region_votes: dict[tuple[str, str], set[str]],
    region: str,
    finding: str,
    status: str,
) -> None:
    """Records a status for a finding, if it is one of the seven diseases."""
    disease = _DISEASE_OF_FINDING.get(finding)
    if disease is None or status not in _STATUSES:
        return
    votes[disease].add(status)
    if region in _REGIONS:
        region_votes[(region, disease)].add(status)


def _pair_labels(
    silver: Iterable[tuple], gold: Iterable[tuple]
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Derives the labels of a pair of images.

    Silver: a region with several comparisons is ambiguous and skipped, each
    remaining region gives its status to each of its findings, and a disease
    whose regions disagree is dropped.

    Gold rows give each disease its own status, with the same rule when
    regions disagree, and replace the silver labels of the pair. At the region
    level they only overwrite the regions and diseases they name.

    Args:
        silver: (bbox_name, findings, comparisons) rows of the pair.
        gold: (bbox_name, finding, comparison) rows of the pair.

    Returns:
        Status per disease, and status per (region, disease).
    """
    silver_votes, silver_regions = defaultdict(set), defaultdict(set)
    for region, findings, comparisons in silver:
        comparisons = (comparisons or "").split(_LIST_SEPARATOR)
        if len(comparisons) != 1:
            continue
        for finding in (findings or "").split(_LIST_SEPARATOR):
            _vote(silver_votes, silver_regions, region, finding, comparisons[0])

    gold_votes, gold_regions = defaultdict(set), defaultdict(set)
    for region, finding, comparison in gold:
        _vote(gold_votes, gold_regions, region, finding, comparison)

    labels = _agreed(gold_votes if gold else silver_votes)
    return labels, {**_agreed(silver_regions), **_agreed(gold_regions)}


class DiseaseProgressionMIMIC4(BaseTask):
    """Disease progression between two consecutive chest X-rays of an ICU stay.

    One sample per pair of consecutive images, following DiPro (Liu et al.,
    2025), on a :class:`~pyhealth.datasets.MIMIC4Dataset` built with
    ``imagenome_root``.

    Stays: as in :class:`EHRCXRMortalityMIMIC4`, then longer than 48 hours and
    inside the IQR limits of the cohort's length of stay. The limits need the
    whole cohort, so they apply in :meth:`pre_filter`, which ``set_task``
    calls, and not when calling the task on one patient.

    Images: the last AP image of each study, inside the stay, with a scene
    graph. A stay needs 2 to 14 of them and is excluded if two neighbours are
    96 hours or more apart. An image less than 8 hours after its predecessor
    is removed. Each pair of neighbours in what is left is a candidate. It is
    a sample only if Chest ImaGenome compares exactly those two images and
    labels at least one disease.

    ``cxr``: ``([first_path, second_path], [first_hour, second_hour])``, in
    hours since ICU admission, which the ``time_image`` processor turns into
    images ``[2, 3, H, W]`` and their times. Nothing later than the second
    image is used, so the sample never holds a later image.

    ``ehr``: ``[T, 34]``, one bin per hour, holding the 17 variables of
    :attr:`FEATURES` and then their measured-in-this-bin masks. Values are
    forward-filled from ICU admission, and the series is cut to the bins from
    the first image's to the second image's. Rows stop at the second image;
    ``include_bin_after_cxr2`` gives DiPro's window instead.

    ``progression``: the :attr:`LABELS` that apply, a list that the
    ``multilabel`` processor turns into 21 binary labels, three per disease in
    the order of :attr:`DISEASES` and :attr:`STATUSES`. A disease with none of
    its three set is "not mentioned", which DiPro's loss ignores, so a model
    trained with plain binary cross-entropy on all 21 treats those as
    negatives.

    Labels of a pair: a region with several comparisons is ambiguous and
    skipped, each other region gives its status to all its findings, and a
    disease whose regions disagree is dropped. Negated findings count, as in
    DiPro. If the gold comparisons cover the pair, the same rule runs on them
    alone, disease by disease.

    Extra fields, not model inputs: ``first_ehr_hour`` (hour of ``ehr``'s
    first bin), ``from_gold`` (1 for gold pairs), and ``region_progression``,
    the ``[10, 7]`` status per :attr:`REGIONS` and :attr:`DISEASES`, using
    the index in :attr:`STATUSES` and 3 for not mentioned. It is a label, so
    it must not be fed to a model as an input feature.

    To reserve the gold subjects for the test set as DiPro does, use
    :class:`~pyhealth.datasets.GoldTestSplit` with :meth:`gold_subjects`.

    Requires the ``chartevents`` and ``labevents`` EHR tables, the
    ``metadata`` CXR table, and the ``silver_images``, ``silver_comparisons``
    and ``gold_comparisons`` Chest ImaGenome tables.

    Args:
        image_size: Side of the square image.
        view_positions: Accepted ``ViewPosition`` values.
        include_bin_after_cxr2: Whether ``ehr`` also holds the bin after the
            second image's, as DiPro's window does. That can include
            measurements up to 2 hours after the second image.
        exclude_los_outliers: Whether to apply the IQR length-of-stay limits.
            They are strict, as in DiPro, so if every stay has the same length
            all of them are removed.

    Examples:
        >>> from pyhealth.tasks import DiseaseProgressionMIMIC4
        >>> task = DiseaseProgressionMIMIC4(image_size=224)
        >>> len(task.LABELS), task.LABELS[0]
        (21, 'atelectasis: worsened')
        >>> sorted(task.input_schema)
        ['cxr', 'ehr']
        >>> from pyhealth.datasets import GoldTestSplit, MIMIC4Dataset
        >>> dataset = MIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     cxr_root="/path/to/mimic-cxr-jpg/2.0.0",
        ...     imagenome_root="/path/to/chest-imagenome/1.0.0",
        ...     ehr_tables=["chartevents", "labevents"],
        ...     cxr_tables=["metadata"],
        ...     imagenome_tables=[
        ...         "silver_images", "silver_comparisons", "gold_comparisons"
        ...     ],
        ... )
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=GoldTestSplit(task.gold_subjects(dataset))
        ... )
    """

    task_name: str = "DiseaseProgressionMIMIC4"

    DISEASES: ClassVar[list[str]] = list(_DISEASES)
    REGIONS: ClassVar[list[str]] = list(_REGIONS)
    STATUSES: ClassVar[list[str]] = list(_STATUSES)
    LABELS: ClassVar[list[str]] = [
        f"{disease}: {status}" for disease in _DISEASES for status in _STATUSES
    ]
    FEATURES: ClassVar[list[str]] = [*_MEASURED_NAMES, "gcs_total"]
    output_schema: ClassVar[dict[str, Any]] = {
        "progression": ("multilabel", {"classes": LABELS})
    }

    MIN_LOS_HOURS: ClassVar[int] = 48
    MAX_IMAGES: ClassVar[int] = 14
    MIN_GAP_HOURS: ClassVar[int] = 8
    MAX_GAP_HOURS: ClassVar[int] = 96

    def __init__(
        self,
        image_size: int = 224,
        view_positions: Sequence[str] = ("AP",),
        include_bin_after_cxr2: bool = False,
        exclude_los_outliers: bool = True,
    ) -> None:
        self.input_schema = {
            "ehr": (
                "tensor",
                {"normalization": "zscore", "impute_strategy": "zero"},
            ),
            "cxr": ("time_image", {"image_size": image_size, "mode": "RGB"}),
        }
        super().__init__()
        self.view_positions = list(view_positions)
        self.include_bin_after_cxr2 = include_bin_after_cxr2
        self.exclude_los_outliers = exclude_los_outliers

    @staticmethod
    def gold_subjects(dataset: Any) -> list[str]:
        """Returns the subjects that have gold comparisons.

        Args:
            dataset: A dataset with the ``gold_comparisons`` table.

        Returns:
            The sorted subject IDs.
        """
        return (
            dataset.global_event_df.filter(
                pl.col("event_type") == "gold_comparisons"
            )
            .select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .sort()
            .to_list()
        )

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops unused measurements and stays with an outlying length of stay."""
        df = _pre_filter(
            df, self.task_name, [*_EVENT_TABLES, "metadata", *_IMAGENOME_TABLES]
        )
        if not self.exclude_los_outliers:
            return df
        outliers = _los_outlier_stays(df, self.MIN_LOS_HOURS)
        return df.filter(
            (pl.col("event_type") != "icustays")
            | ~pl.col("icustays/stay_id").cast(pl.Int64).is_in(outliers)
        )

    def _ehr(
        self,
        measurements: pl.DataFrame,
        stay_id: int,
        intime: datetime,
        outtime: datetime,
        first_time: datetime,
        second_time: datetime,
    ) -> tuple[np.ndarray, int] | None:
        """Builds the hourly series around a pair of images.

        DiPro keeps bin ``b`` when ``b - 1 <= second`` and ``b + 1 >= first``,
        in hours since admission.

        Returns:
            The series and its first bin, or None without any measurement up
            to the end of the window.
        """
        hour = timedelta(hours=1)
        end = second_time
        if self.include_bin_after_cxr2:
            end = min(intime + (floor((second_time - intime) / hour) + 2) * hour, outtime)
        series = _time_series(measurements, stay_id, intime, end, 1)
        if series is None:
            return None
        first_bin = max(ceil((first_time - intime) / hour) - 1, 0)
        return series[first_bin:], first_bin

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the samples of one patient.

        Args:
            patient: The patient.

        Returns:
            One sample per labeled pair of images, with ``patient_id``,
            ``visit_id``, ``stay_id``, ``ehr``, ``first_ehr_hour``, ``cxr``,
            ``progression``, ``region_progression`` and ``from_gold``.
        """
        stays = _eligible_stays(patient)
        if stays.height == 0:
            return []
        images = _latest_image_per_study(
            patient.get_events(event_type="metadata", return_df=True),
            self.view_positions,
        )
        scene_graphs = patient.get_events(event_type="silver_images", return_df=True)
        if images.height == 0 or scene_graphs.height == 0:
            return []
        scene_graphs = set(scene_graphs["silver_images/dicom_id"].drop_nulls())
        silver = _rows_by_pair(
            patient, "silver_comparisons", ("bbox_name", "findings", "comparisons")
        )
        gold = _rows_by_pair(
            patient, "gold_comparisons", ("bbox_name", "finding", "comparison")
        )
        measurements = _measurements(patient)

        min_los = timedelta(hours=self.MIN_LOS_HOURS)
        samples = []
        for stay_id, hadm_id, intime, outtime in stays.iter_rows():
            if outtime - intime <= min_los:
                continue
            selected = _stay_images(
                images,
                intime,
                outtime,
                scene_graphs,
                timedelta(hours=self.MIN_GAP_HOURS),
                timedelta(hours=self.MAX_GAP_HOURS),
                self.MAX_IMAGES,
            )
            for first, second in pairwise(selected):
                first_time, first_id, first_path = first
                second_time, second_id, second_path = second
                pair = (first_id, second_id)
                labels, regions = _pair_labels(
                    silver.get(pair, ()), gold.get(pair, ())
                )
                if not labels:
                    continue
                window = self._ehr(
                    measurements, stay_id, intime, outtime, first_time, second_time
                )
                if window is None:
                    continue
                ehr, first_bin = window
                hours = [
                    (time - intime).total_seconds() / 3600
                    for time in (first_time, second_time)
                ]
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": str(hadm_id),
                        "stay_id": stay_id,
                        "ehr": ehr,
                        "first_ehr_hour": first_bin,
                        "cxr": ([first_path, second_path], hours),
                        "progression": [
                            f"{disease}: {status}"
                            for disease, status in labels.items()
                        ],
                        "region_progression": [
                            [
                                _STATUSES.index(regions[(region, disease)])
                                if (region, disease) in regions
                                else _NOT_MENTIONED
                                for disease in _DISEASES
                            ]
                            for region in _REGIONS
                        ],
                        "from_gold": int(pair in gold),
                    }
                )
        return samples
