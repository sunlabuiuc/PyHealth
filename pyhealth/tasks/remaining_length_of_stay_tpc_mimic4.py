from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import polars as pl

from .base_task import BaseTask


@dataclass(frozen=True)
class _TPCWindow:
    """ICU stay time boundaries used for prefill, resampling, and labels."""

    prefill_start: datetime
    icu_start: datetime
    pred_start: datetime
    pred_end: datetime


class RemainingLengthOfStayTPC_MIMIC4(BaseTask):
    """Paper-setting remaining ICU LoS (MIMIC-IV) for TPC.

    This task matches the *paper setting* (Rocheteau et al., CHIL 2021;
    `arXiv:2007.09483 <https://arxiv.org/abs/2007.09483>`__, PDF
    `https://arxiv.org/pdf/2007.09483` ) for MIMIC-IV:
    predict *remaining* ICU LoS at hourly intervals, starting at `start_hour` hours
    after ICU admission, truncated to the first `max_hours` hours of the stay.

    Output samples are designed for a dedicated TPC processor which performs:
    hourly resampling, forward-fill, and decay-indicator construction, while allowing
    pre-ICU values to be used for forward-fill and then removed.

    Notes:
    - We intentionally do **not** include diagnoses for MIMIC-IV (paper §4.2).
    - Time series use ``chartevents`` / ``labevents`` with user-configured itemids.

    Required dataset tables:
    - patients, admissions, icustays
    - chartevents and/or labevents (depending on provided itemids)

    Attributes:
        input_schema (Dict[str, Any]): Built per instance from the itemid lists passed
            to the constructor. Maps feature keys to ``(processor_name, kwargs)`` pairs.

            .. code-block:: python

                {
                    "ts": ("tpc_timeseries", {}),
                    "static": ("tpc_static", {}),
                }

        output_schema (Dict[str, Any]): Declared next to ``input_schema`` with the same
            pattern for labels.

            .. code-block:: python

                {
                    "y": ("regression_sequence", {}),
                }
    """

    task_name: str = "RemainingLengthOfStayTPC_MIMIC4"

    # Common MIMIC-IV itemids (Chartevents) for static proxies (optional).
    # These are intentionally kept minimal and can be overridden by the user.
    DEFAULT_HEIGHT_ITEMIDS: ClassVar[List[str]] = ["226730"]  # height (cm)
    DEFAULT_WEIGHT_ITEMIDS: ClassVar[List[str]] = ["226512"]  # weight (kg)
    DEFAULT_GCS_EYE_ITEMIDS: ClassVar[List[str]] = ["220739"]
    DEFAULT_GCS_VERBAL_ITEMIDS: ClassVar[List[str]] = ["223900"]
    DEFAULT_GCS_MOTOR_ITEMIDS: ClassVar[List[str]] = ["223901"]

    def __init__(
        self,
        *,
        start_hour: int = 5,
        max_hours: int = 14 * 24,
        pre_icu_hours: int = 24,
        min_icu_hours: int = 5,
        chartevent_itemids: Optional[Sequence[str]] = None,
        labevent_itemids: Optional[Sequence[str]] = None,
        static_height_itemids: Optional[Sequence[str]] = None,
        static_weight_itemids: Optional[Sequence[str]] = None,
        static_gcs_eye_itemids: Optional[Sequence[str]] = None,
        static_gcs_verbal_itemids: Optional[Sequence[str]] = None,
        static_gcs_motor_itemids: Optional[Sequence[str]] = None,
    ) -> None:
        self.start_hour = int(start_hour)
        self.max_hours = int(max_hours)
        self.pre_icu_hours = int(pre_icu_hours)
        self.min_icu_hours = int(min_icu_hours)

        self.chartevent_itemids = [str(x) for x in (chartevent_itemids or [])]
        self.labevent_itemids = [str(x) for x in (labevent_itemids or [])]
        self.feature_itemids = self.chartevent_itemids + self.labevent_itemids

        self.static_height_itemids = [
            str(x) for x in (static_height_itemids or self.DEFAULT_HEIGHT_ITEMIDS)
        ]
        self.static_weight_itemids = [
            str(x) for x in (static_weight_itemids or self.DEFAULT_WEIGHT_ITEMIDS)
        ]
        self.static_gcs_eye_itemids = [
            str(x) for x in (static_gcs_eye_itemids or self.DEFAULT_GCS_EYE_ITEMIDS)
        ]
        self.static_gcs_verbal_itemids = [
            str(x)
            for x in (static_gcs_verbal_itemids or self.DEFAULT_GCS_VERBAL_ITEMIDS)
        ]
        self.static_gcs_motor_itemids = [
            str(x) for x in (static_gcs_motor_itemids or self.DEFAULT_GCS_MOTOR_ITEMIDS)
        ]

        # Input/Output schemas use explicit processor registrations.
        # - ts: custom TPC time-series processor will produce (T, F, 2)
        # - static: custom TPC static processor will encode/scale features
        # - y: custom regression-sequence label processor will produce (T,)
        self.input_schema: Dict[str, Any] = {
            "ts": ("tpc_timeseries", {}),
            "static": ("tpc_static", {}),
        }
        self.output_schema: Dict[str, Any] = {
            "y": ("regression_sequence", {}),
        }

    def _get_admission_for_stay(self, patient: Any, hadm_id: str) -> Optional[Any]:
        """Return the admissions row for ``hadm_id``, or ``None`` if missing."""
        admissions = patient.get_events(
            event_type="admissions",
            filters=[("hadm_id", "==", hadm_id)],
        )
        if not admissions:
            return None
        # Choose the first match (should be unique).
        return admissions[0]

    def _build_window(
        self, icu_start: datetime, icu_end: datetime
    ) -> Optional[_TPCWindow]:
        """Compute the prediction window for a single ICU stay."""
        if icu_end <= icu_start:
            return None  # malformed data
        duration_hours = (icu_end - icu_start).total_seconds() / 3600.0
        if duration_hours < self.min_icu_hours:
            return None  # shorter than minimum ICU length

        prefill_start = icu_start - timedelta(hours=self.pre_icu_hours)
        pred_start = icu_start + timedelta(hours=self.start_hour)
        pred_end_cap = icu_start + timedelta(hours=self.max_hours)
        pred_end = min(icu_end, pred_end_cap)
        if pred_end <= pred_start:
            return None

        return _TPCWindow(
            prefill_start=prefill_start,
            icu_start=icu_start,
            pred_start=pred_start,
            pred_end=pred_end,
        )

    def _extract_static_from_events(
        self,
        patient: Any,
        *,
        stay: Any,
        admission: Optional[Any],
        icu_start: datetime,
        prefill_start: datetime,
    ) -> Dict[str, Any]:
        """Extract the 12 static features from MIMIC-IV tables for one ICU stay.

        Produces a raw dict for ``TPCStaticProcessor``.

        Args:
            patient: Patient record with PyHealth event accessors.
            stay: ``icustays`` row for this ICU episode.
            admission: Matching ``admissions`` row, or ``None``.
            icu_start: ICU ``intime`` (used for age and chart window).
            prefill_start: Start of the pre-ICU lookback for early vitals.

        Returns:
            Mapping of static field names to raw values before encoding.

        Note:
            Height, weight, and GCS proxies use the first chart value in
            ``[prefill_start, icu_start + 1 hour]``.
        """
        static: Dict[str, Any] = {}

        # Table 6 (paper) core fields.
        demographics = patient.get_events(event_type="patients")
        if demographics:
            demo = demographics[0]
            static["gender"] = getattr(demo, "gender", None)
            # static["anchor_age"] = getattr(demo, "anchor_age", None)
            # Compute age at ICU admission
            # Age from ICU intime calendar year vs patient anchor_year.
            try:
                anchor_age = int(demo.anchor_age)
                anchor_year = int(demo.anchor_year)
                static["anchor_age"] = anchor_age + (icu_start.year - anchor_year)
            except Exception:
                static["anchor_age"] = None


        if admission is not None:
            static["race"] = getattr(admission, "race", None)
            static["admission_location"] = getattr(
                admission, "admission_location", None
            )
            static["insurance"] = getattr(admission, "insurance", None)

        static["first_careunit"] = getattr(stay, "first_careunit", None)
        static["hour_of_admission"] = int(icu_start.hour)

        # Approximate admission height/weight/GCS from early chartevents: first value in
        # [prefill_start, icu_start + 1h].
        early_end = icu_start + timedelta(hours=1)
        ce_df = patient.get_events(
            event_type="chartevents",
            start=prefill_start,
            end=early_end,
            return_df=True,
        )
        if ce_df is not None and ce_df.height > 0:
            ce_df = ce_df.select(
                pl.col("timestamp"),
                pl.col("chartevents/itemid").cast(pl.Utf8),
                pl.col("chartevents/valuenum").cast(pl.Float64),
            ).drop_nulls(["timestamp", "chartevents/itemid"])
            if ce_df.height > 0:
                # Take first value per itemid by time.
                ce_df = ce_df.sort("timestamp")

                def first_item_value(itemids: Sequence[str]) -> Optional[float]:
                    """First matching ``valuenum`` in ``ce_df``, else ``None``."""
                    sub = ce_df.filter(
                        pl.col("chartevents/itemid").is_in([str(x) for x in itemids])
                    )
                    if sub.height == 0:
                        return None
                    # first non-null value
                    sub = sub.drop_nulls(["chartevents/valuenum"])
                    if sub.height == 0:
                        return None
                    return float(sub["chartevents/valuenum"][0])

                static["admission_height"] = first_item_value(
                    self.static_height_itemids
                )
                static["admission_weight"] = first_item_value(
                    self.static_weight_itemids
                )
                static["gcs_eye"] = first_item_value(self.static_gcs_eye_itemids)
                static["gcs_verbal"] = first_item_value(self.static_gcs_verbal_itemids)
                static["gcs_motor"] = first_item_value(self.static_gcs_motor_itemids)

        return static

    def _extract_timeseries(
        self,
        patient: Any,
        *,
        prefill_start: datetime,
        pred_end: datetime,
        stay_id: str,
    ) -> Tuple[List[datetime], Dict[str, List[Any]]]:
        """Return irregular observations for requested itemids.

        The returned dataframe is in long format: (timestamp, itemid, value, source).
        """
        frames: List[pl.DataFrame] = []

        if self.chartevent_itemids:
            ce = patient.get_events(
                event_type="chartevents",
                start=prefill_start,
                end=pred_end,
                filters=[("stay_id", "==", stay_id)],
                return_df=True,
            )
            if ce is not None and ce.height > 0:
                ce = (
                    ce.select(
                        pl.col("timestamp"),
                        pl.col("chartevents/itemid").cast(pl.Utf8).alias("itemid"),
                        pl.col("chartevents/valuenum").cast(pl.Float64).alias("value"),
                    )
                    .filter(pl.col("itemid").is_in(self.chartevent_itemids))
                    .drop_nulls(["timestamp", "itemid"])
                    .with_columns(pl.lit("chartevents").alias("source"))
                )
                if ce.height > 0:
                    frames.append(ce)

        if self.labevent_itemids:
            le = patient.get_events(
                event_type="labevents",
                start=prefill_start,
                end=pred_end,
                return_df=True,
            )
            if le is not None and le.height > 0:
                le = (
                    le.select(
                        pl.col("timestamp"),
                        pl.col("labevents/itemid").cast(pl.Utf8).alias("itemid"),
                        pl.col("labevents/valuenum").cast(pl.Float64).alias("value"),
                        pl.col("labevents/hadm_id").cast(pl.Utf8).alias("hadm_id"),
                    )
                    .filter(pl.col("itemid").is_in(self.labevent_itemids))
                    .drop_nulls(["timestamp", "itemid"])
                    .with_columns(pl.lit("labevents").alias("source"))
                )
                if le.height > 0:
                    frames.append(le.drop("hadm_id"))

        if not frames:
            return [], {"timestamp": [], "itemid": [], "value": [], "source": []}

        df = pl.concat(frames, how="vertical").sort("timestamp")
        timestamps = df["timestamp"].to_list()
        # Convert to a pure-Python payload for robust pickling in task caching.
        payload = df.select(
            "timestamp", "itemid", "value", "source"
        ).to_dict(as_series=False)
        return timestamps, payload

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Build TPC training samples for every qualifying ICU stay on ``patient``.

        Args:
            patient: PyHealth patient with MIMIC-IV tables attached.

        Returns:
            List of sample dicts with ``patient_id``, ``stay_id``, ``hadm_id``,
            ``static``, ``ts``, and ``y``. Empty when no stay passes cohort and
            data-availability checks.
        """
        # Adult cohort: anchor age/year fetched once per patient.
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        try:
            anchor_age = int(demographics[0].anchor_age)
            anchor_year = int(demographics[0].anchor_year)
        except Exception:
            return []


        stays = patient.get_events(event_type="icustays")
        if not stays:
            return []

        samples: List[Dict[str, Any]] = []
        for stay in stays:
            try:
                icu_start: datetime = stay.timestamp
                icu_end: datetime = datetime.strptime(stay.outtime, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            # cohort filter: age at this specific ICU admission
            if anchor_age + (icu_start.year - anchor_year) < 18:
                continue


            window = self._build_window(icu_start, icu_end)
            if window is None:
                continue

            stay_id = str(getattr(stay, "stay_id", ""))
            hadm_id = str(getattr(stay, "hadm_id", ""))
            if not stay_id or not hadm_id:
                continue

            admission = self._get_admission_for_stay(patient, hadm_id)

            # Require at least one requested feature id (otherwise model has no inputs).
            if len(self.feature_itemids) == 0:
                continue

            ts_timestamps, ts_long_payload = self._extract_timeseries(
                patient,
                prefill_start=window.prefill_start,
                pred_end=window.pred_end,
                stay_id=stay_id,
            )
            if not ts_timestamps:
                continue

            static = self._extract_static_from_events(
                patient,
                stay=stay,
                admission=admission,
                icu_start=window.icu_start,
                prefill_start=window.prefill_start,
            )

            # Labels: remaining ICU LoS (days) per hour in [pred_start, pred_end).
            total_hours = int(
                (window.pred_end - window.pred_start).total_seconds() // 3600
            )
            if total_hours <= 0:
                continue
            y = []
            for h in range(total_hours):
                t = window.pred_start + timedelta(hours=h)
                rem_days = (icu_end - t).total_seconds() / 86400.0
                # Remaining LoS is positive by construction; enforce the same lower clip
                # used by the paper's output clipping (30 minutes = 1/48 days).
                y.append(max(rem_days, 1.0 / 48.0))

            sample: Dict[str, Any] = {
                "patient_id": patient.patient_id,
                "stay_id": stay_id,
                "hadm_id": hadm_id,
                "static": static,
                "ts": {
                    "prefill_start": window.prefill_start,
                    "icu_start": window.icu_start,
                    "pred_start": window.pred_start,
                    "pred_end": window.pred_end,
                    "feature_itemids": self.feature_itemids,
                    "long_df": ts_long_payload,  # dict[str, list]
                },
                "y": y,
            }
            samples.append(sample)

        return samples

