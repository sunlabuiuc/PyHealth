from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import polars as pl

from pyhealth.data import Patient

from .base_task import BaseTask


class SofaLabForecastingMIMIC3(BaseTask):
    """Forecast future SOFA-related lab values from early ICU measurements.

    Task-only adaptation of Staniek et al. (2024), "Early Prediction of Causes
    (not Effects) in Healthcare by Long-Term Clinical Time Series Forecasting."

    Inputs are hourly binned lab values from an observation window; outputs are
    hourly binned lab values from a future prediction window plus a derived
    binary SOFA-deterioration label. Three SOFA-related labs are modeled:
    bilirubin (50885), creatinine (50912), platelets (51265).

    Note:
        Unlike most PyHealth tasks, samples are emitted per ICU stay rather
        than per patient, so patients with multiple stays produce multiple
        samples. This matches the paper's per-stay forecasting setup.
        Also unlike the paper, no static features are included and no lookback
        it is a very simple adaptation of SOFA related labs for 3 values from MIMIC-III
        with a generalized standardization approach.

    Args:
        lookback_hours: Length of the observation window in hours.
        prediction_hours: Length of the future target window in hours.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import SofaLabForecastingMIMIC3
        >>> dataset = MIMIC3Dataset(root="/path/to/mimic3",
        ...                         tables=["labevents", "icustays"])
        >>> samples = dataset.set_task(SofaLabForecastingMIMIC3())
    """

    task_name: str = "SofaLabForecastingMIMIC3"

    input_schema: Dict[str, str] = {
        "observation_values": "tensor",
        "observation_masks": "tensor",
    }
    output_schema: Dict[str, str] = {
        "target_values": "tensor",
        "target_masks": "tensor",
        "sofa_label": "binary",
    }

    BILIRUBIN_ITEMID: str = "50885"
    CREATININE_ITEMID: str = "50912"
    PLATELETS_ITEMID: str = "51265"

    LAB_ITEMIDS: List[str] = [
        BILIRUBIN_ITEMID,
        CREATININE_ITEMID,
        PLATELETS_ITEMID,
    ]
    NUM_LABS: int = 3

    # Approximate MIMIC-III (mean, std) per lab from MIMIC-Extract
    # (Wang et al., 2020). Used only to standardize forecasting tensors;
    # raw values drive SOFA scoring. 
    # generalize forecasting from this paper is more consistent on the pattern
    LAB_NORMALIZATION_STATS: ClassVar[Dict[str, Tuple[float, float]]] = {
        BILIRUBIN_ITEMID: (2.6, 5.4),
        CREATININE_ITEMID: (1.4, 1.5),
        PLATELETS_ITEMID: (205.2, 113.3),
    }

    def __init__(
        self,
        lookback_hours: int = 24,
        prediction_hours: int = 24,
    ) -> None:
        self.lookback_hours = lookback_hours
        self.prediction_hours = prediction_hours

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Generate one forecasting sample per valid ICU stay."""
        samples: List[Dict[str, Any]] = []
        required = timedelta(hours=self.lookback_hours + self.prediction_hours)

        for stay in patient.get_events(event_type="icustays"):
            icu_in = stay.timestamp
            icu_out = self._parse_outtime(getattr(stay, "outtime", None))
            if icu_in is None or icu_out is None or (icu_out - icu_in) < required:
                continue

            obs_end = icu_in + timedelta(hours=self.lookback_hours)
            pred_end = obs_end + timedelta(hours=self.prediction_hours)
            obs_df = patient.get_events(
                event_type="labevents", start=icu_in, end=obs_end, return_df=True
            )
            pred_df = patient.get_events(
                event_type="labevents", start=obs_end, end=pred_end, return_df=True
            )

            current_sofa = self._compute_lab_sofa(obs_df)
            future_sofa = self._compute_lab_sofa(pred_df)
            if current_sofa is None or future_sofa is None:
                continue

            obs_values, obs_masks = self._bin_hourly(obs_df, icu_in, self.lookback_hours)
            tgt_values, tgt_masks = self._bin_hourly(
                pred_df, obs_end, self.prediction_hours
            )

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id": str(getattr(stay, "icustay_id", "")),
                "observation_values": obs_values,
                "observation_masks": obs_masks,
                "target_values": tgt_values,
                "target_masks": tgt_masks,
                "sofa_label": 1 if (future_sofa - current_sofa) >= 2 else 0,
            })

        return samples

    @staticmethod
    def _parse_outtime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if value is None:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def _bin_hourly(
        self,
        labs_df: pl.DataFrame,
        window_start: datetime,
        num_hours: int,
    ) -> Tuple[List[float], List[float]]:
        """Bin all lab events into fixed hourly value and mask tensors.

        Follows Section 3.1 of Staniek et al. (2024)
        Keep the first observed
        value per (hour, lab) slot and zero-impute misses.
        """
        size = num_hours * self.NUM_LABS
        values = [0.0] * size
        masks = [0.0] * size
        if labs_df.height == 0:
            return values, masks

        filtered = (
            labs_df.filter(
                pl.col("labevents/itemid").cast(pl.Utf8).is_in(self.LAB_ITEMIDS)
            )
            .filter(pl.col("labevents/valuenum").is_not_null())
            .sort("timestamp")
        )

        filled = set()
        for row in filtered.iter_rows(named=True):
            hour = int((row["timestamp"] - window_start).total_seconds() // 3600)
            if not 0 <= hour < num_hours:
                continue
            itemid = str(row["labevents/itemid"])
            lab_idx = self.LAB_ITEMIDS.index(itemid)
            key = (hour, lab_idx)
            if key in filled:
                continue
            filled.add(key)
            flat = hour * self.NUM_LABS + lab_idx
            mean, std = self.LAB_NORMALIZATION_STATS[itemid]
            values[flat] = (float(row["labevents/valuenum"]) - mean) / std if std > 0 else 0.0
            masks[flat] = 1.0

        return values, masks

    def _get_lab_values(self, labs_df: pl.DataFrame, itemid: str) -> List[float]:
        """Return all numeric values for a given lab item."""
        if labs_df.height == 0:
            return []
        filtered = labs_df.filter(
            pl.col("labevents/itemid").cast(pl.Utf8) == itemid
        ).filter(pl.col("labevents/valuenum").is_not_null())
        if filtered.height == 0:
            return []
        return filtered["labevents/valuenum"].cast(pl.Float64).to_list()

    def _sofa_bilirubin(self, value: float) -> int:
        """Paper Appendix B, Table 6: bilirubin -> subscore."""
        for threshold, score in ((12.0, 4), (6.0, 3), (2.0, 2), (1.2, 1)):
            if value >= threshold:
                return score
        return 0

    def _sofa_creatinine(self, value: float) -> int:
        """Paper Appendix B, Table 6: creatinine -> subscore."""
        for threshold, score in ((5.0, 4), (3.5, 3), (2.0, 2), (1.2, 1)):
            if value >= threshold:
                return score
        return 0

    def _sofa_platelets(self, value: float) -> int:
        """Paper Appendix B, Table 6: platelets -> subscore (lower is worse)."""
        for threshold, score in ((20.0, 4), (50.0, 3), (100.0, 2), (150.0, 1)):
            if value < threshold:
                return score
        return 0

    def _compute_lab_sofa(self, labs_df: pl.DataFrame) -> Optional[int]:
        """Lab-only SOFA proxy from the worst values in the window."""
        bili = self._get_lab_values(labs_df, self.BILIRUBIN_ITEMID)
        creat = self._get_lab_values(labs_df, self.CREATININE_ITEMID)
        plt = self._get_lab_values(labs_df, self.PLATELETS_ITEMID)
        if not (bili or creat or plt):
            return None

        score = 0
        if bili:
            score += self._sofa_bilirubin(max(bili))
        if creat:
            score += self._sofa_creatinine(max(creat))
        if plt:
            score += self._sofa_platelets(min(plt))
        return score
