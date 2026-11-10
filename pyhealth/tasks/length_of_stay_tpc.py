"""Author: Tamjit Kumar, Yi Wang
NetID: tkumar70, ywang4715
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Abstract Base Task for Temporal Pointwise Convolution (TPC) Length of Stay
             Prediction and Multi-Modal Feature Extraction across ICU Databases.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Minimum floor for remaining LoS (1/48 days = 30 minutes) to prevent log singularities
LOS_FLOOR_DAYS = 1.0 / 48.0


@dataclass
class RawStayRecord:
    """Standardized intermediate stay container emitted by dataset adapters.

    Attributes:
        stay_id: Unique stay identifier string (e.g. patientunitstayid, stay_id).
        patient_id: Unique patient identifier string.
        stay_hours: Total length of ICU stay in integer hours (T).
        discharge_offset_minutes: Total stay duration in minutes.
        admission_hour: Hour of admission in [0, 23] (local clock time).
        hourly_vitals: List of (hour, feature_name, value) measurement tuples.
        static_continuous: Mapping of feature name to numeric baseline observation.
        static_categorical: Mapping of feature name to categorical string value.
        is_expired: True if patient died in-hospital, False otherwise.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import RawStayRecord
        >>> rec = RawStayRecord(
        ...     stay_id="s1",
        ...     patient_id="p1",
        ...     stay_hours=24,
        ...     discharge_offset_minutes=1440.0,
        ...     admission_hour=14.0,
        ...     hourly_vitals=[(1, "heart_rate", 80.0)],
        ...     static_continuous={"age": 60.0},
        ...     static_categorical={"unittype": "MICU"},
        ...     is_expired=False,
        ... )
        >>> rec.stay_id
        's1'
    """

    stay_id: str
    patient_id: str
    stay_hours: int
    discharge_offset_minutes: float
    admission_hour: float
    hourly_vitals: list[tuple[int, str, float]]
    static_continuous: dict[str, float | None]
    static_categorical: dict[str, str]
    is_expired: bool


class StayDataMap(Mapping[str, Any]):
    """Lightweight mapping wrapper that preserves compact string representation.

    Prevents BaseDataset.set_task from serializing large stay dictionaries
    into multi-megabyte JSON task parameter strings.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import StayDataMap
        >>> sm = StayDataMap({"s1": [1, 2, 3]})
        >>> len(sm)
        1
        >>> "s1" in sm
        True
        >>> sm.get("s1")
        [1, 2, 3]
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        if data is None:
            self._data: dict[str, Any] = {}
        elif isinstance(data, dict):
            self._data = data
        else:
            self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self) -> str:
        return f"<StayDataMap: {len(self._data)} stays>"


def compute_cache_fingerprint(data: Any) -> str:
    """Computes a deterministic lightweight SHA-256 fingerprint for caching.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import compute_cache_fingerprint
        >>> fp = compute_cache_fingerprint({"stay_1": [1, 2]})
        >>> len(fp)
        16
    """
    if data is None:
        return "none"
    if isinstance(data, (str, Path)):
        p = str(data)
        if os.path.exists(p):
            stat = os.stat(p)
            token = f"{p}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        return hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
    if not hasattr(data, "keys") or not data:
        return "empty"
    keys = sorted(data.keys())
    sample = [keys[0], keys[len(keys) // 2], keys[-1]] if len(keys) > 3 else keys
    sample_preview = [str(data[k])[:80] for k in sample]
    token = f"len={len(data)};sample_keys={sample};sample_val={sample_preview}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


_compute_cache_fingerprint = compute_cache_fingerprint


class FastReservoirSampler:
    """Memory-bounded uniform reservoir sampler (Algorithm R).

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import FastReservoirSampler
        >>> sampler = FastReservoirSampler(capacity=5)
        >>> for v in range(10):
        ...     sampler.add(float(v))
        >>> sampler.total_count
        10
        >>> len(sampler.samples)
        5
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.total_count = 0
        self.samples: list[float] = []

    def add(self, val: float) -> None:
        if not np.isfinite(val):
            return
        self.total_count += 1
        if self.capacity <= 0 or len(self.samples) < self.capacity:
            self.samples.append(val)
        elif self.rng.random() < (self.capacity / self.total_count):
            idx = self.rng.randrange(self.capacity)
            self.samples[idx] = val


def generate_tpc_tensors(
    raw_grid: torch.Tensor,
    mask: torch.Tensor,
    decay_rate: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates forward-filled time series and exponential decay indicator tensors.

    Conforms to Rocheteau et al. (ACM CHIL 2021) Section 4.1:
    - Forward-fills across missing hourly intervals.
    - Computes decay indicators: x''_t = (decay_rate)^j, where j is hours
      since last observation.
    - If never observed, value is 0.0 (population mean) and decay indicator is 0.0.

    Examples:
        >>> import torch
        >>> from pyhealth.tasks.length_of_stay_tpc import generate_tpc_tensors
        >>> grid = torch.ones((24, 6))
        >>> mask = torch.ones((24, 6))
        >>> ff, decay = generate_tpc_tensors(grid, mask)
        >>> ff.shape
        torch.Size([24, 6])
        >>> decay.shape
        torch.Size([24, 6])
    """
    seq_len, num_features = raw_grid.shape
    if seq_len == 0 or num_features == 0:
        return raw_grid.clone(), torch.zeros_like(raw_grid)

    device = raw_grid.device
    idx_grid = torch.where(
        mask.bool(),
        torch.arange(1, seq_len + 1, device=device).unsqueeze(1),
        torch.zeros(1, dtype=torch.long, device=device),
    )
    last_obs_1based, _ = torch.cummax(idx_grid, dim=0)
    has_obs = last_obs_1based > 0
    last_obs_idx = torch.clamp(last_obs_1based - 1, min=0)

    vitals_ff = torch.gather(raw_grid, dim=0, index=last_obs_idx)
    vitals_ff = torch.where(has_obs, vitals_ff, torch.zeros_like(raw_grid))

    steps = torch.arange(seq_len, device=device).unsqueeze(1)
    time_since = (steps - (last_obs_1based - 1)).float()
    vitals_decay = torch.where(
        has_obs, torch.pow(decay_rate, time_since), torch.zeros_like(raw_grid)
    )
    return vitals_ff, vitals_decay


class BaseLengthOfStayTPC(BaseTask):
    """Abstract Base Task for ICU Remaining Length of Stay (LoS) Prediction.

    Implements the shared mathematical representation, forward-filling, exponential
    decay calculation, temporal coordinates, and target formulation from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit' (Rocheteau et al., ACM CHIL 2021).

    Subclasses implement ``extract_patient_events(patient)`` to adapt specific clinical
    database schemas (e.g., eICU, MIMIC-IV, OMOP, Epic Clarity).

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import BaseLengthOfStayTPC
        >>> task = BaseLengthOfStayTPC(time_series_features=["hr", "bp"])
        >>> task.continuous_dim
        2
        >>> task.time_dim
        2
    """

    task_name: str = "ICU Remaining Length of Stay Prediction (TPC)"
    default_priors: dict[str, tuple[float, ...]] = {}  # noqa: RUF012
    input_schema: dict[str, str | type] = {  # noqa: RUF012
        "vitals": "tensor",
        "vitals_decay": "tensor",
        "static": "tensor",
        "time_features": "tensor",
        "mask": "tensor",
    }
    output_schema: dict[str, str | type] = {  # noqa: RUF012
        "los": "tensor",
        "mortality": "tensor",
    }

    def __init__(
        self,
        time_series_features: list[str] | None = None,
        norm_stats: Mapping[str, tuple[float, ...]] | None = None,
        default_priors: Mapping[str, tuple[float, ...]] | None = None,
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
        """Initializes the base length of stay TPC task.

        Args:
            time_series_features: List of continuous time-series feature names.
            norm_stats: Empirical normalization statistics (mean, std, p1, p99).
            default_priors: Fallback clinical reference priors for normalization.
            static_stats: Continuous static statistics mapping key to
                (mean, std, median).
            cat_vocabs: Categorical vocabulary mapping feature name to valid
                category list.
            max_stay_hours: Maximum length of stay in hours for sequence truncation.
            min_stay_hours: Minimum length of stay in hours for cohort inclusion.
            min_age: Minimum patient age in years for adult cohort inclusion.
            floor_days: Minimum remaining LoS threshold in days
                (default: 1/48, ~30 mins).
            decay_rate: Exponential decay base rate (0.75^j). Default: 0.75.
            include_mortality: Whether to include in-hospital mortality target.
                Default: True.
            include_time_features: Whether to include non-convolved temporal context
                features. Default: True.
            feat_counts: Optional mapping of feature names to observation counts.
        """
        super().__init__()
        self.time_series_features = (
            time_series_features if time_series_features is not None else []
        )
        self.norm_stats = dict(norm_stats) if norm_stats is not None else {}
        self.default_priors = (
            dict(default_priors)
            if default_priors is not None
            else dict(self.default_priors)
        )
        self.feat_counts = dict(feat_counts) if feat_counts is not None else {}
        self.max_stay_hours = max_stay_hours
        self.min_stay_hours = min_stay_hours
        self.min_stay_offset = float(min_stay_hours * 60.0)
        self.min_age = min_age
        self.floor_days = floor_days
        self.decay_rate = decay_rate
        self.include_mortality = include_mortality
        self.include_time_features = include_time_features

        self.input_schema = {
            "vitals": "tensor",
            "vitals_decay": "tensor",
            "static": "tensor",
            "mask": "tensor",
        }
        if include_time_features:
            self.input_schema["time_features"] = "tensor"
        self.output_schema = {"los": "tensor"}
        if include_mortality:
            self.output_schema["mortality"] = "tensor"

        self.feature_map = {feat: i for i, feat in enumerate(self.time_series_features)}
        self.continuous_dim = len(self.time_series_features)
        self.time_dim = 2 if include_time_features else 0

        self.static_stats = static_stats or {
            "age": (63.0, 15.0, 63.0),
            "height": (170.0, 10.0, 170.0),
            "weight": (80.0, 20.0, 80.0),
            "eyes": (3.5, 0.8, 4.0),
            "motor": (5.5, 1.0, 6.0),
            "verbal": (4.5, 1.0, 5.0),
        }

        self.cat_vocabs = cat_vocabs or {
            "unittype": ["Med-Surg ICU", "CTICU", "MICU", "SICU", "CCU-CTICU"],
            "unitadmitsource": [
                "Emergency Department",
                "Operating Room",
                "Floor",
                "Direct Admit",
            ],
            "unitstaytype": ["admit", "readmit", "transfer"],
            "ethnicity": ["Caucasian", "African American", "Hispanic", "Asian"],
        }

        # 11 continuous/indicator features + one-hot categories (each with unknown slot)
        self.static_dim = 11 + sum(len(v) + 1 for v in self.cat_vocabs.values())

    @property
    def static_feature_names(self) -> list[str]:
        names = [
            "age",
            "gender",
            "admission_hour",
            "height",
            "missing_height",
            "weight",
            "missing_weight",
            "gcs_eyes",
            "gcs_motor",
            "gcs_verbal",
            "missing_gcs",
        ]
        for cat_name, vocab in self.cat_vocabs.items():
            for val in vocab:
                names.append(f"{cat_name}_{val}")
            names.append(f"{cat_name}_other")
        return names

    @staticmethod
    def _one_hot(val: str, vocab: list[str]) -> list[float]:
        vec = [0.0] * (len(vocab) + 1)
        if val in vocab:
            vec[vocab.index(val)] = 1.0
        else:
            vec[-1] = 1.0
        return vec

    def extract_patient_events(self, patient: Any) -> list[RawStayRecord]:
        """Abstract method implemented by database adapters to parse raw events."""
        raise NotImplementedError("Subclasses must implement extract_patient_events")

    def process_stay_record(self, stay: RawStayRecord) -> dict[str, Any] | None:
        """Transforms a standardized RawStayRecord into model-ready tensors."""
        T = min(stay.stay_hours, self.max_stay_hours)
        if T <= 0:
            return None

        raw_grid = torch.zeros((T, self.continuous_dim), dtype=torch.float32)
        mask = torch.zeros((T, self.continuous_dim), dtype=torch.float32)

        # 1. Populate dynamic hourly observations
        for hour, vkey, val in stay.hourly_vitals:
            if vkey not in self.feature_map or not (0 <= hour < T):
                continue
            f_idx = self.feature_map[vkey]
            stat = self.norm_stats.get(
                vkey,
                self.default_priors.get(vkey, (0.0, 1.0, float("-inf"), float("inf"))),
            )
            mean = stat[0]
            std = stat[1] if stat[1] > 1e-4 else 1.0
            if len(stat) >= 4:
                val = max(min(val, stat[3]), stat[2])
            raw_grid[hour, f_idx] = (val - mean) / std
            mask[hour, f_idx] = 1.0

        if mask.sum() == 0:
            return None

        # 2. Forward filling & exponential decay indicators: 0.75^j
        vitals_ff, vitals_decay = generate_tpc_tensors(
            raw_grid, mask, decay_rate=self.decay_rate
        )

        # 3. Remaining LoS Targets anchored at end of hour t
        los_targets = torch.tensor(
            [
                max(
                    (stay.discharge_offset_minutes - (t + 1) * 60.0) / 1440.0,
                    self.floor_days,
                )
                for t in range(T)
            ],
            dtype=torch.float32,
        ).unsqueeze(1)

        # 4. Standardized Static Features
        age_m, age_s, _ = self.static_stats["age"]
        raw_age = stay.static_continuous.get("age", age_m)
        age_norm = ((raw_age if raw_age is not None else age_m) - age_m) / age_s

        adm_hour_norm = (stay.admission_hour - 12.0) / 12.0

        h_m, h_s, h_med = self.static_stats["height"]
        raw_h = stay.static_continuous.get("height")
        if raw_h is not None and raw_h > 0:
            height_norm = (raw_h - h_m) / h_s
            missing_h = 0.0
        else:
            height_norm = (h_med - h_m) / h_s
            missing_h = 1.0

        w_m, w_s, w_med = self.static_stats["weight"]
        raw_w = stay.static_continuous.get("weight")
        if raw_w is not None and raw_w > 0:
            weight_norm = (raw_w - w_m) / w_s
            missing_w = 0.0
        else:
            weight_norm = (w_med - w_m) / w_s
            missing_w = 1.0

        e_m, e_s, e_med = self.static_stats["eyes"]
        m_m, m_s, m_med = self.static_stats["motor"]
        v_m, v_s, v_med = self.static_stats["verbal"]
        raw_e = stay.static_continuous.get("eyes")
        raw_m = stay.static_continuous.get("motor")
        raw_v = stay.static_continuous.get("verbal")
        if raw_e is not None and raw_m is not None and raw_v is not None:
            eyes_norm = (raw_e - e_m) / e_s
            motor_norm = (raw_m - m_m) / m_s
            verbal_norm = (raw_v - v_m) / v_s
            missing_gcs = 0.0
        else:
            eyes_norm = (e_med - e_m) / e_s
            motor_norm = (m_med - m_m) / m_s
            verbal_norm = (v_med - v_m) / v_s
            missing_gcs = 1.0

        gender_val = stay.static_continuous.get("gender", 0.0) or 0.0

        static_list = [
            age_norm,
            gender_val,
            adm_hour_norm,
            height_norm,
            missing_h,
            weight_norm,
            missing_w,
            eyes_norm,
            motor_norm,
            verbal_norm,
            missing_gcs,
        ]
        for cat_key, vocab in self.cat_vocabs.items():
            cat_val = stay.static_categorical.get(cat_key, "")
            static_list.extend(self._one_hot(cat_val, vocab))

        # 5. Non-convolved Temporal Context: (t+1)/24 and cyclic hour-of-day
        time_features = torch.tensor(
            [
                [(t + 1.0) / 24.0, ((stay.admission_hour + t) % 24.0) / 23.0]
                for t in range(T)
            ],
            dtype=torch.float32,
        )

        sample = {
            "visit_id": stay.stay_id,
            "patient_id": stay.patient_id,
            "vitals": vitals_ff,
            "vitals_decay": vitals_decay,
            "static": torch.tensor(static_list, dtype=torch.float32),
            "mask": torch.ones(T, dtype=torch.bool),
            "los": los_targets,
        }
        if self.include_time_features:
            sample["time_features"] = time_features
        if self.include_mortality:
            sample["mortality"] = torch.tensor(
                [1.0 if stay.is_expired else 0.0], dtype=torch.float32
            )
        return sample

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Template Method: Coordinates extraction and tensor construction."""
        raw_stays = self.extract_patient_events(patient)
        if not raw_stays:
            return []
        samples = []
        for stay in raw_stays:
            s = self.process_stay_record(stay)
            if s is not None:
                samples.append(s)
        return samples


class MapDataset(Dataset):
    """Map-style dataset enabling custom BatchSampler and length bucketing.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import MapDataset
        >>> ds = MapDataset([{"x": 1}, {"x": 2}])
        >>> len(ds)
        2
        >>> ds[0]
        {'x': 1}
    """

    def __init__(self, samples: Any, indices: Sequence[int] | None = None):
        self.samples = samples
        self.indices = list(indices) if indices is not None else None

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        if self.indices is not None:
            return self.samples[self.indices[index]]
        return self.samples[index]


class BucketBatchSampler(Sampler[list[int]]):
    """Batches sequences of similar length together to minimize zero padding.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import BucketBatchSampler
        >>> sampler = BucketBatchSampler(lengths=[10, 20, 12, 22], batch_size=2)
        >>> len(sampler)
        2
        >>> list(sampler)  # doctest: +SKIP
        [[0, 2], [1, 3]]
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def reset(self) -> None:
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = [int(idx) for idx in np.argsort(self.lengths)]
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = [int(p) for p in rng.permutation(len(batches))]
            batches = [batches[i] for i in perm]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        if not self.lengths:
            return 0
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def compute_los_metrics(
    y_true: Any,
    y_pred: Any,
    mask: Any | None = None,
    metrics: list[str] | None = None,
    floor_days: float = LOS_FLOOR_DAYS,
    mape_floor_days: float = 4.0 / 24.0,
    min_eval_hours: int = 4,
    bin_edges: Sequence[float] | None = None,
) -> dict[str, float]:
    """Computes specialized evaluation metrics for ICU length of stay prediction.

    Args:
        y_true: Ground truth target array, tensor, or list of batch sequences.
        y_pred: Predicted target array, tensor, or list of batch sequences.
        mask: Optional observation/evaluation mask corresponding to y_true/y_pred.
        metrics: List of metrics to compute. Defaults to
            ['msle', 'mse', 'mae', 'mape', 'r2', 'binned_kappa'].
        floor_days: Minimum threshold floor in days for MSLE clamping. Default: 1/48.
        mape_floor_days: Minimum denominator threshold in days for MAPE. Default: 4/24.
        min_eval_hours: Warmup hours to exclude from evaluation (default: 4).
        bin_edges: Optional custom cutoff thresholds in days for binned Cohen's Kappa.

    Returns:
        dict[str, float]: Dictionary of metric names mapped to evaluated scores.

    Note on MSLE Calculation:
        - 'msle': Flat MSLE pooled across all valid evaluation time-steps,
          matching Rocheteau et al. Tables 2-4 and official evaluation scripts.
        - 'stay_msle': Per-stay averaged MSLE, matching the Equation (6) training
          loss formulation.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc import compute_los_metrics
        >>> yt = [5.0, 4.0, 3.0]
        >>> yp = [4.8, 3.9, 3.1]
        >>> scores = compute_los_metrics(yt, yp)
        >>> "msle" in scores
        True
    """
    if (
        isinstance(y_true, (list, tuple))
        and len(y_true) > 0
        and hasattr(y_true[0], "shape")
    ):

        def _get_seq_len(item: Any) -> int:
            t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            if t.ndim == 1:
                return t.shape[0]
            if t.ndim == 2:
                return t.shape[0] if t.shape[1] == 1 else t.shape[1]
            return t.shape[1]

        max_T = max(_get_seq_len(item) for item in y_true)
        padded_yt = []
        padded_yp = []
        padded_m = []
        for idx, yt_item in enumerate(y_true):
            yp_item = y_pred[idx]
            m_item = mask[idx] if mask is not None else None
            yt_t = (
                yt_item
                if isinstance(yt_item, torch.Tensor)
                else torch.as_tensor(yt_item)
            )
            yp_t = (
                yp_item
                if isinstance(yp_item, torch.Tensor)
                else torch.as_tensor(yp_item)
            )
            if yt_t.ndim == 1:
                yt_t = yt_t.unsqueeze(0).unsqueeze(-1)
            elif yt_t.ndim == 2:
                if yt_t.shape[1] == 1:
                    yt_t = yt_t.unsqueeze(0)
                else:
                    yt_t = yt_t.unsqueeze(-1)
            if yp_t.ndim == 1:
                yp_t = yp_t.unsqueeze(0).unsqueeze(-1)
            elif yp_t.ndim == 2:
                if yp_t.shape[1] == 1:
                    yp_t = yp_t.unsqueeze(0)
                else:
                    yp_t = yp_t.unsqueeze(-1)

            if m_item is not None:
                m_t = (
                    m_item
                    if isinstance(m_item, torch.Tensor)
                    else torch.as_tensor(m_item)
                )
                if m_t.ndim == 1:
                    m_t = m_t.unsqueeze(0)
                elif m_t.ndim == 2 and m_t.shape[1] == 1:
                    m_t = m_t.transpose(0, 1)
                elif m_t.ndim == 3:
                    m_t = m_t.squeeze(-1)
            else:
                m_t = (yt_t.squeeze(-1) > 0).bool()

            cur_T = yt_t.shape[1]
            if cur_T < max_T:
                yt_t = torch.nn.functional.pad(yt_t, (0, 0, 0, max_T - cur_T))
                yp_t = torch.nn.functional.pad(yp_t, (0, 0, 0, max_T - cur_T))
                m_t = torch.nn.functional.pad(m_t.float(), (0, max_T - cur_T)).bool()

            padded_yt.append(yt_t)
            padded_yp.append(yp_t)
            padded_m.append(m_t)
        y_true = torch.cat(padded_yt, dim=0)
        y_pred = torch.cat(padded_yp, dim=0)
        mask = torch.cat(padded_m, dim=0)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if metrics is None:
        metrics = ["msle", "mse", "mae", "mape", "r2", "binned_kappa"]

    msle_val = 0.0
    stay_msle_val = 0.0
    if y_true.ndim >= 2:
        yt_2d = y_true.squeeze(-1) if y_true.ndim == 3 else y_true
        yp_2d = y_pred.squeeze(-1) if y_pred.ndim == 3 else y_pred
        if mask is not None:
            m_arr = np.asarray(mask, dtype=bool)
            if m_arr.ndim == 3:
                m_arr = m_arr.squeeze(-1)
        else:
            m_arr = yt_2d > 0

        T = yt_2d.shape[1]
        if min_eval_hours > 0 and T > min_eval_hours:
            eval_mask = m_arr.copy()
            eval_mask[:, :min_eval_hours] = False
        else:
            eval_mask = m_arr.copy()

        # Per-stay MSLE (Equation 6 in Rocheteau et al.)
        yp_clamped = np.maximum(yp_2d, floor_days)
        yt_clamped = np.maximum(yt_2d, floor_days)
        sq_log_err = ((np.log(yp_clamped) - np.log(yt_clamped)) ** 2) * eval_mask
        n_per_stay = eval_mask.sum(axis=1)
        valid_stays = n_per_stay > 0
        if np.any(valid_stays):
            stay_msles = sq_log_err[valid_stays].sum(axis=1) / n_per_stay[valid_stays]
            stay_msle_val = float(np.mean(stay_msles))

        yp_flat = yp_2d[eval_mask]
        yt_flat = yt_2d[eval_mask]
    else:
        yp_flat = y_pred.flatten()
        yt_flat = y_true.flatten()

    # Flat MSLE (matching Rocheteau et al. Tables 2-4 & metrics.py)
    yp_clamped_flat = np.maximum(yp_flat, floor_days)
    yt_clamped_flat = np.maximum(yt_flat, floor_days)
    msle_val = (
        float(np.mean((np.log(yt_clamped_flat / yp_clamped_flat)) ** 2))
        if len(yp_flat) > 0
        else 0.0
    )

    n_samples = len(yp_flat)
    output: dict[str, float] = {}
    if n_samples == 0:
        for metric in metrics:
            output[metric] = 0.0
        output["stay_msle"] = 0.0
        return output

    diff = yp_flat - yt_flat
    output["stay_msle"] = stay_msle_val
    for metric in metrics:
        if metric == "msle":
            output["msle"] = msle_val
        elif metric == "mse":
            output["mse"] = float(np.mean(diff**2))
        elif metric == "mae":
            output["mae"] = float(np.mean(np.abs(diff)))
        elif metric == "mape":
            denom = np.maximum(yt_flat, mape_floor_days)
            output["mape"] = float(np.mean(np.abs(diff) / denom * 100.0))
        elif metric == "r2":
            ss_tot = float(np.sum((yt_flat - np.mean(yt_flat)) ** 2))
            ss_res = float(np.sum(diff**2))
            output["r2"] = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        elif metric == "binned_kappa":
            edges = (
                np.asarray(bin_edges, dtype=np.float64)
                if bin_edges is not None
                else np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 14.0])
            )
            n_bins = len(edges) + 1
            yt_bins = np.digitize(yt_flat, edges, right=False)
            yp_bins = np.digitize(yp_flat, edges, right=False)
            conf_mat = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(conf_mat, (yt_bins, yp_bins), 1.0)
            total_cm = float(np.sum(conf_mat))
            if total_cm > 0:
                row_sums = conf_mat.sum(axis=1)
                col_sums = conf_mat.sum(axis=0)
                expected = np.outer(row_sums, col_sums) / total_cm
                idx = np.arange(n_bins, dtype=np.float64)
                w_mat = np.abs(idx[:, None] - idx[None, :]) / float(n_bins - 1)
                denom = float(np.sum(w_mat * expected))
                output["binned_kappa"] = (
                    1.0 - float(np.sum(w_mat * conf_mat)) / denom
                    if denom > 1e-12
                    else (1.0 if np.array_equal(yt_bins, yp_bins) else 0.0)
                )
            else:
                output["binned_kappa"] = 0.0
        else:
            raise ValueError(f"Unknown metric for length of stay task: {metric}")

    return output


length_of_stay_metrics_fn = compute_los_metrics

__all__ = [
    "LOS_FLOOR_DAYS",
    "BaseLengthOfStayTPC",
    "BucketBatchSampler",
    "FastReservoirSampler",
    "MapDataset",
    "RawStayRecord",
    "StayDataMap",
    "compute_cache_fingerprint",
    "compute_los_metrics",
    "generate_tpc_tensors",
    "length_of_stay_metrics_fn",
]
