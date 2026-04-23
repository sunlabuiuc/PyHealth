from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("tpc_static")
class TPCStaticProcessor(FeatureProcessor):
    """
    Feature processor for TPC static inputs

    Encodes the 12 static features for one ICU stay into a fixed-size 1D float tensor.
    Categorical features are one-hot encoded over a vocabulary built during fit().
    Numeric features are scaled to [-1, 1] using per-feature 5th/95th percentiles
    computed during fit(), then clipped to [clip_min, clip_max].

    Categorical features (one-hot encoded):
        gender, race (paper: ethnicity), admission_location, insurance, first_careunit

    Numeric features (robust scaled):
        hour_of_admission, admission_height, admission_weight,
        gcs_eye, gcs_motor, gcs_verbal, anchor_age

    Input format (dict) produced by RemainingLengthOfStayTPC_MIMIC4 task:
        {
            "gender":             str,    # "M" or "F"
            "race":               str,    # categorical — paper calls this ethnicity
            "admission_location": str,    # categorical
            "insurance":          str,    # categorical
            "first_careunit":     str,    # categorical
            "hour_of_admission":  int,    # 0-23
            "admission_height":   float,  # cm, or None if not recorded
            "admission_weight":   float,  # kg, or None if not recorded
            "gcs_eye":            float,  # 1-4, or None if not recorded
            "gcs_motor":          float,  # 1-6, or None if not recorded
            "gcs_verbal":         float,  # 1-5, or None if not recorded
            "anchor_age":         int,    # age at ICU admission
        }

    Args:
        clip_min: Lower clip bound after scaling. Default: -4.0.
        clip_max: Upper clip bound after scaling. Default:  4.0.

    Returns:
        ``torch.FloatTensor`` of shape ``(S,)`` where ``S`` is the sum of one-hot
        vocabulary sizes plus seven numeric features.
        Missing categorical values map to the <UNK> one-hot position.
        Missing numeric values map to 0.0 (the scaled midpoint).

    Examples:
        >>> processor = TPCStaticProcessor()
        >>> samples = [{"static": {"gender": "M", "race": "WHITE", ...}}]
        >>> processor.fit(samples, "static")
        >>> out = processor.process({"gender": "M", "race": "WHITE", ...})
        >>> out.shape    # (S,) where S depends on vocab sizes seen during fit
        >>> out.dtype    # torch.float32
    """

    CATEGORICAL_KEYS: Tuple[str, ...] = (
        "gender",
        "race",
        "admission_location",
        "insurance",
        "first_careunit",
    )
    NUMERIC_KEYS: Tuple[str, ...] = (
        "hour_of_admission",
        "admission_height",
        "admission_weight",
        "gcs_eye",
        "gcs_motor",
        "gcs_verbal",
        "anchor_age",
    )

    def __init__(self, clip_min: float = -4.0, clip_max: float = 4.0) -> None:
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self._cat_vocab: Dict[str, List[str]] = {k: [] for k in self.CATEGORICAL_KEYS}
        self._cat_index: Dict[str, Dict[str, int]] = {
            k: {} for k in self.CATEGORICAL_KEYS
        }

        self._p5: Dict[str, float] = {}
        self._p95: Dict[str, float] = {}

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Build vocabularies and numeric scaling bounds from ``samples``.

        Collects categorical tokens and numeric values for each static field, then
        stores sorted vocabularies (with ``<UNK>``) and 5th/95th percentiles per
        numeric feature.
        """
        cat_values: Dict[str, set[str]] = {k: set() for k in self.CATEGORICAL_KEYS}
        num_values: Dict[str, List[float]] = {k: [] for k in self.NUMERIC_KEYS}

        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            s = sample[field]
            if not isinstance(s, dict):
                continue

            for k in self.CATEGORICAL_KEYS:
                v = s.get(k, None)
                if v is None:
                    continue
                cat_values[k].add(str(v))

            for k in self.NUMERIC_KEYS:
                v = s.get(k, None)
                if v is None:
                    continue
                try:
                    num_values[k].append(float(v))
                except Exception:
                    continue

        for k in self.CATEGORICAL_KEYS:
            vocab = sorted(cat_values[k])
            # reserve index 0 for unknown/missing
            self._cat_vocab[k] = ["<UNK>"] + vocab
            self._cat_index[k] = {tok: i for i, tok in enumerate(self._cat_vocab[k])}

        for k in self.NUMERIC_KEYS:
            arr = np.asarray(num_values[k], dtype=float)
            if arr.size == 0:
                self._p5[k] = 0.0
                self._p95[k] = 1.0
            else:
                self._p5[k] = float(np.nanpercentile(arr, 5))
                self._p95[k] = float(np.nanpercentile(arr, 95))

    def _scale(self, key: str, x: float) -> float:
        """Linearly scale ``x`` into ``[-1, 1]`` using stored percentiles.

        Values are clipped to ``[clip_min, clip_max]`` after scaling. If the
        percentile range is degenerate, returns ``0.0``.
        """

        p5 = self._p5.get(key, 0.0)
        p95 = self._p95.get(key, 1.0)
        if p95 == p5:
            return 0.0
        scaled = 2.0 * (x - p5) / (p95 - p5) - 1.0
        return float(np.clip(scaled, self.clip_min, self.clip_max))

    def process(self, value: Dict[str, Any]) -> torch.Tensor:
        """Encode ``value`` into a 1D float tensor.

        Categorical columns are one-hot encoded; numeric columns are robust-scaled.
        """
        parts: List[float] = []

        # Categorical one-hots.
        for k in self.CATEGORICAL_KEYS:
            vocab = self._cat_vocab.get(k, ["<UNK>"])
            idx_map = self._cat_index.get(k, {"<UNK>": 0})
            one_hot = np.zeros(len(vocab), dtype=float)
            raw = value.get(k, None)
            tok = "<UNK>" if raw is None else str(raw)
            one_hot[idx_map.get(tok, 0)] = 1.0
            parts.extend(one_hot.tolist())

        # Numeric robust scaling.
        for k in self.NUMERIC_KEYS:
            raw = value.get(k, None)
            if raw is None:
                parts.append(0.0)
                continue
            try:
                parts.append(self._scale(k, float(raw)))
            except Exception:
                parts.append(0.0)

        return torch.tensor(parts, dtype=torch.float32)

    def size(self) -> int:
        """Return total static dimension (one-hot widths plus numeric count)."""
        cat_size = sum(
            len(self._cat_vocab.get(k, ["<UNK>"])) for k in self.CATEGORICAL_KEYS
        )
        return cat_size + len(self.NUMERIC_KEYS)

    def is_token(self) -> bool:
        """Static features are continuous, not discrete tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Output is a tuple of (value) tensor."""
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 1D tensor."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        """Static features are not spatial."""
        return (False,)

