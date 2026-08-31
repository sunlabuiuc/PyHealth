"""Train-split-only standardisation for masked temporal laboratory values.

The task records keep labs and their observation mask as separate temporal
fields.  This module deliberately fits only rows whose corresponding mask is
true: zero-filled / forward-filled missing values must never affect a lab's
mean or variance.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from typing import Any, Optional

import torch
from torch import nn


def _provenance_indices(dataset: Any) -> Optional[list[int]]:
    """Every sample index of this split, or ``None`` for a plain iterable.

    ``SampleDataset`` subclasses ``litdata.StreamingDataset``, whose ``__iter__``
    and ``__len__`` are both sharded by ``WORLD_SIZE``.  Under ``torchrun`` that
    silently reduces a fit to 1/WORLD_SIZE of the train split, and to the *same*
    shard on every rank, because ``torch.distributed`` is not yet initialised
    when the dataset is built.  Indexing is not sharded, so the fit is driven by
    explicit indices.

    The count comes from ``region_of_interest``, which is the only unsharded
    description of what this dataset holds.  Measured against real litdata with
    20 samples: ``len()`` reports 5 under ``WORLD_SIZE=4`` while the region of
    interest still sums to 20.

    ``patient_to_index`` is NOT usable here.  ``SampleDataset.subset`` copies it
    unchanged, so after ``split_by_patient`` it still holds indices into the
    PARENT dataset while ``__getitem__`` is restricted to the subset's own
    region.  Driving the fit from it made a real training split raise
    ``ValueError: The provided index 237 didn't find a match within the chunk
    intervals``.
    """
    roi = getattr(dataset, "region_of_interest", None)
    if not roi:
        return None
    return list(range(sum(end - start for start, end in roi)))


def _is_shardable_dataset(obj: Any) -> bool:
    """Whether iterating ``obj`` risks silently yielding only one shard.

    ``litdata.StreamingDataset`` shards ``__iter__`` by ``WORLD_SIZE``. Anything
    that subclasses it must be driven by explicit indices instead.
    """
    try:
        from litdata.streaming.dataset import StreamingDataset
    except Exception:  # litdata absent: nothing can be sharded
        return False
    return isinstance(obj, StreamingDataset)


class LabStandardizer(nn.Module):
    """Per-feature z-score transform with persistent train-only statistics.

    ``mean``, ``std`` and ``observed_count`` are buffers rather than ordinary
    attributes.  Consequently they are included in every model ``state_dict``
    and a checkpoint always transforms raw serving inputs exactly as it did at
    training time.

    Constant train features use a unit denominator.  Features with no observed
    train values are emitted as zero, because a scale for them cannot be learnt
    without looking at validation/test data.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        observed_count: torch.Tensor,
        *,
        version: int = 2,
        fit_scope: str | bytes | None = None,
    ) -> None:
        super().__init__()
        if mean.ndim != 1 or std.shape != mean.shape or observed_count.shape != mean.shape:
            raise ValueError("Lab standardisation statistics must be one vector per feature.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Lab standardisation statistics must be finite.")
        if (std <= 0).any() or (observed_count < 0).any():
            raise ValueError("Lab standardisation std must be positive and counts non-negative.")
        self.register_buffer("mean", mean.detach().to(torch.float32).clone())
        self.register_buffer("std", std.detach().to(torch.float32).clone())
        self.register_buffer(
            "observed_count", observed_count.detach().to(torch.long).clone()
        )
        self.register_buffer("version", torch.tensor([version], dtype=torch.long))
        # This is a privacy-preserving SHA-256 digest of the exact train-split
        # identity.  It lets transfer loading reject a checkpoint whose fitted
        # transform came from a different cohort/split before that transform can
        # reach a downstream test patient.
        self.register_buffer(
            "fit_scope_digest", self._scope_digest(fit_scope), persistent=True
        )

    @property
    def feature_dim(self) -> int:
        return int(self.mean.numel())

    @classmethod
    def fit(
        cls,
        samples: Iterable[dict[str, Any]],
        *,
        value_field: str = "labs",
        observation_mask_field: Optional[str] = None,
        fit_scope: str | bytes | None = None,
    ) -> "LabStandardizer":
        """Fit only observed, finite values from the supplied samples.

        The caller supplies the already split training dataset.  No reference
        to a parent/full dataset is used here, which makes train-only fitting
        auditable at the call site.
        """
        observation_mask_field = observation_mask_field or f"{value_field}_mask"
        indices = _provenance_indices(samples)
        if indices is None and _is_shardable_dataset(samples):
            # A plain list of dicts is safe to iterate; a StreamingDataset is not,
            # because __iter__ is sharded by WORLD_SIZE. Falling back to iteration
            # for one while silently doing it for the other is how a fit ends up
            # on 1/N of the training data with nothing reporting it.
            raise ValueError(
                "Refusing to fit lab standardisation by iterating a "
                f"{type(samples).__name__}: its __iter__ is sharded by WORLD_SIZE, "
                "so under torchrun this would silently fit on a fraction of the "
                "training split. The dataset exposes no region_of_interest "
                "to drive an unsharded fit."
            )
        stream = samples if indices is None else (samples[index] for index in indices)
        consumed = 0
        count: Optional[torch.Tensor] = None
        total: Optional[torch.Tensor] = None
        total_sq: Optional[torch.Tensor] = None

        for sample in stream:
            consumed += 1
            if value_field not in sample or observation_mask_field not in sample:
                continue
            values = cls._value_tensor(sample[value_field]).to(torch.float64)
            observed = cls._value_tensor(sample[observation_mask_field]).bool()
            if values.shape != observed.shape:
                raise ValueError(
                    f"{value_field!r} and {observation_mask_field!r} must have the "
                    f"same shape, got {tuple(values.shape)} and {tuple(observed.shape)}."
                )
            if values.ndim == 1:
                values = values.unsqueeze(-1)
                observed = observed.unsqueeze(-1)
            if values.ndim != 2:
                raise ValueError(
                    f"Expected temporal values shaped (time, features), got {tuple(values.shape)}."
                )
            valid = observed & torch.isfinite(values)
            if count is None:
                feature_dim = values.shape[-1]
                count = torch.zeros(feature_dim, dtype=torch.long)
                total = torch.zeros(feature_dim, dtype=torch.float64)
                total_sq = torch.zeros(feature_dim, dtype=torch.float64)
            elif values.shape[-1] != count.numel():
                raise ValueError("All fitted samples must have the same laboratory feature dimension.")

            assert total is not None and total_sq is not None
            count += valid.sum(dim=0).to(torch.long)
            total += torch.where(valid, values, torch.zeros_like(values)).sum(dim=0)
            total_sq += torch.where(valid, values.square(), torch.zeros_like(values)).sum(dim=0)

        if indices is not None and consumed != len(indices):
            raise RuntimeError(
                f"Lab standardisation consumed {consumed} of the {len(indices)} "
                "samples its split declares; the statistics would be fitted on a "
                "fraction of the training data."
            )

        if count is None or total is None or total_sq is None:
            raise ValueError(
                f"No samples with both {value_field!r} and {observation_mask_field!r} were available."
            )

        has_observed = count > 0
        denominator = count.clamp(min=1).to(torch.float64)
        mean = torch.where(has_observed, total / denominator, torch.zeros_like(total))
        variance = torch.where(
            has_observed,
            (total_sq / denominator - mean.square()).clamp_min(0),
            torch.ones_like(total),
        )
        # A constant train feature is well-defined: it maps to zero.  Unit std
        # avoids division by zero and gives a finite, explicit OOD behaviour.
        std = torch.where(variance > 0, variance.sqrt(), torch.ones_like(variance))
        return cls(
            mean.to(torch.float32), std.to(torch.float32), count,
            fit_scope=fit_scope,
        )

    @staticmethod
    def _scope_digest(fit_scope: str | bytes | None) -> torch.Tensor:
        """Return a stable, non-reversible identifier for the fitting split."""
        if fit_scope is None:
            return torch.zeros(32, dtype=torch.uint8)
        payload = fit_scope.encode("utf-8") if isinstance(fit_scope, str) else fit_scope
        return torch.tensor(list(hashlib.sha256(payload).digest()), dtype=torch.uint8)

    @staticmethod
    def _value_tensor(value: Any) -> torch.Tensor:
        """Extract the ``value`` component from processor output or raw tuples."""
        if isinstance(value, dict):
            value = value["value"]
        elif isinstance(value, tuple):
            # StageNet temporal processors return ``(time, value)``.
            value = value[1]
        return torch.as_tensor(value)

    def forward(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> torch.Tensor:
        """Standardise observed values and map missing/unfittable values to zero.

        We intentionally do not clip.  The pipeline has no universally valid
        physiological range for these MIMIC category aggregates; clipping would
        silently overwrite potentially meaningful values.  Values far outside
        train support are therefore represented by a large (but finite) z-score
        and remain auditable.
        """
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} lab features, got {values.shape[-1]}."
            )
        if observed_mask.shape != values.shape:
            raise ValueError(
                "Laboratory observation mask must have exactly the values shape; "
                f"got {tuple(observed_mask.shape)} for {tuple(values.shape)}."
            )
        values = values.to(dtype=self.mean.dtype)
        observed = observed_mask.bool() & torch.isfinite(values)
        fitted = self.observed_count > 0
        z = (values - self.mean) / self.std
        return torch.where(observed & fitted, z, torch.zeros_like(z))


def fit_lab_standardizer(
    train_dataset: Iterable[dict[str, Any]],
    *,
    value_field: str = "labs",
    observation_mask_field: Optional[str] = None,
    fit_scope: str | bytes | None = None,
) -> LabStandardizer:
    """Explicit train-dataset entry point used by downstream experiment scripts."""
    return LabStandardizer.fit(
        train_dataset,
        value_field=value_field,
        observation_mask_field=observation_mask_field,
        fit_scope=fit_scope,
    )


def lab_standardizer_fit_scope(dataset: Any, *, value_field: str = "labs") -> str:
    """Fingerprint the patient/record membership of a split for safe transfer.

    ``SampleDataset.subset`` preserves these maps, so this checks the actual
    split membership without iterating protected test data or serialising any
    patient identifier into a checkpoint.  A dataset without this provenance is
    refused by the experiment scripts rather than being treated as safe by
    default.
    """
    patients = getattr(dataset, "patient_to_index", None) or {}
    records = getattr(dataset, "record_to_index", None) or {}
    indices = _provenance_indices(dataset)
    if indices is None:
        raise ValueError(
            "Cannot bind lab standardisation to a train split: dataset has no "
            "region_of_interest to drive an unsharded fit."
        )
    payload = {
        "value_field": value_field,
        # ``len(dataset)`` is sharded by WORLD_SIZE on a StreamingDataset, so a
        # digest taken under torchrun could never match the single-process one.
        "n_samples": len(indices),
        "patients": sorted(str(patient_id) for patient_id in patients),
        "records": sorted(str(record_id) for record_id in records),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
