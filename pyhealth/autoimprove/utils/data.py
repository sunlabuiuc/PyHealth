"""Facts about a SampleDataset that search and evaluation both need."""

from __future__ import annotations

import contextlib
import itertools
import logging
from collections.abc import Iterator

from pyhealth.datasets import SampleDataset
from pyhealth.processors import LabelProcessor


MAX_LABEL_SCAN = 50000
"""Most samples whose labels are read to count positives; reading a label runs
the sample's processors, so larger datasets skip the count."""


def _probe_indices(dataset: SampleDataset, n: int, scan: int = 1000) -> list[int]:
    """Up to ``n`` sample indices, taking each label value in turn."""
    key = next(iter(dataset.output_schema))
    by_label: dict[float, list[int]] = {}
    for i in range(min(len(dataset), scan)):
        try:
            label = float(dataset[i][key])
        except (TypeError, ValueError, RuntimeError):  # not a single number
            return list(range(min(n, len(dataset))))
        by_label.setdefault(label, []).append(i)
    rounds = itertools.zip_longest(*by_label.values())
    picked = [i for group in rounds for i in group if i is not None]
    return sorted(picked[:n])


@contextlib.contextmanager
def _quiet_trainer() -> Iterator[None]:
    """Hides the Trainer's per-epoch log lines while a trial trains."""
    trainer_logger = logging.getLogger("pyhealth.trainer")
    level = trainer_logger.level
    trainer_logger.setLevel(max(level, logging.WARNING))
    try:
        yield
    finally:
        trainer_logger.setLevel(level)


def describe_task(train_dataset: SampleDataset, val_dataset: SampleDataset) -> str:
    """A plain-text task summary for prompts: schema, sizes and label counts.

    Only aggregate numbers are included, never individual samples.

    Examples:
        >>> print(describe_task(train_ds, val_ds))  # doctest: +SKIP
    """
    lines = [
        f"Inputs: {_schema_names(train_dataset.input_schema)}",
        (
            f"Label: {_schema_names(train_dataset.output_schema)} "
            f"(type: {_label_mode(train_dataset)})"
        ),
        (
            f"Training samples: {len(train_dataset)}; validation samples: "
            f"{len(val_dataset)}"
        ),
    ]
    for key, proc in getattr(train_dataset, "input_processors", {}).items():
        try:
            lines.append(f"- {key}: {type(proc).__name__}, size {proc.size()}")
        except Exception:  # noqa: BLE001 - size() is optional on processors
            lines.append(f"- {key}: {type(proc).__name__}")
    if _label_mode(train_dataset) == "binary":
        positives = _positive_count(train_dataset)
        if positives is not None:
            lines.append(
                f"Training positives: {positives} of {len(train_dataset)} "
                f"({positives / max(len(train_dataset), 1):.1%})"
            )
    return "\n".join(lines)


def _label_mode(dataset: SampleDataset) -> str:
    """The label type: the ``mode`` of the dataset's label processor."""
    processor = next(iter(dataset.output_processors.values()), None)
    return processor.mode if isinstance(processor, LabelProcessor) else "unknown"


def _schema_names(schema: dict) -> str:
    return ", ".join(
        f"{k} ({v if isinstance(v, str) else getattr(v, '__name__', type(v).__name__)})"
        for k, v in schema.items()
    )


def _positive_count(dataset: SampleDataset) -> int | None:
    key = next(iter(dataset.output_schema))
    if len(dataset) > MAX_LABEL_SCAN:
        return None
    try:
        return int(sum(float(dataset[i][key]) > 0.5 for i in range(len(dataset))))
    except Exception:  # noqa: BLE001 - labels that are not 0/1 scalars
        return None
