"""Fixed-width feature encodings for SEER survival prediction.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import torch
from sklearn.preprocessing import StandardScaler

from .tensor_processor import TensorProcessor

_CONTINUOUS = {"diagnosis_year", "tumor_size"}
_NUMERIC = _CONTINUOUS | {
    "primary_site", "histology", "extension", "lymph_nodes", "metastases",
}


def _category(value: str | int | None) -> str:
    return "" if value is None else str(value).strip()


def _is_number(value: str) -> bool:
    return value.isascii() and value.isdecimal()


def _measurement(field: str, value: str) -> float | None:
    if not _is_number(value):
        return None
    number = int(value)
    if field == "tumor_size" and not 1 <= number <= 988:
        return None
    return float(number)


class SEERFeatureProcessor(TensorProcessor):
    """Fit tabular encodings compatible with PyHealth models.

    Args:
        fields: Ordered diagnosis-time field names.
        encoding: One-hot categories or standardized numeric and ordinal codes.

    Field slices map the original field names to their encoded column positions.

    In one-hot mode, tumor sizes outside 1-988 (SEER special codes such as
    999 for unknown) get their own indicator columns. In numeric mode, coded
    fields keep their SEER code as a number, so a special code such as 999 is
    standardized like a measurement.

    Examples:
        >>> processor = SEERFeatureProcessor(("stage",), "one_hot")
        >>> processor.fit([{"features": {"stage": "I"}}], "features")
        >>> processor.process({"stage": "I"}).tolist()
        [0.0, 0.0, 1.0]
    """

    def __init__(
        self, fields: tuple[str, ...], encoding: Literal["one_hot", "numeric"]
    ) -> None:
        super().__init__()
        if encoding not in {"one_hot", "numeric"}:
            raise ValueError("encoding must be one_hot or numeric")
        self.learns_statistics = True
        self.fields = fields
        self.encoding = encoding
        self.field_slices: dict[str, slice] = {}
        self._levels: dict[str, dict[str, int]] = {}
        self._scaling: dict[str, tuple[float, float]] = {}

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Learn category vocabularies and scaling from the supplied partition.

        Args:
            samples: Training samples, each holding a mapping of SEER fields.
            field: Key of the field mapping in each sample.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        counts: dict[str, Counter[str]] = {name: Counter() for name in self.fields}
        for sample in samples:
            for name, counter in counts.items():
                counter[_category(sample[field][name])] += 1
        if not any(counts.values()):
            raise ValueError("SEER encoding requires a nonempty training partition")

        offset = 0
        for name, counter in counts.items():
            continuous = name in _CONTINUOUS
            if self.encoding == "one_hot":
                measured = {
                    value: number for value in counter
                    if continuous and (number := _measurement(name, value)) is not None
                }
                levels = sorted(set(counter) - measured.keys() - {""})
                self._levels[name] = {
                    value: index for index, value in enumerate(levels, start=2)
                }
                width = int(continuous) + 2 + len(levels)
            else:
                levels = sorted(set(counter) - {""})
                self._levels[name] = {
                    value: index for index, value in enumerate(levels)
                }
                measured = {
                    value: self._numeric_value(name, value) for value in counter
                }
                width = 1
            if measured:
                scaler = StandardScaler().fit(
                    [[number] for number in measured.values()],
                    sample_weight=[counter[value] for value in measured],
                )
                self._scaling[name] = (float(scaler.mean_[0]), float(scaler.scale_[0]))
            else:
                self._scaling[name] = (0.0, 1.0)
            self.field_slices[name] = slice(offset, offset + width)
            offset += width
        self._feature_dim = offset

    def _numeric_value(self, field: str, value: str) -> float:
        if not value:
            return -1.0
        if field in _NUMERIC and _is_number(value):
            return float(value)
        return float(self._levels[field][value])

    def process(self, value: Mapping[str, str | int | None]) -> torch.Tensor:
        """Convert raw fields into a vector using fitted training statistics.

        Args:
            value: Mapping from each SEER field name to its raw value.

        Returns:
            A float vector of length ``size()``.

        Raises:
            RuntimeError: If the processor has not been fitted.
        """
        width = self.size()
        if width is None:
            raise RuntimeError("Fit the SEER processor on training samples first")
        vector = torch.zeros(width, dtype=self.dtype)
        for name, columns in self.field_slices.items():
            category = _category(value[name])
            center, scale = self._scaling[name]
            start = columns.start
            levels = self._levels[name]
            if self.encoding == "numeric":
                if category in levels or not category or (
                    name in _NUMERIC and _is_number(category)
                ):
                    number = self._numeric_value(name, category)
                    vector[start] = (number - center) / scale
                continue
            if name in _CONTINUOUS:
                measurement = _measurement(name, category)
                if measurement is not None:
                    vector[start] = (measurement - center) / scale
                    continue
                start += 1
            category_index = levels.get(category, 1) if category else 0
            vector[start + category_index] = 1.0
        return vector
