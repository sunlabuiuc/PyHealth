"""Shared view recipes for tasks.

A *view* is one input schema of a task: which processor, with which settings,
turns each sample field into model input (see
:class:`~pyhealth.tasks.BaseTask`). Tasks whose samples have the same layout
take their views from one recipe here instead of writing processor settings
by hand, so a view name means the same thing in every task and a model can
say which views it reads (``accepted_views``).

View names
----------
``"jagged"``
    Each field is a :class:`~pyhealth.processors.JaggedFeatureProcessor`
    record structure: a ``series`` of hourly records for the measurements and
    a ``single`` record for the static features. Read by
    :class:`~pyhealth.models.RNN_V2`.

More views (``"dense"``, ``"tabular"``, ``"timed"``, ``"codes"``) are planned;
:data:`VIEW_NAMES` lists the names models may declare.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

#: Every view name a recipe may offer and a model may accept.
VIEW_NAMES: tuple[str, ...] = ("jagged", "dense", "timed", "tabular", "codes")


@dataclass(frozen=True)
class Preprocessing:
    """How a view fills in and scales numeric features.

    Everything here that is learned from data (means, standard deviations) is
    learned when the processors are fitted, so on the training part only with
    ``set_task(task, split=...)``.

    Args:
        series_impute: Imputation steps for a time series, applied in order.
        static_impute: Imputation steps for a single static record.
        normalize: Normalization of the values, or None.
        missing_indicator: Whether to add one "was missing" flag per feature.

    Examples:
        >>> from pyhealth.tasks.views import YAIB_PREPROCESSING
        >>> [step["method"] for step in YAIB_PREPROCESSING.series_impute]
        ['forward_fill', 'mean', 'zero']
    """

    series_impute: tuple[dict[str, Any], ...] = ()
    static_impute: tuple[dict[str, Any], ...] = ()
    normalize: dict[str, Any] | None = field(default=None)
    missing_indicator: bool = False


#: YAIB's preprocessing. Measurements are forward-filled; values before the
#: first measurement become the training mean (0 once standardized); a
#: feature never seen in training becomes 0. Static features become the
#: training mean, then 0. All values are standardized and each feature gets a
#: missing-value indicator.
YAIB_PREPROCESSING = Preprocessing(
    series_impute=(
        {"method": "forward_fill"},
        {"method": "mean", "by_dims": ()},  # per feature
        {"method": "zero"},
    ),
    static_impute=(
        {"method": "mean", "by_dims": ()},  # per feature
        {"method": "zero"},
    ),
    normalize={"method": "zscore"},
    missing_indicator=True,
)


def _jagged_level(
    ordering: str, width: int, impute: tuple[dict[str, Any], ...], pre: Preprocessing
) -> dict[str, Any]:
    level: dict[str, Any] = {"ordering": ordering, "values": [width]}
    if impute:
        level["impute"] = copy.deepcopy(list(impute))
    if pre.normalize is not None:
        level["normalize"] = dict(pre.normalize)
    if pre.missing_indicator:
        level["missing_indicator"] = True
    return level


def numeric_series_views(
    dynamic: int,
    static: int,
    preprocessing: Preprocessing = YAIB_PREPROCESSING,
    dynamic_key: str = "dynamic",
    static_key: str = "static",
) -> dict[str, dict[str, Any]]:
    """Views for a task with a numeric time series and a static record.

    The task's samples hold, under ``dynamic_key``, a series
    ``{"time": [T], "values": [T, dynamic]}`` and, under ``static_key``, one
    record ``{"values": [[...static values]]}``, with ``NaN`` for a value
    that was not measured. The task itself does not impute or scale.

    Args:
        dynamic: Number of features in the time series.
        static: Number of static features.
        preprocessing: Imputation and scaling shared by the views.
        dynamic_key: Sample key of the time series.
        static_key: Sample key of the static record.

    Returns:
        View name to input schema, for a task's ``views`` attribute.

    Examples:
        >>> from pyhealth.tasks.views import numeric_series_views
        >>> views = numeric_series_views(dynamic=48, static=4)
        >>> sorted(views)
        ['jagged']
        >>> views["jagged"]["dynamic"][0]
        'jagged'
        >>> views["jagged"]["dynamic"][1]["levels"][0]["values"]
        [48]
    """
    pre = preprocessing
    return {
        "jagged": {
            dynamic_key: (
                "jagged",
                {"levels": [_jagged_level("series", dynamic, pre.series_impute, pre)]},
            ),
            static_key: (
                "jagged",
                {"levels": [_jagged_level("single", static, pre.static_impute, pre)]},
            ),
        },
    }
