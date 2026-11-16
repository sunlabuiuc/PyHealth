"""Task cards: what a task predicts, for whom and when, and why each choice was made.

A task's code fixes its cohort, prediction time, outcome and features, but
not the reasons for them. A :class:`TaskCard` records those reasons in a form
reports can render. It also holds a dictionary of the task's features:
meaning, clinical group and the expected direction of each feature's effect.
Reports use the dictionary to explain a model in clinical terms and to check
that what it learned is plausible. Tasks provide a card through the optional
:meth:`~pyhealth.tasks.BaseTask.task_card` hook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DIRECTIONS = {
    "+": "higher values mean higher risk",
    "-": "higher values mean lower risk",
    "u": "both low and high values mean higher risk",
    "": "no expected direction",
}
"""Expected directions of a feature's effect on the outcome."""


@dataclass
class FeatureInfo:
    """One feature in a task card's dictionary.

    Args:
        meaning: What the value is, in plain words, e.g. "highest lactate in
            the first 48 h, in SDs from normal".
        group: Clinical factor group, e.g. "labs", "vital signs".
        direction: Expected effect on the outcome: ``"+"``, ``"-"``, ``"u"``
            (U-shaped) or ``""`` (no expectation); see :data:`DIRECTIONS`.
        unit: Unit of the value, if any.
        rationale: Why the feature is clinically relevant.

    Examples:
        >>> FeatureInfo("highest lactate in the first 48 h", "labs", "+").direction
        '+'
    """

    meaning: str
    group: str = ""
    direction: str = ""
    unit: str = ""
    rationale: str = ""

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"direction must be one of {list(DIRECTIONS)}, not {self.direction!r}"
            )


def _choice(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {"text": str(value.get("text", "")),
                "rationale": str(value.get("rationale", ""))}
    if isinstance(value, (tuple, list)):
        text, rationale = (list(value) + ["", ""])[:2]
        return {"text": str(text), "rationale": str(rationale)}
    return {"text": str(value or ""), "rationale": ""}


@dataclass
class TaskCard:
    """What a task predicts, for whom and when, with the reason for each choice.

    Each design item is a text or a ``(text, rationale)`` pair.

    Args:
        population: Who is included.
        prediction_time: When the prediction is made.
        outcome: What is predicted.
        horizon: Over what period after the prediction time.
        exclusions: Who is left out, each with its reason.
        features: Feature name to :class:`FeatureInfo`.
        columns: For fields with named columns (tensors, time series), the
            input field to its column names, in order.
        notes: Anything else a reader should know, e.g. known data limits.

    Examples:
        >>> card = TaskCard(
        ...     population=("Adults in hospital at 48 h", "the prediction time"),
        ...     outcome="death before discharge",
        ...     features={"age": FeatureInfo("age in years", "demographics", "+")},
        ...     columns={"tabular": ["age"]},
        ... )
        >>> card.population["rationale"], card.feature("age").group
        ('the prediction time', 'demographics')
        >>> TaskCard.from_dict(card.to_dict()) == card
        True
    """

    population: Any = ""
    prediction_time: Any = ""
    outcome: Any = ""
    horizon: Any = ""
    exclusions: list[Any] = field(default_factory=list)
    features: dict[str, FeatureInfo] = field(default_factory=dict)
    columns: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        for name in ("population", "prediction_time", "outcome", "horizon"):
            setattr(self, name, _choice(getattr(self, name)))
        self.exclusions = [_choice(x) for x in self.exclusions]
        self.features = {
            k: v if isinstance(v, FeatureInfo) else FeatureInfo(**v)
            for k, v in self.features.items()
        }
        self.columns = {k: list(v) for k, v in self.columns.items()}

    def feature(self, name: str) -> FeatureInfo | None:
        """The dictionary entry for a feature, or ``None``."""
        return self.features.get(name)

    def design(self) -> list[tuple[str, dict[str, str]]]:
        """The design items in order, with their names."""
        items = [("Population", self.population),
                 ("Prediction time", self.prediction_time),
                 ("Outcome", self.outcome), ("Horizon", self.horizon)]
        items += [("Exclusion", x) for x in self.exclusions]
        return [(k, v) for k, v in items if v["text"]]

    def to_dict(self) -> dict[str, Any]:
        """The card as JSON-ready types."""
        data = asdict(self)
        data["features"] = {k: asdict(v) for k, v in self.features.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskCard:
        """Rebuilds a card saved with :meth:`to_dict`."""
        return cls(**data)
