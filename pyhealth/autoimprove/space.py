"""The configurations an improver may propose, and how they are checked.

A :class:`PipelineConfig` names a PyHealth model, its constructor arguments
and its training settings. A :class:`SearchSpace` lists the models and the
allowed range of every argument. Proposals from any source (an LLM, random
search, a person) are checked against the space before anything runs, so no
proposed text is ever executed.
"""

from __future__ import annotations

import importlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    """The allowed values of one argument.

    Args:
        kind: ``"int"``, ``"float"`` or ``"choice"``.
        low: Smallest value (``int`` and ``float``).
        high: Largest value (``int`` and ``float``).
        choices: Allowed values (``choice``).
        log: Sample ``float`` values on a log scale.
        default: Value used when a proposal leaves the argument out.

    Examples:
        >>> Param("int", 16, 256, default=128).check(1000)
        (256, 'clipped to 256')
        >>> Param("choice", choices=("GRU", "LSTM"), default="GRU").check("RNN")
        ('GRU', "'RNN' is not one of ['GRU', 'LSTM']; used 'GRU'")
    """

    kind: str
    low: float | None = None
    high: float | None = None
    choices: Sequence[Any] = ()
    log: bool = False
    default: Any = None

    def check(self, value: Any) -> tuple[Any, str | None]:
        """Returns the value made valid, and a note if it had to change."""
        if self.kind == "choice":
            if value in self.choices:
                return value, None
            return self.default, (
                f"{value!r} is not one of {list(self.choices)}; used {self.default!r}"
            )
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default, f"{value!r} is not a number; used {self.default!r}"
        if math.isnan(number):
            return self.default, f"NaN is not allowed; used {self.default!r}"
        clipped = min(max(number, self.low), self.high)
        if self.kind == "int":
            clipped = round(clipped)
        note = None if clipped == number else f"clipped to {clipped}"
        return clipped, note

    def sample(self, rng: random.Random) -> Any:
        """Draws a random allowed value."""
        if self.kind == "choice":
            return rng.choice(list(self.choices))
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        if self.log:
            return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return rng.uniform(self.low, self.high)

    def describe(self) -> str:
        """A short description for prompts, e.g. ``int in [16, 256]``."""
        if self.kind == "choice":
            return f"one of {list(self.choices)}"
        scale = ", log scale" if self.log else ""
        return f"{self.kind} in [{self.low:g}, {self.high:g}]{scale}"


@dataclass(frozen=True)
class ModelSpec:
    """A model the improver may use, and its tunable constructor arguments.

    Args:
        target: Import path of the model class, e.g. ``"pyhealth.models.RNN"``.
        params: Tunable constructor arguments.
        trainer: ``"torch"`` for models trained with :class:`~pyhealth.trainer.
            Trainer`, ``"fit"`` for models with their own ``fit`` method (the
            gradient-boosted trees).

    Examples:
        >>> spec = ModelSpec("pyhealth.models.RNN", {"hidden_dim": Param("int", 16, 256)})
        >>> spec.load().__name__
        'RNN'
    """

    target: str
    params: dict[str, Param] = field(default_factory=dict)
    trainer: str = "torch"

    def load(self) -> type:
        """Imports and returns the model class."""
        module, _, name = self.target.rpartition(".")
        return getattr(importlib.import_module(module), name)


@dataclass
class PipelineConfig:
    """One pipeline to try: a model, its arguments and how to train it.

    Args:
        model: A model name from the search space, e.g. ``"RNN"``.
        model_params: Constructor arguments for the model.
        train_params: Training settings (``epochs``, ``lr``, ``weight_decay``,
            ``batch_size``, ``pos_weight``).
        rationale: Why the proposer chose it, kept for the report.

    Examples:
        >>> config = PipelineConfig("RNN", {"hidden_dim": 64}, {"lr": 1e-3})
        >>> config.key()
        '{"model": "RNN", "model_params": {"hidden_dim": 64}, "train_params": {"lr": 0.001}}'
    """

    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    train_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The configuration without its rationale, as plain JSON types."""
        return {
            "model": self.model,
            "model_params": dict(sorted(self.model_params.items())),
            "train_params": dict(sorted(self.train_params.items())),
        }

    def key(self) -> str:
        """A canonical string, so that repeated proposals are recognised."""
        return json.dumps(self.to_dict(), sort_keys=False, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Builds a config from a dict such as one parsed from JSON.

        Raises:
            TypeError: If ``model_params`` or ``train_params`` is not a dict.
        """
        params = {}
        for key in ("model_params", "train_params"):
            value = data.get(key) or {}
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be an object, not {type(value).__name__}")
            params[key] = dict(value)
        return cls(
            model=str(data.get("model", "")),
            rationale=str(data.get("rationale", "")),
            **params,
        )


TRAIN_PARAMS: dict[str, Param] = {
    "epochs": Param("int", 1, 50, default=10),
    "lr": Param("float", 1e-5, 1e-1, log=True, default=1e-3),
    # Log scale up to 0.1: small datasets can need strong L2. 1e-6 is
    # effectively none.
    "weight_decay": Param("float", 1e-6, 1e-1, log=True, default=1e-6),
    "batch_size": Param("choice", choices=(16, 32, 64, 128, 256), default=64),
    "pos_weight": Param("choice", choices=("none", "balanced"), default="none"),
    "patience": Param("int", 1, 10, default=3),
}
"""Training settings for models trained with :class:`~pyhealth.trainer.Trainer`."""

_EMB = Param("int", 16, 256, default=128)
_HID = Param("int", 16, 256, default=128)
_DROP = Param("float", 0.0, 0.7, default=0.5)

DEFAULT_MODELS: dict[str, ModelSpec] = {
    "LogisticRegression": ModelSpec(
        "pyhealth.models.LogisticRegression", {"embedding_dim": _EMB}
    ),
    "MLP": ModelSpec(
        "pyhealth.models.MLP",
        {"embedding_dim": _EMB, "hidden_dim": _HID,
         "n_layers": Param("int", 1, 4, default=2)},
    ),
    "RNN": ModelSpec(
        "pyhealth.models.RNN",
        {"embedding_dim": _EMB, "hidden_dim": _HID,
         "rnn_type": Param("choice", choices=("GRU", "LSTM", "RNN"), default="GRU"),
         "num_layers": Param("int", 1, 3, default=1),
         "dropout": _DROP,
         "bidirectional": Param("choice", choices=(False, True), default=False)},
    ),
    "Transformer": ModelSpec(
        "pyhealth.models.Transformer",
        # Attention needs embedding_dim divisible by heads.
        {"embedding_dim": Param(
            "choice", choices=(16, 32, 64, 128, 256), default=128
        ),
         "heads": Param("choice", choices=(1, 2, 4), default=1),
         "num_layers": Param("int", 1, 4, default=1),
         "dropout": _DROP},
    ),
    "RETAIN": ModelSpec(
        "pyhealth.models.RETAIN", {"embedding_dim": _EMB, "dropout": _DROP}
    ),
    "CNN": ModelSpec(
        "pyhealth.models.CNN",
        {"embedding_dim": _EMB, "hidden_dim": _HID,
         "num_layers": Param("int", 1, 3, default=1)},
    ),
}
"""Models available by default. All accept code-sequence inputs."""


class SearchSpace:
    """The models and argument ranges an improver may propose.

    Args:
        models: Model name to :class:`ModelSpec`. Defaults to
            :data:`DEFAULT_MODELS`.
        train_params: Training settings for ``"torch"`` models. Defaults to
            :data:`TRAIN_PARAMS`.

    Examples:
        >>> space = SearchSpace()
        >>> config, notes = space.validate(
        ...     PipelineConfig("RNN", {"hidden_dim": 999, "colour": "red"}, {})
        ... )
        >>> config.model_params["hidden_dim"], notes
        (256, ['hidden_dim: clipped to 256', 'colour: not an argument of RNN; dropped'])
    """

    def __init__(
        self,
        models: dict[str, ModelSpec] | None = None,
        train_params: dict[str, Param] | None = None,
    ):
        self.models = dict(DEFAULT_MODELS if models is None else models)
        self.train_params = dict(TRAIN_PARAMS if train_params is None else train_params)

    def validate(self, config: PipelineConfig) -> tuple[PipelineConfig, list[str]]:
        """Makes a proposal valid, listing every change that was needed.

        Unknown arguments are dropped, numbers outside their range are
        clipped, and invalid choices fall back to the default.

        Raises:
            ValueError: If the model is not in the space.
        """
        if config.model not in self.models:
            raise ValueError(
                f"Unknown model {config.model!r}; choose one of {sorted(self.models)}"
            )
        spec = self.models[config.model]
        notes: list[str] = []
        model_params = self._check(
            config.model_params, spec.params, config.model, notes
        )
        train_space = self.train_params if spec.trainer == "torch" else {}
        train_params = self._check(
            config.train_params, train_space, "training", notes
        )
        clean = PipelineConfig(
            config.model, model_params, train_params, config.rationale
        )
        return clean, notes

    @staticmethod
    def _check(values, space, owner, notes) -> dict[str, Any]:
        out = {}
        for name, value in values.items():
            if name not in space:
                notes.append(f"{name}: not an argument of {owner}; dropped")
                continue
            out[name], note = space[name].check(value)
            if note:
                notes.append(f"{name}: {note}")
        for name, param in space.items():
            out.setdefault(name, param.default)
        return out

    def sample(self, rng: random.Random) -> PipelineConfig:
        """Draws a random valid configuration."""
        model = rng.choice(sorted(self.models))
        spec = self.models[model]
        model_params = {k: p.sample(rng) for k, p in spec.params.items()}
        train_space = self.train_params if spec.trainer == "torch" else {}
        train_params = {k: p.sample(rng) for k, p in train_space.items()}
        return PipelineConfig(model, model_params, train_params, "random sample")

    def describe(self) -> str:
        """The space in plain text, for an LLM prompt."""
        lines = ["Models and their arguments:"]
        for name in sorted(self.models):
            spec = self.models[name]
            args = ", ".join(f"{k} ({p.describe()})" for k, p in spec.params.items())
            lines.append(f"- {name}: {args or 'no tunable arguments'}")
        if self.train_params:
            lines.append("Training settings (all models above):")
            for k, p in self.train_params.items():
                lines.append(f"- {k} ({p.describe()})")
        return "\n".join(lines)


def with_xgboost(space: SearchSpace) -> SearchSpace:
    """Adds :class:`~pyhealth.models.XGBoostModel` to a search space.

    XGBoost needs fixed-width (tabular) inputs, so it is not in the default
    space. Add it when the task's features are tensors.

    Examples:
        >>> "XGBoost" in with_xgboost(SearchSpace()).models
        True
    """
    models = dict(space.models)
    models["XGBoost"] = ModelSpec(
        "pyhealth.models.XGBoostModel",
        {"n_estimators": Param("int", 50, 1000, default=300),
         "max_depth": Param("int", 2, 10, default=4),
         "learning_rate": Param("float", 0.01, 0.3, log=True, default=0.05),
         "subsample": Param("float", 0.5, 1.0, default=0.8),
         "colsample_bytree": Param("float", 0.5, 1.0, default=0.8),
         "scale_pos_weight": Param("choice", choices=(None, "balanced"), default=None)},
        trainer="fit",
    )
    return SearchSpace(models, space.train_params)
