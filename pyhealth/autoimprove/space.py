"""The configurations an improver may propose, and how they are checked.

A :class:`PipelineConfig` names a PyHealth model, its constructor arguments
and its training settings. A :class:`SearchSpace` lists the models and the
allowed range of every argument. Proposals from any source (an LLM, random
search, a person) are checked against the space before anything runs, so no
proposed text is ever executed.

Every model also has a plain-language description and a rung on an
interpretability ladder, and every argument a plain-language meaning, so a
report can say what a configuration does in words a clinician can check.
"""

from __future__ import annotations

import importlib
import json
import math
import random
from collections.abc import Callable, Sequence
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
        meaning: What a value means, in plain words: a format string with
            ``{value}``, a dict from value to text, or a function of the value.

    Examples:
        >>> Param("int", 16, 256, default=128).check(1000)
        (256, 'clipped to 256')
        >>> Param("choice", choices=("GRU", "LSTM"), default="GRU").check("RNN")
        ('GRU', "'RNN' is not one of ['GRU', 'LSTM']; used 'GRU'")
        >>> Param("int", 1, 50, meaning="up to {value} passes over the data").explain(30)
        'up to 30 passes over the data'
    """

    kind: str
    low: float | None = None
    high: float | None = None
    choices: Sequence[Any] = ()
    log: bool = False
    default: Any = None
    meaning: str | dict | Callable[[Any], str] = ""

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

    def explain(self, value: Any) -> str:
        """What ``value`` means in plain words; empty if no meaning is set."""
        if callable(self.meaning):
            return self.meaning(value)
        if isinstance(self.meaning, dict):
            return self.meaning.get(value, "")
        if self.meaning:
            return self.meaning.format(value=value)
        return ""


@dataclass(frozen=True)
class ModelSpec:
    """A model the improver may use, and its tunable constructor arguments.

    Args:
        target: Import path of the model class, e.g. ``"pyhealth.models.RNN"``.
        params: Tunable constructor arguments.
        trainer: ``"torch"`` for models trained with :class:`~pyhealth.trainer.
            Trainer`, ``"fit"`` for models with their own ``fit`` method (the
            gradient-boosted trees).
        description: What the model is, in plain words.
        rung: Place on the interpretability ladder, 1 being the easiest to
            check (see :meth:`SearchSpace.rung`): a number, or a function of
            the model arguments.

    Examples:
        >>> spec = ModelSpec("pyhealth.models.RNN", {"hidden_dim": Param("int", 16, 256)})
        >>> spec.load().__name__
        'RNN'
    """

    target: str
    params: dict[str, Param] = field(default_factory=dict)
    trainer: str = "torch"
    description: str = ""
    rung: int | Callable[[dict[str, Any]], int] = 4

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
        hypothesis: What the proposer expects, stated before the trial runs:
            ``expected`` (e.g. "PR-AUC up, smaller spread between folds"),
            ``kind`` (``"clinical"`` or ``"technical"``) and ``based_on``
            (the id of the trial it changes, if any).

    Examples:
        >>> config = PipelineConfig("RNN", {"hidden_dim": 64}, {"lr": 1e-3})
        >>> config.key()
        '{"model": "RNN", "model_params": {"hidden_dim": 64}, "train_params": {"lr": 0.001}}'
    """

    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    train_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    hypothesis: dict[str, Any] = field(default_factory=dict)

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

        The hypothesis may be nested under ``"hypothesis"`` or given as
        top-level ``expected``, ``kind`` and ``based_on`` keys.

        Raises:
            TypeError: If ``model_params``, ``train_params`` or ``hypothesis``
                is not a dict.
        """
        params = {}
        for key in ("model_params", "train_params", "hypothesis"):
            value = data.get(key) or {}
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be an object, not {type(value).__name__}")
            params[key] = dict(value)
        for key in ("expected", "kind", "based_on"):
            if data.get(key) not in (None, ""):
                params["hypothesis"].setdefault(key, data[key])
        return cls(
            model=str(data.get("model", "")),
            rationale=str(data.get("rationale", "")),
            **params,
        )


def config_diff(old: PipelineConfig, new: PipelineConfig) -> dict[str, str]:
    """The settings that differ between two configs, as ``"old → new"``.

    When the model changes, its arguments are not compared (they belong to
    different models), and only the training settings both configs have.

    Examples:
        >>> config_diff(PipelineConfig("XGBoost", {"max_depth": 4}),
        ...             PipelineConfig("XGBoost", {"max_depth": 2}))
        {'max_depth': '4 → 2'}
    """
    out: dict[str, str] = {}
    if old.model != new.model:
        out["model"] = f"{old.model} → {new.model}"
        shared = set(old.train_params) & set(new.train_params)
        _diff_into(out, {k: old.train_params[k] for k in shared},
                   {k: new.train_params[k] for k in shared})
    else:
        _diff_into(out, old.model_params, new.model_params)
        _diff_into(out, old.train_params, new.train_params)
    return out


def _diff_into(out, old, new):
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a != b:
            out[key] = f"{_short(a)} → {_short(b)}"


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _pct(value: float) -> str:
    return f"{value:.0%}"


TRAIN_PARAMS: dict[str, Param] = {
    "epochs": Param("int", 1, 50, default=10,
                    meaning="trained for up to {value} passes over the training data"),
    "lr": Param("float", 1e-5, 1e-1, log=True, default=1e-3,
                meaning=lambda v: f"learning rate {v:.2g}: the size of each update"),
    # Log scale up to 0.1: small datasets can need strong L2. 1e-6 is
    # effectively none.
    "weight_decay": Param(
        "float", 1e-6, 1e-1, log=True, default=1e-6,
        meaning=lambda v: (
            "no weight penalty" if v <= 1e-6 else
            f"weight penalty {v:.2g}: larger values keep the model simpler"
        ),
    ),
    "batch_size": Param("choice", choices=(16, 32, 64, 128, 256), default=64,
                        meaning="{value} samples per update"),
    "pos_weight": Param(
        "choice", choices=("none", "balanced"), default="none",
        meaning={
            "none": "no class weighting: predicted risks stay on the scale of "
                    "observed frequencies",
            "balanced": "positive samples weighted up to balance the classes: "
                        "predicted risks are inflated and need recalibration",
        },
    ),
    "patience": Param(
        "int", 1, 10, default=3,
        meaning="training stops after {value} epochs without improvement on "
                "the validation split",
    ),
}
"""Training settings for models trained with :class:`~pyhealth.trainer.Trainer`."""

_EMB = Param("int", 16, 256, default=128,
             meaning="each input is first mapped to {value} numbers")
_HID = Param("int", 16, 256, default=128, meaning="{value} hidden units")
_DROP = Param("float", 0.0, 0.7, default=0.5,
              meaning=lambda v: f"{v:.0%} of units dropped at random while "
                                "training, against overfitting")

DEFAULT_MODELS: dict[str, ModelSpec] = {
    "LogisticRegression": ModelSpec(
        "pyhealth.models.LogisticRegression",
        {"embedding_dim": Param(
            "int", 16, 256, default=128,
            meaning="inputs pass through a linear map to {value} numbers; the "
                    "model stays linear",
        )},
        description="logistic regression: each input adds a fixed amount to "
                    "the log-odds of the outcome",
        rung=1,
    ),
    "MLP": ModelSpec(
        "pyhealth.models.MLP",
        {"embedding_dim": _EMB, "hidden_dim": _HID,
         "n_layers": Param("int", 1, 4, default=2, meaning="{value} hidden layers")},
        description="a feed-forward neural network: inputs are combined in "
                    "hidden layers, so their effects can interact",
        rung=3,
    ),
    "RNN": ModelSpec(
        "pyhealth.models.RNN",
        {"embedding_dim": _EMB, "hidden_dim": _HID,
         "rnn_type": Param("choice", choices=("GRU", "LSTM", "RNN"), default="GRU",
                           meaning="{value} cells"),
         "num_layers": Param("int", 1, 3, default=1, meaning="{value} stacked layers"),
         "dropout": _DROP,
         "bidirectional": Param(
             "choice", choices=(False, True), default=False,
             meaning={False: "reads the record forward in time",
                      True: "reads the record forward and backward"},
         )},
        description="a recurrent neural network that reads the record in time "
                    "order",
    ),
    "Transformer": ModelSpec(
        "pyhealth.models.Transformer",
        # Attention needs embedding_dim divisible by heads.
        {"embedding_dim": Param(
            "choice", choices=(16, 32, 64, 128, 256), default=128,
            meaning="each input is first mapped to {value} numbers",
        ),
         "heads": Param("choice", choices=(1, 2, 4), default=1,
                        meaning="{value} attention heads"),
         "num_layers": Param("int", 1, 4, default=1, meaning="{value} layers"),
         "dropout": _DROP},
        description="a self-attention network: each event can draw on every "
                    "other event in the record",
    ),
    "RETAIN": ModelSpec(
        "pyhealth.models.RETAIN", {"embedding_dim": _EMB, "dropout": _DROP},
        description="a reverse-time attention network whose visit and code "
                    "weights can be inspected",
    ),
    "CNN": ModelSpec(
        "pyhealth.models.CNN",
        {"embedding_dim": _EMB, "hidden_dim": _HID,
         "num_layers": Param("int", 1, 3, default=1, meaning="{value} layers")},
        description="a convolutional network over the event sequence",
    ),
}
"""Models available by default. All accept code-sequence inputs."""

LADDER = {
    1: "linear: each input has one coefficient (an odds ratio)",
    2: "additive: each input has one effect curve, with no interactions",
    3: "small interactions between inputs",
    4: "deep interactions or sequence models: explained only after the fact",
}
"""The interpretability ladder: lower rungs are easier for a clinician to check."""


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
            config.model, model_params, train_params, config.rationale,
            dict(config.hypothesis),
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

    def rung(self, config: PipelineConfig) -> int:
        """The config's rung on the interpretability ladder (:data:`LADDER`).

        Models not in the space get the top rung.

        Examples:
            >>> space = with_xgboost(SearchSpace())
            >>> space.rung(PipelineConfig("LogisticRegression"))
            1
            >>> space.rung(PipelineConfig("XGBoost", {"max_depth": 1}))
            2
        """
        spec = self.models.get(config.model)
        if spec is None:
            return max(LADDER)
        return spec.rung(config.model_params) if callable(spec.rung) else spec.rung

    def explain(self, config: PipelineConfig) -> list[str]:
        """The config in plain words: the model, then each setting.

        Examples:
            >>> space = with_xgboost(SearchSpace())
            >>> space.explain(PipelineConfig("XGBoost", {"max_depth": 2}))[:2]
            ['XGBoost: gradient-boosted decision trees', 'max_depth = 2: each tree combines at most 2 inputs']
        """
        spec = self.models.get(config.model)
        if spec is None:
            return [config.model] + [
                f"{k} = {_short(v)}"
                for k, v in {**config.model_params, **config.train_params}.items()
            ]
        lines = [f"{config.model}: {spec.description}" if spec.description
                 else config.model]
        train_space = self.train_params if spec.trainer == "torch" else {}
        for params, space in ((config.model_params, spec.params),
                              (config.train_params, train_space)):
            # the space's order, so a config reads the same however it was built
            order = [k for k in space if k in params] + sorted(
                k for k in params if k not in space
            )
            for name in order:
                value = params[name]
                meaning = space[name].explain(value) if name in space else ""
                line = f"{name} = {_short(value)}"
                lines.append(f"{line}: {meaning}" if meaning else line)
        return lines

    def describe(self) -> str:
        """The space in plain text, for an LLM prompt."""
        lines = ["Models and their arguments (rung 1 is the easiest to interpret):"]
        for name in sorted(self.models):
            spec = self.models[name]
            args = ", ".join(f"{k} ({p.describe()})" for k, p in spec.params.items())
            rung = "rung by depth" if callable(spec.rung) else f"rung {spec.rung}"
            lines.append(f"- {name} [{rung}]: {args or 'no tunable arguments'}")
        if self.train_params:
            lines.append("Training settings (all models above):")
            for k, p in self.train_params.items():
                lines.append(f"- {k} ({p.describe()})")
        return "\n".join(lines)


def _xgboost_rung(params: dict[str, Any]) -> int:
    depth = params.get("max_depth") or 6
    return 2 if depth <= 1 else 3 if depth <= 3 else 4


def _depth_meaning(depth: int) -> str:
    if depth <= 1:
        return "each tree splits on one input, so the model is additive"
    return f"each tree combines at most {depth} inputs"


XGBOOST = ModelSpec(
    "pyhealth.models.XGBoostModel",
    {"n_estimators": Param("int", 50, 1000, default=300, meaning="{value} trees"),
     "max_depth": Param("int", 1, 10, default=4, meaning=_depth_meaning),
     "learning_rate": Param(
         "float", 0.01, 0.3, log=True, default=0.05,
         meaning=lambda v: f"each tree's correction is scaled by {v:.2g}; "
                           "smaller learns more slowly and steadily",
     ),
     "subsample": Param(
         "float", 0.5, 1.0, default=0.8,
         meaning=lambda v: f"each tree sees a random {_pct(v)} of the samples",
     ),
     "colsample_bytree": Param(
         "float", 0.5, 1.0, default=0.8,
         meaning=lambda v: f"each tree sees a random {_pct(v)} of the inputs",
     ),
     "scale_pos_weight": Param(
         "choice", choices=(None, "balanced"), default=None,
         meaning={
             None: "no class weighting: predicted risks stay on the scale of "
                   "observed frequencies",
             "balanced": "positive samples weighted up to balance the classes: "
                         "predicted risks are inflated and need recalibration",
         },
     )},
    trainer="fit",
    description="gradient-boosted decision trees",
    rung=_xgboost_rung,
)
"""XGBoost's search space; depth 1 makes it an additive model (rung 2)."""


def with_xgboost(space: SearchSpace) -> SearchSpace:
    """Adds :class:`~pyhealth.models.XGBoostModel` to a search space.

    XGBoost needs fixed-width (tabular) inputs, so it is not in the default
    space. Add it when the task's features are tensors.

    Examples:
        >>> "XGBoost" in with_xgboost(SearchSpace()).models
        True
    """
    return SearchSpace({**space.models, "XGBoost": XGBOOST}, space.train_params)
