"""A run's settings, as one typed object that reads from and writes to YAML.

:class:`RunConfig` groups every setting of an
:class:`~pyhealth.autoimprove.AutoImprover` run into sections (``search``, ``budget``, ``evaluation``, ...). It is written
in YAML, or built in Python, and every key is checked: a misspelled key or a
value of the wrong type is an error, not a silently ignored setting::

    search:
      metric: pr_auc
      proposals_per_round: 3
      instructions: Prefer models a clinician can read.
    budget: {rounds: 5, patience: 2}
    baseline: {model: RNN, model_params: {hidden_dim: 128}, train_params: {epochs: 5}}
    proposer: {type: llm, backend: claude}
    evaluation: {cv_folds: 5}
    output: {run_dir: runs/mortality}

Objects that YAML cannot describe (the datasets, a custom proposer, a
:class:`~pyhealth.autoimprove.SearchSpace` or
:class:`~pyhealth.autoimprove.FeatureSpace`, the task) are passed to the
improver in Python and take precedence over the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .search.space import (
    DEFAULT_MODELS,
    TRAIN_PARAMS,
    XGBOOST,
    Param,
    PipelineConfig,
    SearchSpace,
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchConfig(_Section):
    """What the search optimizes and how each round is made.

    Attributes:
        metric: The metric selection is based on. Defaults by task type
            (``pr_auc`` for binary, ``f1_macro`` for multiclass, ...).
        report_metrics: Other metrics recorded for every trial.
        plan: Kinds of round, repeated in order: ``model`` (a config from the
            search space), ``features`` (engineered inputs) and ``code`` (a
            model the proposer writes). Evaluations and analyses are not
            rounds: see the ``guardrails`` and ``analysis`` sections. Defaults to ``[model]``, or two
            feature rounds then a model round when a FeatureSpace is given.
        proposals_per_round: How many agents propose in each round.
        mode: ``batched``: all of a round's proposals are made from the
            results before the round, then scored. ``sequential``: each agent
            proposes after the previous one's trial is scored.
        workers: Trials scored at once in batched mode; above 1 they run in
            separate processes.
        instructions: A plain-text strategy from you, added to every prompt.
        strategies: The strategies agents rotate through after the first
            round (where all explore): some of ``explore``, ``refine``,
            ``crossover``, ``contrastive``. Defaults to all four.
        seed: Seed for folds, training and sampling.
        device: Torch device; defaults to CUDA when available.
    """

    metric: str | None = None
    report_metrics: list[str] = Field(default_factory=list)
    plan: list[str] | None = None
    proposals_per_round: int = Field(3, ge=1)
    mode: Literal["batched", "sequential"] = "batched"
    workers: int = Field(1, ge=1)
    instructions: str = ""
    strategies: list[str] | None = None
    seed: int = 0
    device: str | None = None

    @field_validator("strategies")
    @classmethod
    def _known_strategies(cls, names):
        from .search.prompts import STRATEGIES

        unknown = [n for n in names or [] if n not in STRATEGIES]
        if unknown or names == []:
            raise ValueError(f"strategies must be some of {list(STRATEGIES)}; "
                             f"got {names}")
        return names


class BudgetConfig(_Section):
    """When the search stops: whichever limit is reached first.

    Attributes:
        rounds: Most rounds to run; 0 scores only the baseline.
        patience: Rounds without a better selection before stopping.
        max_trials: Most trials to score, baseline included.
        time_budget: Most seconds to search.
        early_stop: Stop a trial's folds early once it is this many standard
            errors behind the incumbent.
    """

    rounds: int = Field(5, ge=0)
    patience: int = Field(2, ge=1)
    max_trials: int | None = None
    time_budget: float | None = None
    early_stop: float | None = None


class BaselineConfig(_Section):
    """The starting pipeline every trial is compared with."""

    model: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    train_params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    code: str = ""

    def to_pipeline(self) -> PipelineConfig:
        return PipelineConfig(self.model, dict(self.model_params),
                              dict(self.train_params), rationale=self.rationale,
                              code=self.code)


class ParamConfig(_Section):
    """One searchable setting: see :class:`~pyhealth.autoimprove.Param`."""

    kind: Literal["int", "float", "choice"]
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log: bool = False
    default: Any = None
    meaning: str = ""

    def to_param(self) -> Param:
        return Param(self.kind, self.low, self.high, choices=tuple(self.choices or ()),
                     log=self.log, default=self.default, meaning=self.meaning)


class SpaceConfig(_Section):
    """Which models, and which training settings, the search may use.

    Attributes:
        models: Model names from the default space (plus ``XGBoost``).
            Defaults to all default models.
        train_params: Training settings to add or override, by name.
    """

    models: list[str] | None = None
    train_params: dict[str, ParamConfig] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def _known(cls, models):
        known = {**DEFAULT_MODELS, "XGBoost": XGBOOST}
        unknown = [m for m in models or [] if m not in known]
        if unknown:
            raise ValueError(f"unknown models {unknown}; known: {sorted(known)}")
        return models

    def build(self) -> SearchSpace:
        known = {**DEFAULT_MODELS, "XGBoost": XGBOOST}
        models = None if self.models is None else {m: known[m] for m in self.models}
        changed = {k: v.to_param() for k, v in self.train_params.items()}
        train = {**TRAIN_PARAMS, **changed}
        return SearchSpace(models=models, train_params=train)


class ProposerConfig(_Section):
    """Who proposes: random search, or an LLM.

    Attributes:
        type: ``random`` or ``llm``.
        backend: For ``llm``: ``claude``, ``openai`` or ``gemini``.
        model: The LLM's model id; defaults to the backend's default.
        seed: For ``random``.
        options: Other keyword arguments for the LLM client (``effort``,
            ``max_tokens``, ``base_url``...).
    """

    type: Literal["random", "llm"] = "random"
    backend: Literal["claude", "openai", "gemini"] = "claude"
    model: str | None = None
    seed: int = 0
    options: dict[str, Any] = Field(default_factory=dict)

    def build(self):
        from .search.proposers import LLMProposer, RandomProposer

        if self.type == "random":
            return RandomProposer(seed=self.seed)
        from .utils.llm import ClaudeLLM, GeminiLLM, OpenAILLM

        clients = {"claude": ClaudeLLM, "openai": OpenAILLM, "gemini": GeminiLLM}
        client = clients[self.backend]
        kwargs = dict(self.options)
        if self.model:
            kwargs["model"] = self.model
        return LLMProposer(client(**kwargs))


class EvaluationConfig(_Section):
    """How trials are scored and the deployed one is chosen.

    Attributes:
        cv_folds: Patient-grouped folds of the training split; without them
            each trial is scored once on the validation split.
        promote_se: A trial is a candidate when it beats the baseline by this
            many paired standard errors.
        one_se: Among candidates, select the simplest within one SE of the best.
        check: Remove models that cannot read the inputs before searching.
        suspicion_auc: An engineered feature that alone separates the labels
            this well is flagged as possible leakage.
        min_events_per_fold: With ``cv_folds`` on a binary task, warn and
            raise a ``few_events`` flag when a fold has fewer positive
            samples than this. With so few events one lucky fold can carry a
            trial's whole gain, so the search's gains are unconfirmed.
    """

    cv_folds: int | None = None
    promote_se: float | None = 1.0
    one_se: bool = True
    check: bool = True
    suspicion_auc: float = 0.95
    min_events_per_fold: int = Field(10, ge=1)


class ConfirmConfig(_Section):
    """Re-scoring the leaders on new folds and seeds after the search."""

    enabled: bool = False
    top: int = 3
    seeds: int = 3
    se: float = 2.0
    prune: bool = True


class CodeConfig(_Section):
    """Code rounds: models the proposer writes."""

    timeout: float = 1800.0


class GuardrailConfig(_Section):
    """Evaluations the proposer writes, adopted as guardrails.

    Attributes:
        enabled: After rounds, the proposer writes new ways to score
            predictions; one that passes the checks becomes a guardrail, which
            removes candidates clearly worse than the baseline on it. Needs
            ``evaluation.cv_folds``.
        every: Propose evaluations after every this many rounds.
        turns: Evaluations proposed each time.
        max: Most guardrails active at once; more are kept for the report.
        se: Standard errors by which a trial must be worse than the baseline
            on a guardrail to be removed.
        timeout: Seconds one evaluation may take.
    """

    enabled: bool = False
    every: int = Field(1, ge=1)
    turns: int = Field(1, ge=1)
    max: int = 3
    se: float = 1.0
    timeout: float = 300.0


class AnalysisConfig(_Section):
    """Analyses of the training data the proposer writes.

    Attributes:
        enabled: Before rounds, the proposer asks a question of the training
            data in code; only aggregate tables (groups under 11 patients
            withheld) come back, and its findings go into later prompts.
        every: Run analyses before every this many rounds, starting with the
            first.
        turns: Analyses each time.
        max_findings: Most findings kept for prompts.
    """

    enabled: bool = False
    every: int = Field(1, ge=1)
    turns: int = Field(1, ge=1)
    max_findings: int = 12


class OutputConfig(_Section):
    """Where a run is kept.

    Attributes:
        run_dir: The run's directory; a run there is resumed.
        lessons_path: A file of lessons shared across runs.
    """

    run_dir: str | None = None
    lessons_path: str | None = None


class RunConfig(_Section):
    """Every setting of an AutoImprover run.

    Examples:
        >>> config = RunConfig(budget={"rounds": 3}, evaluation={"cv_folds": 5})
        >>> config.budget.rounds, config.search.proposals_per_round
        (3, 3)
        >>> RunConfig.from_yaml_text(config.to_yaml()) == config
        True
    """

    search: SearchConfig = Field(default_factory=SearchConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    baseline: BaselineConfig | None = None
    space: SpaceConfig | None = None
    proposer: ProposerConfig = Field(default_factory=ProposerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    confirm: ConfirmConfig = Field(default_factory=ConfirmConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("baseline", mode="before")
    @classmethod
    def _baseline(cls, value):
        if isinstance(value, PipelineConfig):
            return {"model": value.model, "model_params": dict(value.model_params),
                    "train_params": dict(value.train_params),
                    "rationale": value.rationale, "code": value.code}
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Reads a config from a YAML file."""
        return cls.from_yaml_text(Path(path).read_text())

    @classmethod
    def from_yaml_text(cls, text: str) -> RunConfig:
        return cls.model_validate(yaml.safe_load(text) or {})

    def to_yaml(self) -> str:
        """The config as YAML, leaving out settings at their defaults."""
        return yaml.safe_dump(self.model_dump(exclude_defaults=True), sort_keys=False)

    def with_overrides(self, **sections: Any) -> RunConfig:
        """A copy with some settings changed, checked like the original.

        Each keyword is a section; a dict is merged into that section, any
        other value replaces it.

        Examples:
            >>> RunConfig().with_overrides(budget={"rounds": 2}).budget.rounds
            2
        """
        data = self.model_dump()
        for name, value in sections.items():
            if isinstance(value, dict) and isinstance(data.get(name), dict):
                data[name] = {**data[name], **value}
            else:
                data[name] = value
        return RunConfig.model_validate(data)

    @classmethod
    def load(cls, config: RunConfig | str | Path | dict | None) -> RunConfig:
        """A RunConfig from itself, a YAML path, a dict, or ``None`` (defaults)."""
        if config is None:
            return cls()
        if isinstance(config, RunConfig):
            return config
        if isinstance(config, dict):
            return cls.model_validate(config)
        return cls.from_yaml(config)
