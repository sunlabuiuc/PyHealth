"""Sources of new pipeline configs: an LLM, or random search as a baseline."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..evaluation.guardrails import EvaluationSpec
from ..utils.llm import LLM
from .analysis import AnalysisSpec
from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    FEATURE_SYSTEM_PROMPT,
    FINDINGS_PROMPT,
    SYSTEM_PROMPT,
    _bullets,
    build_analysis_prompt,
    build_code_prompt,
    build_evaluation_prompt,
    build_feature_prompt,
    build_prompt,
    format_trials,
    parse_code,
    parse_json,
)
from .space import CODE_MODEL, PipelineConfig, SearchSpace


class ProposerExhausted(ValueError):
    """Raised by a proposer with nothing left to propose; the search ends.

    Examples:
        >>> issubclass(ProposerExhausted, ValueError)
        True
    """


class Proposer(ABC):
    """Proposes the next configuration to try.

    Examples:
        >>> class AlwaysMLP(Proposer):
        ...     def propose(self, context):
        ...         return PipelineConfig("MLP", rationale="a fixed choice")
        >>> AlwaysMLP().propose(None).model
        'MLP'
    """

    proposes_models: bool = True
    """Whether the proposer takes model turns; feature-only proposers don't."""

    @abstractmethod
    def propose(self, context: ProposalContext) -> PipelineConfig:
        """Returns one configuration (it is validated by the improver)."""

    def propose_features(self, context: ProposalContext) -> dict[str, Any]:
        """Returns a group of new features for a feature turn.

        Returns:
            ``{"features": [spec, ...], "rationale": str, "expected": str}``;
            each spec is validated by the improver's FeatureSpace.

        Raises:
            ValueError: If the proposer cannot propose features.
        """
        raise ValueError(f"{type(self).__name__} cannot propose features")

    def propose_code(self, context: ProposalContext) -> PipelineConfig:
        """Returns a code config for a code turn (``PipelineConfig.code``).

        Raises:
            ValueError: If the proposer cannot write code.
        """
        raise ValueError(f"{type(self).__name__} cannot propose code")

    def propose_evaluation(self, context: ProposalContext) -> Any:
        """Returns a :class:`~pyhealth.autoimprove.evaluation.guardrails.EvaluationSpec`.

        Raises:
            ValueError: If the proposer cannot propose evaluations.
        """
        raise ValueError(f"{type(self).__name__} cannot propose evaluations")

    def propose_analysis(self, context: ProposalContext) -> Any:
        """Returns an :class:`~pyhealth.autoimprove.search.analysis.AnalysisSpec`.

        Raises:
            ValueError: If the proposer cannot propose analyses.
        """
        raise ValueError(f"{type(self).__name__} cannot propose analyses")

    def summarize_analysis(
        self, spec: Any, results: str, context: ProposalContext
    ) -> list[str]:
        """Short findings from an analysis's results; the default makes none.

        The improver then keeps a plain summary of the results instead.
        """
        return []

    def reflect(
        self,
        trials: Sequence[Any],
        lessons: Sequence[str],
        metric: str,
        space: SearchSpace | None = None,
    ) -> list[str]:
        """Updates the lessons after a cycle. The default keeps them as they are."""
        return list(lessons)


class RandomProposer(Proposer):
    """Random search over the space: a baseline that needs no LLM.

    Args:
        seed: Seed for the random draws.

    Examples:
        >>> from pyhealth.autoimprove import RandomProposer, SearchSpace
        >>> from pyhealth.autoimprove.search.proposers import ProposalContext
        >>> context = ProposalContext("a0", "explore", SearchSpace(), "", "pr_auc", True)
        >>> RandomProposer(seed=0).propose(context).model in SearchSpace().models
        True
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, context: ProposalContext) -> PipelineConfig:
        config = context.space.sample(self.rng)
        for _ in range(20):
            if context.space.validate(config)[0].key() not in context.tried:
                break
            config = context.space.sample(self.rng)
        return config

    def propose_features(self, context: ProposalContext) -> dict[str, Any]:
        """One or two random valid specs, without a clinical rationale."""
        specs = [context.features.sample(self.rng)
                 for _ in range(self.rng.randint(1, 2))]
        return {"features": specs, "rationale": "random sample", "expected": ""}


class QueueProposer(Proposer):
    """Tries a ranked list of feature groups in order: no LLM, no randomness.

    Use it to test hand-written features, or features ranked from the
    literature, with the improver's acceptance, confirmation and pruning.
    It proposes features only; model turns are skipped.

    Args:
        groups: Feature groups in the order to try: each a spec dict or a
            list of spec dicts tried together.

    Examples:
        >>> queue = QueueProposer([{"op": "count", "window": 365,
        ...                         "select": {"event_type": "ed_visits"},
        ...                         "rationale": "prior ED use", "direction": "+"}])
        >>> queue.proposes_models
        False
    """

    proposes_models = False

    def __init__(self, groups: Sequence[dict | Sequence[dict]]):
        self.groups = [list(g) if isinstance(g, (list, tuple)) else [g] for g in groups]
        self.position = 0

    def propose(self, context: ProposalContext) -> PipelineConfig:
        raise ValueError("QueueProposer proposes features only")

    def propose_features(self, context: ProposalContext) -> dict[str, Any]:
        if self.position >= len(self.groups):
            raise ProposerExhausted("the queue is empty")
        group = self.groups[self.position]
        self.position += 1
        rationale = "; ".join(str(s.get("rationale", "")) for s in group)
        return {"features": group, "rationale": f"queued: {rationale}",
                "expected": "score up"}


class LLMProposer(Proposer):
    """Asks an LLM for configurations, with results and lessons as context.

    Each prompt holds the task summary, the search space, the agent's own
    trials, the leaderboard, the worst trials, the lessons learned so far and
    one improvement strategy (:data:`STRATEGIES`). After each cycle,
    :meth:`reflect` asks the LLM to rewrite the lessons from the new results.
    The lessons are passed to every later prompt and can be saved and reused
    across runs, so the improver also improves the guidance it works from.

    Args:
        llm: Backend, e.g. :class:`~pyhealth.autoimprove.ClaudeLLM`, or any
            callable ``llm(system, prompt) -> str``.
        max_attempts: Requests per proposal before giving up on unparseable
            replies.
        max_lessons: Most lessons kept.

    Examples:
        >>> from pyhealth.autoimprove import ClaudeLLM, LLMProposer
        >>> proposer = LLMProposer(ClaudeLLM())  # doctest: +SKIP
    """

    def __init__(self, llm: LLM, max_attempts: int = 3, max_lessons: int = 10):
        self.llm = llm
        self.max_attempts = max_attempts
        self.max_lessons = max_lessons

    def propose(self, context: ProposalContext) -> PipelineConfig:
        prompt = build_prompt(context)
        error = ""
        for _ in range(self.max_attempts):
            reply = self.llm(SYSTEM_PROMPT, prompt + error)
            try:
                return PipelineConfig.from_dict(parse_json(reply))
            except (TypeError, ValueError) as e:
                error = f"\n\nYour last reply could not be used: {e}. Reply again."
        raise ValueError(f"No usable configuration after {self.max_attempts} replies")

    def propose_features(self, context: ProposalContext) -> dict[str, Any]:
        prompt = build_feature_prompt(context)
        error = ""
        for _ in range(self.max_attempts):
            reply = self.llm(FEATURE_SYSTEM_PROMPT, prompt + error)
            try:
                data = parse_json(reply)
                if not isinstance(data.get("features"), list) or not data["features"]:
                    raise ValueError('the reply needs a non-empty "features" list')
                return data
            except ValueError as e:
                error = f"\n\nYour last reply could not be used: {e}. Reply again."
        raise ValueError(f"No usable features after {self.max_attempts} replies")

    def _code_reply(self, system: str, prompt: str, build: Any) -> Any:
        """Asks for a module and its JSON, retrying on replies that don't parse."""
        error = ""
        for _ in range(self.max_attempts):
            reply = self.llm(system, prompt + error)
            try:
                code, meta = parse_code(reply)
                return build(code, meta)
            except (TypeError, ValueError) as e:
                error = f"\n\nYour last reply could not be used: {e}. Reply again."
        raise ValueError(f"No usable module after {self.max_attempts} replies")

    def propose_code(self, context: ProposalContext) -> PipelineConfig:
        def build(code, meta):
            config = PipelineConfig.from_dict({**meta, "model": CODE_MODEL,
                                               "code": code})
            config.model_params, config.train_params = {}, {}
            return config

        return self._code_reply(CODE_SYSTEM_PROMPT, build_code_prompt(context), build)

    def propose_evaluation(self, context: ProposalContext) -> Any:
        return self._code_reply(EVALUATION_SYSTEM_PROMPT,
                                build_evaluation_prompt(context),
                                lambda code, meta: EvaluationSpec.from_dict(meta, code))

    def propose_analysis(self, context: ProposalContext) -> Any:
        return self._code_reply(ANALYSIS_SYSTEM_PROMPT, build_analysis_prompt(context),
                                lambda code, meta: AnalysisSpec.from_dict(meta, code))

    def summarize_analysis(
        self, spec: Any, results: str, context: ProposalContext
    ) -> list[str]:
        prompt = FINDINGS_PROMPT.format(question=spec.question or spec.name,
                                        results=results or "(no results)")
        try:
            data = parse_json(self.llm(ANALYSIS_SYSTEM_PROMPT, prompt))
        except ValueError:
            return []
        found = data.get("findings")
        if not isinstance(found, list):
            return []
        return [str(x).strip()[:300] for x in found if str(x).strip()][:3]

    def reflect(
        self,
        trials: Sequence[Any],
        lessons: Sequence[str],
        metric: str,
        space: SearchSpace | None = None,
    ) -> list[str]:
        available = (
            f"Models available in this search: {', '.join(sorted(space.models))}. "
            "Only state lessons about these models and their settings."
            if space is not None else ""
        )
        prompt = "\n\n".join(filter(None, [
            f"Validation metric: {metric}.",
            available,
            "All trials so far:\n" + format_trials(trials),
            "Current lessons:\n" + (_bullets(lessons) or "(none)"),
            (
                f"Rewrite the lessons: at most {self.max_lessons} short, specific "
                "statements about what helps or hurts on THIS task, supported by "
                "the trials above. Drop lessons the results contradict. Reply "
                'with a ```json block: {"lessons": ["...", ...]}'
            ),
        ]))
        try:
            data = parse_json(self.llm(SYSTEM_PROMPT, prompt))
        except ValueError:
            return list(lessons)
        if not isinstance(data.get("lessons"), list):
            return list(lessons)
        new = [str(x).strip() for x in data["lessons"] if str(x).strip()]
        return new[: self.max_lessons] or list(lessons)


# ---- what a proposer knows --------------------------------------------------
@dataclass
class ProposalContext:
    """What a proposer knows when it makes one proposal.

    Args:
        agent: Name of the agent proposing.
        strategy: A key of :data:`STRATEGIES`.
        space: The search space.
        task: Plain-text summary of the task (schema, sizes, label counts).
        metric: Validation metric being optimised.
        maximize: Whether larger metric values are better.
        own_trials: This agent's earlier trials, oldest first.
        leaderboard: The best trials of all agents, best first.
        worst: The worst finished trials, worst first.
        tried: Keys of every config already tried.
        lessons: Notes carried over from earlier cycles and runs.
        feedback: Why the previous attempt in this turn was rejected, if any.
        selected: The trial currently selected as the result, if any.
        selection_rule: How a trial replaces the selected one, for the prompt.
        kind: ``"model"`` or ``"features"``: what to propose this turn.
        features: The :class:`~pyhealth.autoimprove.FeatureSpace`, in a
            feature turn.
        accepted: The accepted feature specs (dicts with ``name`` and
            ``definition``), which a feature turn adds to.
        feature_trials: The latest feature trials, oldest first.
        tried_features: Names of every feature tried so far, in any trial.
        card: A plain-text summary of the task card, if any.
        task_source: The task's code and the batch layout of its inputs and
            label (:func:`~pyhealth.autoimprove.search.prompts.task_context`), for
            code, evaluation and analysis turns.
        findings: Short findings from earlier analysis turns.
        evaluations: One line per proposed evaluation: its status, metrics
            and, for active guardrails, the trials that fail it.
        code_examples: ``(summary line, code)`` of the best code attempts.
        metrics_menu: The :mod:`pyhealth.metrics` functions for the label
            kind, for evaluation turns.
        columns: Sample columns available to evaluations and analyses.

    Examples:
        >>> from pyhealth.autoimprove import SearchSpace
        >>> context = ProposalContext("agent0", "explore", SearchSpace(),
        ...                           "Inputs: conditions (sequence)", "pr_auc", True)
        >>> context.strategy, context.tried
        ('explore', set())
    """

    agent: str
    strategy: str
    space: SearchSpace
    task: str
    metric: str
    maximize: bool
    own_trials: Sequence[Any] = ()
    leaderboard: Sequence[Any] = ()
    worst: Sequence[Any] = ()
    tried: set[str] = field(default_factory=set)
    lessons: Sequence[str] = ()
    feedback: str = ""
    selected: Any = None
    selection_rule: str = ""
    kind: str = "model"
    features: Any = None
    accepted: Sequence[dict] = ()
    feature_trials: Sequence[Any] = ()
    card: str = ""
    tried_features: Sequence[str] = ()
    task_source: str = ""
    findings: Sequence[str] = ()
    evaluations: Sequence[str] = ()
    code_examples: Sequence[tuple[str, str]] = ()
    metrics_menu: str = ""
    columns: Sequence[str] = ()
