"""Sources of new pipeline configs: an LLM, or random search as a baseline."""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM
from .space import PipelineConfig, SearchSpace

STRATEGIES = {
    "explore": (
        "Explore: propose a configuration that differs clearly from everything "
        "tried so far, for example a model family nobody has tried yet."
    ),
    "refine": (
        "Refine: start from your own best configuration and change one or two "
        "settings that the results suggest will help."
    ),
    "crossover": (
        "Crossover: combine what works in your best configuration with what "
        "works in the best configuration from another agent."
    ),
    "contrastive": (
        "Contrast: compare the best and the worst configurations so far, and "
        "propose one that keeps what the best ones share and avoids what the "
        "worst ones share."
    ),
}
"""Instructions for each improvement strategy, adapted from codegen-classroom."""


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


class Proposer(ABC):
    """Proposes the next configuration to try.

    Examples:
        >>> class AlwaysMLP(Proposer):
        ...     def propose(self, context):
        ...         return PipelineConfig("MLP", rationale="a fixed choice")
        >>> AlwaysMLP().propose(None).model
        'MLP'
    """

    @abstractmethod
    def propose(self, context: ProposalContext) -> PipelineConfig:
        """Returns one configuration (it is validated by the improver)."""

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
        >>> from pyhealth.autoimprove.proposers import ProposalContext
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


SYSTEM_PROMPT = """\
You are an expert in machine learning for electronic health records, \
tuning a PyHealth pipeline. Each turn you propose ONE configuration from a \
fixed search space. The harness trains it on the training split and reports \
its score on the validation split. You never see patient records or the test \
split.

Reply with a short analysis, then one JSON object in a ```json block:
{"model": "<model name>", "model_params": {...}, "train_params": {...}, \
"rationale": "<one sentence>"}
Use only the models and arguments listed. Do not repeat a configuration that \
was already tried. Keep the time budget in mind: prefer small changes that \
are likely to help over expensive long shots."""


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


def build_prompt(context: ProposalContext) -> str:
    """Builds the user prompt for one proposal.

    Examples:
        >>> from pyhealth.autoimprove import SearchSpace
        >>> context = ProposalContext("a0", "explore", SearchSpace(), "binary task",
        ...                           "pr_auc", True)
        >>> "Strategy for this turn" in build_prompt(context)
        True
    """
    direction = "higher" if context.maximize else "lower"
    parts = [
        f"Task:\n{context.task}",
        context.space.describe(),
        f"Validation metric: {context.metric} ({direction} is better).",
    ]
    if context.lessons:
        parts.append("Lessons from earlier cycles:\n" + _bullets(context.lessons))
    if context.leaderboard:
        parts.append("Best trials so far:\n" + format_trials(context.leaderboard))
    if context.worst:
        parts.append("Worst trials so far:\n" + format_trials(context.worst))
    if context.selected is not None:
        parts.append(
            f"Current selection: #{context.selected.trial_id}. "
            f"{context.selection_rule}".strip()
        )
    if context.own_trials:
        parts.append(
            f"Your own trials ({context.agent}):\n" + format_trials(context.own_trials)
        )
    parts.append(f"Strategy for this turn: {STRATEGIES[context.strategy]}")
    if context.feedback:
        parts.append(f"Note: {context.feedback}")
    return "\n\n".join(parts)


def format_trials(trials: Sequence[Any]) -> str:
    """One line per trial: score, config, time and any error.

    An error is shown by its exception type and the line that raised it
    (``Trial.error_brief``), never its message, which can quote a raw value.

    Examples:
        >>> from pyhealth.autoimprove import PipelineConfig, Trial
        >>> trial = Trial(3, 0, "agent0", "explore", PipelineConfig("MLP"),
        ...               metric_name="pr_auc", score=0.41, seconds=12)
        >>> print(format_trials([trial]))
        - #3 [pr_auc=0.4100, 12s] {"model": "MLP", "model_params": {}, "train_params": {}}
    """
    lines = []
    for t in trials:
        if t.error:
            # trials saved before error_brief existed: the exception type only
            brief = getattr(t, "error_brief", None) or t.error.split(":")[0]
            result = f"error: {brief}"
        else:
            result = f"{t.metric_name}={t.score:.4f}"
            spread = t.metrics.get(f"{t.metric_name}_sd")
            if spread is not None:
                result += f" (sd {spread:.4f})"
        line = f"- #{t.trial_id} [{result}, {t.seconds:.0f}s] {t.config.key()}"
        if t.notes:
            line += f" (changed to fit the search space: {'; '.join(t.notes)})"
        lines.append(line)
    return "\n".join(lines)


def parse_json(text: str) -> dict[str, Any]:
    """Extracts the last JSON object from an LLM reply.

    Examples:
        >>> parse_json('Try a GRU.\\n```json\\n{"model": "RNN"}\\n```')
        {'model': 'RNN'}

    Raises:
        ValueError: If the reply holds no JSON object.
    """
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = blocks or re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("no JSON object found")


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {x}" for x in items)
