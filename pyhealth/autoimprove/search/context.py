"""What a proposer knows when it makes one proposal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .space import SearchSpace


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
            label (:func:`~pyhealth.autoimprove.search.codegen.task_context`), for
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
