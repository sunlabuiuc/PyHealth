"""The text proposers send to an LLM, and how replies are read back."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ..evaluation.guardrails import EVALUATION_CONTRACT
from .analysis import ANALYSIS_CONTRACT
from .codegen import CODE_CONTRACT, TEMPLATE
from .context import ProposalContext


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


FEATURE_STRATEGIES = {
    "explore": (
        "Explore: propose features of a kind nobody has tried yet, for a "
        "clinical mechanism the current inputs do not capture."
    ),
    "refine": (
        "Refine: change the window, threshold or selector of a promising "
        "feature, or of one that just missed acceptance."
    ),
    "crossover": (
        "Crossover: combine what the accepted features capture with a "
        "related mechanism, e.g. a ratio, a trend or a threshold."
    ),
    "contrastive": (
        "Contrast: compare the accepted and the rejected feature groups, and "
        "propose features like the accepted ones and unlike the rejected ones."
    ),
}
"""Instructions for each strategy in a feature cycle."""


CODE_STRATEGIES = {
    "explore": (
        "Explore: write a model unlike any tried so far, e.g. a different way of "
        "reading the record or a different training objective."
    ),
    "refine": (
        "Refine: start from the best code attempt (or the best configuration) and "
        "change one thing the results suggest will help."
    ),
    "crossover": (
        "Crossover: combine what works in the best code attempt with what works in "
        "the best configuration from the search space."
    ),
    "contrastive": (
        "Contrast: compare the best and worst attempts, and write a model that "
        "keeps what the best share and avoids what the worst share."
    ),
}
"""Instructions for each strategy in a code cycle."""


FEATURE_SYSTEM_PROMPT = """\
You are a clinician-scientist and machine-learning engineer improving a \
PyHealth prediction model by adding input features. Each turn you propose ONE \
group of new features, written as JSON specs from a fixed set of operators over \
the patients' event records. The harness compiles them, checks they cannot see \
the future, and accepts the group only if it improves the score consistently \
across folds. You never see patient records or the test split.

Prefer features with a clear clinical mechanism: each needs a one-sentence \
"rationale" and the expected "direction" of its effect on the outcome. Do not \
repeat a feature that was already tried. Reply with a short analysis, then one \
JSON object in a ```json block:
{"features": [SPEC, ...], "rationale": "<why this group should help>", \
"expected": "<what you expect, e.g. score up by about 0.01>", "kind": "clinical"}"""


SYSTEM_PROMPT = """\
You are an expert in machine learning for electronic health records, \
tuning a PyHealth pipeline. Each turn you propose ONE configuration from a \
fixed search space. The harness trains it on the training split and reports \
its score on the validation split. You never see patient records or the test \
split.

Reply with a short analysis, then one JSON object in a ```json block:
{"model": "<model name>", "model_params": {...}, "train_params": {...}, \
"rationale": "<one sentence: why this should help>", \
"expected": "<what you expect to change, e.g. score up, smaller spread \
between folds>", "kind": "<clinical or technical>", \
"based_on": <id of the trial you are changing, or null>}
The rationale and expectation are recorded before the trial runs and \
compared with its result in the run's report. Use only the models and \
arguments listed. Do not repeat a configuration that was already tried. Keep \
the time budget in mind: prefer small changes that are likely to help over \
expensive long shots."""


CODE_SYSTEM_PROMPT = """\
You are an expert in machine learning for electronic health records, \
improving a PyHealth prediction model by writing your own model and training \
code. Each turn you write ONE Python module. The harness runs it in a \
separate process on the training split only, predicts held-out patients \
itself, and scores your model exactly like the models in the search space, \
on the same folds. You never see patient records or the test split.

Reply with a short analysis, then the module in one ```python block, then \
one JSON object in a ```json block:
{"rationale": "<one sentence: why this should help>", \
"expected": "<what you expect to change>", "kind": "<clinical or technical>", \
"based_on": <id of the trial you are changing, or null>}"""


EVALUATION_SYSTEM_PROMPT = """\
You are a clinician-scientist and biostatistician reviewing how a PyHealth \
prediction model is scored. Each turn you propose ONE extra evaluation: a \
Python function scoring held-out predictions, fold by fold, for something \
the main metric misses (calibration, a subgroup, alert yield, net benefit). \
If it passes the harness's checks it becomes a guardrail: no attempt clearly \
worse than the starting model on it can be selected. It never replaces the \
main metric. You never see patient records.

Reply with a short analysis, then the module in one ```python block, then \
one JSON object in a ```json block:
{"name": "<short_identifier>", "metrics": {"<metric>": "higher" or "lower"}, \
"needs": ["<column>", ...], "rationale": "<why it matters clinically>", \
"kind": "<calibration, subgroup, utility or robustness>"}"""


ANALYSIS_SYSTEM_PROMPT = """\
You are a clinician-scientist and data analyst. Each turn you ask ONE \
question of the training data that would guide the next attempts (where \
errors concentrate, how the label varies, what is missing), and write a \
Python function that answers it. It runs on the training split only, and \
returns only aggregate tables with small groups withheld. You never see \
patient records.

Reply with a short analysis, then the module in one ```python block, then \
one JSON object in a ```json block:
{"name": "<short_identifier>", "question": "<the question>", \
"rationale": "<how the answer would change what we try next>"}"""


FINDINGS_PROMPT = """\
An analysis of the training data asked: {question}

Its results (groups under 11 patients withheld):
{results}

State at most 3 findings that should guide the next attempts. Each one \
sentence, quoting only numbers that appear above. Reply with a ```json \
block: {{"findings": ["...", ...]}}"""


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
    parts += _shared_parts(context)
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


def build_feature_prompt(context: ProposalContext) -> str:
    """Builds the user prompt for one feature proposal.

    Examples:
        >>> build_feature_prompt(context)  # doctest: +SKIP
    """
    direction = "higher" if context.maximize else "lower"
    parts = [f"Task:\n{context.task}"]
    if context.card:
        parts.append(f"Task card:\n{context.card}")
    parts += [
        f"Validation metric: {context.metric} ({direction} is better).",
        context.features.describe(),
    ]
    if context.selected is not None:
        parts.append("The model is held fixed during this feature cycle: "
                     f"{context.selected.config.key()}")
    if context.accepted:
        parts.append("Accepted features (your group is added to these):\n" + _bullets(
            f"{s['name']}: {s['definition']}" for s in context.accepted))
    else:
        parts.append("No features have been accepted yet.")
    if context.feature_trials:
        parts.append("Latest feature groups tried:\n" + format_trials(
            context.feature_trials))
    if context.tried_features:
        parts.append("Every feature tried so far (do not propose these again): "
                     + ", ".join(sorted(context.tried_features)))
    if context.lessons:
        parts.append("Lessons from earlier cycles:\n" + _bullets(context.lessons))
    parts.append(f"Strategy for this turn: {FEATURE_STRATEGIES[context.strategy]}")
    parts.append(f"Propose at most {context.features.max_features} features.")
    if context.feedback:
        parts.append(f"Note: {context.feedback}")
    return "\n\n".join(parts)


def _shared_parts(context: ProposalContext) -> list[str]:
    """Analysis findings and guardrails, which every kind of prompt includes."""
    parts = []
    if context.findings:
        parts.append("Findings from analyses of the training data:\n"
                     + _bullets(context.findings))
    if context.evaluations:
        parts.append("Extra evaluations (an active guardrail removes any attempt "
                     "clearly worse than the starting model on it):\n"
                     + _bullets(context.evaluations))
    return parts


def _common_parts(context: ProposalContext, title: str) -> list[str]:
    direction = "higher" if context.maximize else "lower"
    parts = [f"Task:\n{context.task}"]
    if context.task_source:
        parts.append(context.task_source)
    if context.card:
        parts.append(f"Task card:\n{context.card}")
    parts.append(f"Main metric: {context.metric} ({direction} is better).")
    if context.lessons:
        parts.append("Lessons from earlier cycles:\n" + _bullets(context.lessons))
    parts += _shared_parts(context)
    if context.leaderboard:
        parts.append(f"{title}:\n" + format_trials(context.leaderboard))
    if context.selected is not None:
        parts.append(f"Current selection: #{context.selected.trial_id}. "
                     f"{context.selection_rule}".strip())
    return parts


def build_code_prompt(context: ProposalContext) -> str:
    """Builds the user prompt for a code turn.

    Examples:
        >>> from pyhealth.autoimprove import SearchSpace
        >>> context = ProposalContext("a0", "explore", SearchSpace(), "binary task",
        ...                           "pr_auc", True)
        >>> "build_and_train" in build_code_prompt(context)
        True
    """
    parts = _common_parts(context, "Best attempts so far (configs and code)")
    parts.append(CODE_CONTRACT)
    if context.code_examples:
        for line, code in context.code_examples:
            parts.append(f"A code attempt {line}:\n```python\n{code}\n```")
    else:
        parts.append(f"A minimal module that meets the contract:\n```python\n"
                     f"{TEMPLATE}```")
    if context.own_trials:
        parts.append(f"Your own trials ({context.agent}):\n"
                     + format_trials(context.own_trials))
    parts.append(f"Strategy for this turn: {CODE_STRATEGIES[context.strategy]}")
    if context.feedback:
        parts.append(f"Note: {context.feedback}")
    return "\n\n".join(parts)


def build_evaluation_prompt(context: ProposalContext) -> str:
    """Builds the user prompt for an evaluation turn.

    Examples:
        >>> from pyhealth.autoimprove import SearchSpace
        >>> context = ProposalContext("a0", "explore", SearchSpace(), "binary task",
        ...                           "pr_auc", True, columns=["age"])
        >>> "evaluate(y_true, y_prob, info)" in build_evaluation_prompt(context)
        True
    """
    parts = _common_parts(context, "Best attempts so far")
    parts.append(EVALUATION_CONTRACT)
    if context.metrics_menu:
        parts.append(context.metrics_menu)
    parts.append("Columns you can list in \"needs\": "
                 + (", ".join(context.columns) or "(none)"))
    parts.append("Propose an evaluation that is not already listed above, for a "
                 "failure the main metric would miss.")
    if context.feedback:
        parts.append(f"Note: {context.feedback}")
    return "\n\n".join(parts)


def build_analysis_prompt(context: ProposalContext) -> str:
    """Builds the user prompt for an analysis turn.

    Examples:
        >>> from pyhealth.autoimprove import SearchSpace
        >>> context = ProposalContext("a0", "explore", SearchSpace(), "binary task",
        ...                           "pr_auc", True, columns=["age", "label"])
        >>> "analyze(data, out)" in build_analysis_prompt(context)
        True
    """
    parts = _common_parts(context, "Best attempts so far")
    parts.append(ANALYSIS_CONTRACT)
    parts.append("Columns in data: " + (", ".join(context.columns) or "(none)"))
    parts.append("Ask a question whose answer is not already in the findings above.")
    if context.feedback:
        parts.append(f"Note: {context.feedback}")
    return "\n\n".join(parts)


def parse_code(text: str) -> tuple[str, dict[str, Any]]:
    """Extracts the last ```python block and the JSON object from an LLM reply.

    Examples:
        >>> code, meta = parse_code('Plan.\\n```python\\nx = 1\\n```\\n```json\\n{"a": 1}\\n```')
        >>> code, meta
        ('x = 1\\n', {'a': 1})

    Raises:
        ValueError: If the reply holds no python block.
    """
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        raise ValueError("no ```python block found")
    code = blocks[-1]
    rest = re.sub(r"```(?:python|py)\s*\n.*?```", "", text, flags=re.DOTALL)
    try:
        meta = parse_json(rest)
    except ValueError:
        meta = {}
    return code, meta


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
        added = getattr(t, "added", None)
        if added:
            verdict = {True: "accepted", False: "rejected"}.get(
                getattr(t, "accepted", None), "")
            line += f" + features {', '.join(added)} ({verdict})"
        if t.notes:
            line += f" (notes: {'; '.join(t.notes)})"
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
