"""Published evidence in the feature search and the report.

A :class:`~pyhealth.autoimprove.KnowledgeBase` holds verified, mapped
published predictors. This module puts them to use without changing the
improver:

- :class:`EvidenceProposer` is an :class:`~pyhealth.autoimprove.LLMProposer`
  whose feature prompts also list the untried published predictors, with
  their definitions, directions, priors and citations;
- :func:`literature_section` and :func:`add_literature` add a *Literature*
  section to a run's report: the citation-check counts, the gap report and
  the published discrimination values.

The literature only orders and justifies candidates: the LLM may use the
listed predictors or not, and the feature search decides what is kept.

Examples:
    >>> from pyhealth.autoimprove import EvidenceProposer, add_literature
    >>> proposer = EvidenceProposer(ClaudeLLM(), kb)  # doctest: +SKIP
    >>> improver = AutoImprover(train, val, proposer, features=fs)  # doctest: +SKIP
    >>> report = add_literature(improver.report(), kb,  # doctest: +SKIP
    ...                         path="runs/asthma/report.html")
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .knowledge import KnowledgeBase
from .llm import LLM
from .proposers import FEATURE_SYSTEM_PROMPT, LLMProposer, ProposalContext
from .report import Report, Section

# Published effect types on the scale of each test metric.
_METRIC_TYPES = {
    "roc_auc": {"AUC", "AUROC", "ROC-AUC", "C", "C-STATISTIC"},
    "c_index": {"AUC", "AUROC", "ROC-AUC", "C", "C-STATISTIC"},
    "pr_auc": {"PR-AUC", "AUPRC"},
}

_DIRECTIONS = {"+": "higher → more risk", "-": "higher → less risk",
               "u": "U-shaped", "": "ambiguous"}


def evidence_lines(
    knowledge: KnowledgeBase, tried: Iterable[str] = (), k: int = 8
) -> list[str]:
    """Up to ``k`` untried published predictors, best prior first, one line each.

    Each line gives the spec's name, definition, direction, prior and
    citations, then the spec itself as JSON, so an LLM can propose it as is
    or adapt it.

    Args:
        knowledge: A knowledge base, already verified and mapped.
        tried: Names of features already tried; they are left out.
        k: Most lines.

    Examples:
        >>> kb = KnowledgeBase()
        >>> evidence_lines(kb)
        []
    """
    tried = set(tried)
    lines: list[str] = []
    for spec in knowledge.ranked_specs():
        if len(lines) >= k:
            break
        if spec["name"] in tried:
            continue
        cites = ", ".join(spec.get("citations", []))
        body = {key: v for key, v in spec.items()
                if key not in ("name", "definition", "prior")}
        lines.append(f"{spec['name']}: {spec['definition']} (direction "
                     f"{spec.get('direction') or 'ambiguous'}; prior "
                     f"{spec.get('prior', 0):.2f}; {cites}). Spec: "
                     + json.dumps(body))
    return lines


class EvidenceProposer(LLMProposer):
    """An LLM proposer whose feature prompts list untried published predictors.

    Model proposals and lessons are exactly those of
    :class:`~pyhealth.autoimprove.LLMProposer`. For each feature proposal,
    the prompt gains a list of the knowledge base's buildable predictors
    that have not been tried yet (by this run's feature trials, the
    accepted features or this proposer's own earlier proposals; the
    improver passes only the latest feature trials, and the proposer's own
    memory is not saved, so after a resume an earlier predictor may be listed
    again). A
    predictor also counts as tried when one of those specs adapts it, for
    example with another window: it cites the same source and selects the
    same events. Once all are tried, the prompt is the plain one.

    Args:
        llm: Backend, e.g. :class:`~pyhealth.autoimprove.ClaudeLLM`, or any
            callable ``llm(system, prompt) -> str``.
        knowledge: A :class:`~pyhealth.autoimprove.KnowledgeBase`, already
            verified and mapped (:meth:`~pyhealth.autoimprove.KnowledgeBase.map`).
        max_evidence: Most predictors listed per prompt.
        **kwargs: Passed to :class:`~pyhealth.autoimprove.LLMProposer`.

    Examples:
        >>> from pyhealth.autoimprove import EvidenceProposer, KnowledgeBase
        >>> proposer = EvidenceProposer(lambda system, prompt: "", KnowledgeBase())
        >>> proposer.proposes_models
        True
    """

    def __init__(
        self,
        llm: LLM,
        knowledge: KnowledgeBase,
        max_evidence: int = 8,
        **kwargs: Any,
    ):
        super().__init__(self._ask, **kwargs)
        self.backend = llm
        self.knowledge = knowledge
        self.max_evidence = max_evidence
        self.proposed: list[dict[str, Any]] = []
        self._evidence: list[str] = []

    def propose_features(self, context: ProposalContext) -> dict[str, Any]:
        """A feature group, from a prompt that also lists untried evidence."""
        tried = {name for t in context.feature_trials for name in t.added}
        tried |= {s["name"] for s in context.accepted if "name" in s}
        # A predictor also counts as tried when a proposed or accepted spec
        # adapts it (e.g. another window): same citation, same events.
        used = [*self.proposed, *context.accepted]
        tried |= {spec["name"] for spec in self.knowledge.ranked_specs()
                  if any(_adapts(u, spec) for u in used)}
        self._evidence = evidence_lines(self.knowledge, tried, self.max_evidence)
        try:
            proposal = super().propose_features(context)
        finally:
            self._evidence = []
        for spec in proposal.get("features") or []:
            try:
                self.proposed.append(context.features.validate(spec).to_dict())
            except (ValueError, TypeError, KeyError, AttributeError):
                continue  # the improver reports invalid specs
        return proposal

    def _ask(self, system: str, prompt: str) -> str:
        if system == FEATURE_SYSTEM_PROMPT and self._evidence:
            prompt += (
                "\n\nPublished predictors not tried yet (verified citations; the "
                "prior only orders them). Consider them, adapting windows or "
                "thresholds to this task if needed, and keep their citations in "
                "each spec's \"citations\":\n"
                + "\n".join(f"- {line}" for line in self._evidence)
            )
        return self.backend(system, prompt)


def _selects(node: Any) -> set[tuple]:
    """The selections a spec reads, at any depth: event type, group, filters.

    Two labs from one table differ in their group or ``where`` filter, so
    they are different selections.
    """
    out: set[tuple] = set()
    if isinstance(node, dict):
        select = node.get("select")
        if isinstance(select, dict):
            where = tuple(sorted((k, tuple(sorted(map(str, v))))
                                 for k, v in (select.get("where") or {}).items()))
            out.add((select.get("event_type"), select.get("group"), where))
        for value in node.values():
            if isinstance(value, (dict, list)):
                out |= _selects(value)
    elif isinstance(node, list):
        for value in node:
            out |= _selects(value)
    return out


def _adapts(spec: dict[str, Any], listed: dict[str, Any]) -> bool:
    """Whether ``spec`` is ``listed`` or a variant of it.

    A variant cites the same source and reads the same set of selections
    (event type, group and filters), e.g. with another window or threshold.
    A composite spec (a ratio or difference of two selections) is not a
    variant of either part. Two predictors from one paper that read exactly
    the same selections count as one.
    """
    if spec.get("name") and spec.get("name") == listed.get("name"):
        return True
    cites = set(spec.get("citations") or ()) & set(listed.get("citations") or ())
    selects = _selects(listed)
    return bool(cites) and bool(selects) and _selects(spec) == selects


def literature_section(
    knowledge: KnowledgeBase, test: dict[str, Any] | None = None
) -> Section:
    """A report section: citation-check counts, gap report, published values.

    Args:
        knowledge: A knowledge base, already verified (and mapped, for the
            gap report).
        test: The result of :meth:`~pyhealth.autoimprove.AutoImprover.evaluate_on_test`
            for the model this section describes, to put its test score next
            to the published values. The caller is responsible for the match;
            :func:`add_literature` checks it against the report's selection.

    Examples:
        >>> literature_section(KnowledgeBase()).title
        'Literature'
    """
    verified = len(knowledge.records())
    rejected = sum(1 for e in knowledge.evidence if e.verified is False)
    blocks: list[tuple] = [("p", (
        f"{verified} published effects passed the citation check ({rejected} "
        "rejected: a source not gathered, a quote not in the source or not "
        "naming the predictor, or an effect size not attached to the effect "
        "in the quote). The literature orders and justifies candidates; only "
        "this task's data decides. *(measured)*"))]
    rows = []
    for status, items in knowledge.gap_report().items():
        for r in items:
            detail = r.get("column") or r.get("definition") or r.get("note", "")
            rows.append([r["predictor"], status, detail,
                         _DIRECTIONS.get(r.get("direction", ""), r.get("direction")),
                         ", ".join(r.get("sources", [])[:3])])
    if rows:
        blocks += [("p", ("**Gap report**: each published predictor, and whether "
                          "the task has it, can build it, lacks the data (a "
                          "data request), or is not mapped yet.")),
                   ("table", ["predictor", "status", "detail", "expected",
                              "sources"], rows)]
    values = [f"{b['type']} {b['value']:g} ({b['source']}"
              + (f"; {b['population']}" if b.get("population") else "") + ")"
              for b in knowledge.benchmark_range()
              if isinstance(b.get("value"), (int, float))]
    if values:
        text = f"**Published discrimination:** {'; '.join(values)}."
        estimate = ((test or {}).get("best") or {}).get("estimate")
        if estimate is not None:
            metric = test.get("metric", "")
            text += f" This model on test: {estimate:.3f} ({metric})."
            published = {str(b.get("type", "")).upper()
                         for b in knowledge.benchmark_range()}
            family = _METRIC_TYPES.get(metric.lower())
            if family is not None and not published <= family:
                other = ", ".join(sorted(published - family))
                text += (f" Published {other} values are not on the scale of "
                         f"{metric}.")
        blocks.append(("p", text))
    return Section("Literature", blocks)


def add_literature(
    report: Report,
    knowledge: KnowledgeBase,
    test: dict[str, Any] | None = None,
    path: str | None = None,
) -> Report:
    """Appends :func:`literature_section` to a report and returns it.

    The gap report, the published values and the citation-check counts are
    also stored in ``report.data["literature"]``.

    :meth:`~pyhealth.autoimprove.AutoImprover.report` saves its files before
    this runs, so pass ``path`` to write the report again with the section.

    Args:
        report: The run's report.
        knowledge: A knowledge base, already verified and mapped.
        test: The result of :meth:`~pyhealth.autoimprove.AutoImprover.evaluate_on_test`.
            Its score is shown only if it is for the trial the report selected
            (by ``trial_id``), as :func:`~pyhealth.autoimprove.build_report`
            does: after pruning or confirmation the selection can change.
        path: Where to write the report again: ``path`` (HTML) plus the
            ``.md`` and ``.json`` next to it, as
            :meth:`~pyhealth.autoimprove.Report.save` does.

    Examples:
        >>> from pyhealth.autoimprove import Report
        >>> report = add_literature(Report("Run", []), KnowledgeBase())
        >>> [s.title for s in report.sections]
        ['Literature']
    """
    best = (test or {}).get("best")
    selected = report.data.get("selected")
    if test is not None and not (
        isinstance(best, dict) and selected is not None
        and best.get("trial_id") == selected
    ):
        test = None  # not this report's selected trial: show no score
    report.sections.append(literature_section(knowledge, test))
    report.data["literature"] = {
        "gap": knowledge.gap_report(),
        "benchmark": knowledge.benchmark_range(),
        "verified": len(knowledge.records()),
        "rejected": sum(1 for e in knowledge.evidence if e.verified is False),
    }
    if path is not None:
        report.save(path)
    return report
