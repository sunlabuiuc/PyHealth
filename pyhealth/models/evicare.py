"""EviCare: deep-model-guided evidence for LLM diagnosis prediction.

Contributor: Kavish Shah (GitHub: thekavishshah)

Paper: Hengyu Zhang, Xuyun Zhang, Pengxiang Zhan, Linhao Luo, Hang Lv,
Yanchao Tan, Shirui Pan, Carl Yang. EviCare: Enhancing Diagnosis Prediction
with Deep Model-Guided Evidence for In-Context Reasoning. KDD 2026.
https://arxiv.org/abs/2604.10455, https://doi.org/10.1145/3770854.3780257

Official code: https://github.com/zhyccc/EviCare (no license file; this is an
independent re-implementation written from the paper; the prompt wording is
paraphrased from the paper's prompt skeleton, Sections 3.3 to 3.5).

Description:
    EviCare predicts the CCS diagnosis categories of a patient's next visit
    with an off-the-shelf LLM, without fine-tuning. A trained deep model
    (the "guide", RETAIN by default) supplies three kinds of evidence that
    are written into the prompt:

    1. Candidate selection: the top-``k`` CCS codes by guide logit. In the
       ``"novel"`` setting, codes already in the patient's history are
       removed first.
    2. Evidential prioritization: the patient's historical CCS codes are
       sorted by guide logit, and each historical ICD code is listed under
       its CCS parent in that order (paper Eqs. 7-8).
    3. Relational evidence: ``G = A^T A`` counts, over training patients
       (all of their admissions), how many patients have both of two CCS
       codes. Each candidate ``c``
       that is new for the patient is linked to the historical code
       ``argmax_h G[h, c]`` (paper Eqs. 9-10).

    The LLM re-ranks the candidates and replies ``Answer: <CCS 1>, ...``.
    :func:`parse_answer` keeps only valid candidates, and
    :func:`merge_ranking` fills the rest in guide order, so a malformed or
    failed reply falls back to the guide's ranking.

    :class:`EviCare` wraps the guide as a :class:`~pyhealth.models.BaseModel`
    (``forward`` is the guide's, so the usual :class:`~pyhealth.trainer.Trainer`
    trains it) and adds :meth:`EviCare.fit_cooccurrence`,
    :meth:`EviCare.predict` and :meth:`EviCare.evaluate`. The LLM is any
    callable ``llm(system, prompt) -> str``, such as
    :class:`~pyhealth.autoimprove.ClaudeLLM` or
    :class:`~pyhealth.autoimprove.OpenAILLM` (which also serves vLLM and
    other OpenAI-compatible endpoints).
"""

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ..datasets import SampleDataset, get_dataloader
from .base_model import BaseModel
from .retain import RETAIN

logger = logging.getLogger(__name__)

Setting = Literal["overall", "novel"]
LLM = Callable[[str, str], str]

SYSTEM_PROMPT = (
    "You are a clinical diagnosis assistant. You re-rank candidate "
    "diagnosis categories (CCS) for a patient's next hospital visit. Base "
    "the ranking on the patient's diagnosis history, the evidence from a "
    "trained predictive model, and your medical knowledge. Treat the model "
    "evidence as supporting clues, not as the answer. End your reply with "
    "one line in the form: Answer: <CCS 1>, <CCS 2>, ..."
)
"""System prompt sent with every EviCare request."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def format_time_gap(days: float) -> str:
    """Formats a number of days as ``"1 year 2 months 3 days ago"``.

    Years count 365 days and months 30 days. Less than one day is
    ``"today"``.

    Args:
        days: Days between a past visit and the predicted visit.

    Returns:
        The phrase used in the prompt.

    Examples:
        >>> format_time_gap(8)
        '8 days ago'
        >>> format_time_gap(400.5)
        '1 year 1 month 5 days ago'
        >>> format_time_gap(0.3)
        'today'
    """
    total = int(days)
    if total < 1:
        return "today"
    years, rest = divmod(total, 365)
    months, rest = divmod(rest, 30)
    parts = []
    for value, unit in ((years, "year"), (months, "month"), (rest, "day")):
        if value:
            parts.append(f"{value} {unit}{'s' if value > 1 else ''}")
    return " ".join(parts) + " ago"


def rank_codes(
    scores: Sequence[float] | torch.Tensor, codes: Sequence[str]
) -> list[str]:
    """Sorts codes by score, highest first (ties keep the vocabulary order).

    Args:
        scores: One score (logit) per code.
        codes: The codes, aligned with ``scores``.

    Returns:
        The codes from highest to lowest score.

    Raises:
        ValueError: If the lengths differ.

    Examples:
        >>> rank_codes([0.1, 2.0, -1.0, 2.0], ["a", "b", "c", "d"])
        ['b', 'd', 'a', 'c']
    """
    values = (
        scores.detach().float().cpu().numpy()
        if isinstance(scores, torch.Tensor)
        else np.asarray(scores, dtype=float)
    ).reshape(-1)
    if len(values) != len(codes):
        raise ValueError(f"Got {len(values)} scores for {len(codes)} codes.")
    order = np.argsort(-values, kind="stable")
    return [codes[i] for i in order]


def select_candidates(
    ranking: Sequence[str],
    history: Iterable[str],
    k: int = 50,
    setting: Setting = "overall",
) -> list[str]:
    """Selects the top-``k`` candidate codes from a guide ranking.

    In the ``"novel"`` setting the patient's historical codes are removed
    before taking the top ``k``, so there are ``k`` novel candidates when
    the vocabulary allows it. In the ``"overall"`` setting recurring codes
    stay.

    Args:
        ranking: All codes, best first (see :func:`rank_codes`).
        history: The patient's historical CCS codes.
        k: The number of candidates. The paper uses 50.
        setting: ``"overall"`` or ``"novel"``.

    Returns:
        The candidates, best first.

    Examples:
        >>> ranking = ["98", "108", "106", "157"]
        >>> select_candidates(ranking, ["98"], k=2)
        ['98', '108']
        >>> select_candidates(ranking, ["98"], k=2, setting="novel")
        ['108', '106']
    """
    return eligible_codes(ranking, history, setting)[:k]


def eligible_codes(
    ranking: Sequence[str], history: Iterable[str], setting: Setting = "overall"
) -> list[str]:
    """Returns the ranking without historical codes in the ``"novel"`` setting.

    Args:
        ranking: All codes, best first.
        history: The patient's historical CCS codes.
        setting: ``"overall"`` (ranking unchanged) or ``"novel"``.

    Returns:
        The codes that may be predicted, best first.

    Examples:
        >>> eligible_codes(["a", "b", "c"], ["b"], setting="novel")
        ['a', 'c']
        >>> eligible_codes(["a", "b", "c"], ["b"])
        ['a', 'b', 'c']
    """
    if setting not in ("overall", "novel"):
        raise ValueError(f"setting must be 'overall' or 'novel', got {setting!r}.")
    if setting == "overall":
        return list(ranking)
    seen = set(history)
    return [code for code in ranking if code not in seen]


def prioritize_history(
    icd_names: Sequence[str],
    ccs_codes: Sequence[str],
    ranking: Sequence[str],
) -> list[tuple[str, list[str]]]:
    """Groups ICD diagnoses under their CCS parent, by guide priority.

    Paper Eqs. 7-8: the CCS codes are sorted by the guide's logit (here:
    their position in ``ranking``; codes missing from it go last), and each
    CCS keeps its ICD diagnoses in their original order, without duplicates.

    Args:
        icd_names: ICD descriptions (or codes) of one or more visits.
        ccs_codes: The CCS parent of each entry of ``icd_names``.
        ranking: All codes, best first.

    Returns:
        ``(ccs_code, [icd_name, ...])`` pairs, highest priority first.

    Examples:
        >>> prioritize_history(
        ...     ["Atrial fibrillation", "Essential hypertension", "Flutter"],
        ...     ["106", "98", "106"],
        ...     ranking=["98", "108", "106"],
        ... )  # doctest: +NORMALIZE_WHITESPACE
        [('98', ['Essential hypertension']),
         ('106', ['Atrial fibrillation', 'Flutter'])]
    """
    if len(icd_names) != len(ccs_codes):
        raise ValueError("icd_names and ccs_codes must have the same length.")
    groups: dict[str, list[str]] = {}
    for name, ccs in zip(icd_names, ccs_codes):
        names = groups.setdefault(ccs, [])
        if name not in names:
            names.append(name)
    position = {code: i for i, code in enumerate(ranking)}
    order = sorted(groups, key=lambda c: position.get(c, len(position)))
    return [(ccs, groups[ccs]) for ccs in order]


# ---------------------------------------------------------------------------
# Relational evidence
# ---------------------------------------------------------------------------


class CooccurrenceGraph:
    """Patient-level CCS co-occurrence counts, ``G = A^T A`` (paper Eq. 9).

    ``A`` is the binary patient-by-CCS matrix of the training patients, so
    ``G[i, j]`` is the number of patients who have both ``i`` and ``j`` (and
    ``G[i, i]`` the number who have ``i``). Fit it on training patients
    only, so that test labels do not leak into the prompt.

    Args:
        codes: The CCS vocabulary, in matrix order.
        counts: The ``len(codes) x len(codes)`` count matrix.

    Examples:
        >>> graph = CooccurrenceGraph.fit([["98", "108"], ["98", "106"], ["98"]])
        >>> graph.count("98", "108"), graph.count("98", "98")
        (1, 3)
        >>> graph.best_partner("108", history=["106", "98"])
        '98'
    """

    def __init__(self, codes: Sequence[str], counts: np.ndarray) -> None:
        counts = np.asarray(counts)
        if counts.shape != (len(codes), len(codes)):
            raise ValueError(
                f"counts has shape {counts.shape}, expected "
                f"({len(codes)}, {len(codes)})."
            )
        self.codes = list(codes)
        self.index = {code: i for i, code in enumerate(self.codes)}
        self.counts = counts

    @classmethod
    def fit(
        cls,
        patients: Iterable[Iterable[str]],
        vocab: Iterable[str] | None = None,
    ) -> "CooccurrenceGraph":
        """Counts co-occurrences over patients.

        Args:
            patients: For each patient, all of their CCS codes over all of
                their admissions (any order, duplicates allowed), for
                example the values of
                :meth:`~pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3.patient_code_sets`.
            vocab: Codes to include even if no patient has them (their rows
                are zero). The vocabulary is the sorted union of ``vocab``
                and the patients' codes.

        Returns:
            The fitted graph.

        Examples:
            >>> CooccurrenceGraph.fit([["a", "b"], ["b", "b"]]).count("b", "b")
            2
            >>> CooccurrenceGraph.fit([["a"]], vocab=["c"]).codes
            ['a', 'c']
        """
        code_sets = [set(codes) for codes in patients]
        vocab = sorted(set(vocab or ()).union(*code_sets))
        index = {code: i for i, code in enumerate(vocab)}
        counts = np.zeros((len(vocab), len(vocab)), dtype=np.int64)
        for codes in code_sets:
            idx = np.fromiter((index[c] for c in codes), dtype=np.int64)
            counts[np.ix_(idx, idx)] += 1
        return cls(vocab, counts)

    def count(self, a: str, b: str) -> int:
        """Returns the number of training patients with both ``a`` and ``b``."""
        i, j = self.index.get(a), self.index.get(b)
        if i is None or j is None:
            return 0
        return int(self.counts[i, j])

    def best_partner(self, candidate: str, history: Sequence[str]) -> str | None:
        """Returns the historical code that co-occurs most with ``candidate``.

        Paper Eq. 10. Ties go to the earlier code in ``history`` (pass it in
        priority order); a candidate that co-occurs with no historical code
        has no partner.

        Args:
            candidate: A candidate CCS code.
            history: The patient's historical CCS codes, highest priority
                first.

        Returns:
            The partner code, or None.
        """
        best, best_count = None, 0
        for code in history:
            if code == candidate:
                continue
            value = self.count(code, candidate)
            if value > best_count:
                best, best_count = code, value
        return best


def relational_evidence(
    candidates: Sequence[str],
    history: Sequence[str],
    graph: CooccurrenceGraph,
) -> list[tuple[str, str]]:
    """Links each new candidate to its most co-occurring historical code.

    Candidates already in the history are skipped (they need no relational
    support), as are candidates that co-occur with no historical code.

    Args:
        candidates: The candidate codes, best first.
        history: The patient's historical CCS codes, highest priority first.
        graph: A :class:`CooccurrenceGraph` fitted on training patients.

    Returns:
        ``(historical_code, candidate)`` pairs, in candidate order.

    Examples:
        >>> graph = CooccurrenceGraph.fit([["98", "108"], ["106", "105"]])
        >>> relational_evidence(["98", "108", "105", "157"], ["98", "106"], graph)
        [('98', '108'), ('106', '105')]
    """
    seen = set(history)
    links = []
    for candidate in candidates:
        if candidate in seen:
            continue
        partner = graph.best_partner(candidate, history)
        if partner is not None:
            links.append((partner, candidate))
    return links


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _quote(text: str) -> str:
    return '"' + str(text).replace('"', "'") + '"'


def _visit_lines(
    icd_names: Sequence[str],
    ccs_codes: Sequence[str],
    ranking: Sequence[str],
    ccs_names: Mapping[str, str],
) -> list[str]:
    lines = []
    for ccs, names in prioritize_history(icd_names, ccs_codes, ranking):
        members = ", ".join(_quote(name) for name in names)
        lines.append(
            f"- {{{members}}} BELONG TO {_quote(ccs_names.get(ccs, ccs))}"
        )
    return lines


def build_prompt(
    history_icd_names: Sequence[Sequence[str]],
    history_ccs: Sequence[Sequence[str]],
    days_ago: Sequence[float],
    candidates: Sequence[str],
    ranking: Sequence[str],
    ccs_names: Mapping[str, str] | None = None,
    setting: Setting = "overall",
    relations: Sequence[tuple[str, str]] = (),
    max_visits: int | None = None,
) -> str:
    """Builds the EviCare user prompt for one prediction.

    The layout follows the paper's prompt skeleton (Section 3.5): the
    visit(s) with their time gap and prioritized diagnoses, the relational
    evidence, the candidate list and the instruction. The ``"novel"``
    setting shows only the last visit; the ``"overall"`` setting shows every
    history visit (or the last ``max_visits``).

    Args:
        history_icd_names: Per history visit, the ICD descriptions.
        history_ccs: Per history visit, the CCS parent of each description.
        days_ago: Per history visit, the days to the predicted visit.
        candidates: The candidate CCS codes, best first.
        ranking: All codes in guide order, used to prioritize the history.
        ccs_names: CCS code -> name. Codes without a name are shown as is.
        setting: ``"overall"`` or ``"novel"``.
        relations: ``(historical_code, candidate)`` pairs from
            :func:`relational_evidence`. The section is left out if empty.
        max_visits: In the ``"overall"`` setting, show only the most recent
            visits. Default None (all).

    Returns:
        The prompt text.

    Examples:
        >>> prompt = build_prompt(
        ...     history_icd_names=[["Atrial fibrillation"]],
        ...     history_ccs=[["106"]],
        ...     days_ago=[8.0],
        ...     candidates=["105", "108"],
        ...     ranking=["105", "108", "106"],
        ...     ccs_names={"105": "Conduction disorders",
        ...                "106": "Cardiac dysrhythmias"},
        ...     setting="novel",
        ...     relations=[("106", "105")],
        ... )
        >>> print(prompt)  # doctest: +NORMALIZE_WHITESPACE
        Task: predict the diagnosis categories that will be NEW at the patient's
        next visit (none of them has been recorded for this patient before).
        <BLANKLINE>
        Evidential Prioritization for Set-based EHRs: diagnoses are grouped
        under their CCS category, and categories are listed from most to least
        relevant according to a trained predictive model.
        <BLANKLINE>
        Last Diagnostic Visit (8 days ago):
        - {"Atrial fibrillation"} BELONG TO "Cardiac dysrhythmias"
        <BLANKLINE>
        Relational Evidence for Novel Diagnoses: each line links a past
        diagnosis to a candidate that often occurs with it in other patients'
        records.
        - "Cardiac dysrhythmias" => "Conduction disorders"
        <BLANKLINE>
        Candidate CCS Codes (Novel Only):
        "Conduction disorders", "108"
        <BLANKLINE>
        Instruction:
        - Re-rank the candidate CCS categories from most to least likely.
        - Use the candidate names exactly as written and add no other categories.
        - Output format: Answer: <CCS 1>, <CCS 2>, ...
    """
    if setting not in ("overall", "novel"):
        raise ValueError(f"setting must be 'overall' or 'novel', got {setting!r}.")
    if not (len(history_icd_names) == len(history_ccs) == len(days_ago)):
        raise ValueError("history fields must have one entry per visit.")
    names = dict(ccs_names or {})
    lines: list[str] = []
    if setting == "novel":
        lines += [
            (
                "Task: predict the diagnosis categories that will be NEW at the "
                "patient's next visit (none of them has been recorded for this "
                "patient before)."
            ),
        ]
    else:
        lines += [
            (
                "Task: predict the diagnosis categories of the patient's next "
                "visit. They may repeat earlier diagnoses or be new."
            ),
        ]
    lines += [
        "",
        (
            "Evidential Prioritization for Set-based EHRs: diagnoses are grouped "
            "under their CCS category, and categories are listed from most to "
            "least relevant according to a trained predictive model."
        ),
        "",
    ]

    n_visits = len(history_ccs)
    if setting == "novel":
        shown = [n_visits - 1] if n_visits else []
    else:
        first = 0 if max_visits is None else max(n_visits - max_visits, 0)
        shown = list(range(first, n_visits))
    for i in shown:
        gap = format_time_gap(days_ago[i])
        if i == n_visits - 1:
            title = f"Last Diagnostic Visit ({gap})"
        else:
            title = f"Visit {i + 1} ({gap})"
        lines.append(f"{title}:")
        lines += _visit_lines(history_icd_names[i], history_ccs[i], ranking, names)
        lines.append("")

    if relations:
        lines.append(
            "Relational Evidence for Novel Diagnoses: each line links a past "
            "diagnosis to a candidate that often occurs with it in other "
            "patients' records."
        )
        for past, candidate in relations:
            lines.append(
                f"- {_quote(names.get(past, past))} => "
                f"{_quote(names.get(candidate, candidate))}"
            )
        lines.append("")

    header = (
        "Candidate CCS Codes (Novel Only):"
        if setting == "novel"
        else "Candidate CCS Codes:"
    )
    lines.append(header)
    lines.append(", ".join(_quote(names.get(c, c)) for c in candidates))
    lines += [
        "",
        "Instruction:",
        "- Re-rank the candidate CCS categories from most to least likely.",
        (
            "- Use the candidate names exactly as written and add no other "
            "categories."
        ),
        "- Output format: Answer: <CCS 1>, <CCS 2>, ...",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?(</think>|$)", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"answer\s*[:：]", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\s*[.):])\s*")
_EDGE = "\"'`“”‘’[]{}<>*"
_PREFIXED = re.compile(r"^([A-Za-z]{0,3}\d+)\s*[:\-–—]\s*(.+)$")
_SUFFIXED = re.compile(r"^(.+?)\s*\(([A-Za-z]{0,3}\d+)\)$")


def _normalize(text: str) -> str:
    text = _BULLET.sub("", text.strip())
    text = text.strip().strip(_EDGE).strip().rstrip(".").strip().strip(_EDGE)
    return " ".join(text.lower().split())


def parse_answer(
    reply: str | None,
    candidates: Sequence[str],
    ccs_names: Mapping[str, str] | None = None,
) -> list[str]:
    """Extracts the ranked candidate codes from an LLM reply.

    Reads the text after the last ``Answer:`` (the whole reply if there is
    none, and never inside a ``<think>`` block), splits it on commas and
    line breaks, and matches each item exactly (ignoring case, quotes,
    brackets, list numbering and a trailing period) to a candidate's name or
    code. Items such as ``"108: Congestive heart failure"`` or
    ``"Congestive heart failure (108)"`` also match. Items that are not
    candidates (hallucinated or junk) are dropped, and repeated codes keep
    their first position. Adjacent items are re-joined when needed, so names
    that contain commas still match.

    Args:
        reply: The LLM reply.
        candidates: The candidate codes.
        ccs_names: CCS code -> name, as shown in the prompt.

    Returns:
        The candidate codes in the LLM's order; possibly empty.

    Examples:
        >>> names = {"108": "Congestive heart failure; nonhypertensive",
        ...          "157": "Acute and unspecified renal failure"}
        >>> parse_answer(
        ...     "Reasoning...\\nAnswer: Acute and unspecified renal failure, "
        ...     "Sepsis, 108, acute and unspecified renal failure",
        ...     candidates=["108", "157"], ccs_names=names)
        ['157', '108']
        >>> parse_answer("I am not sure.", ["108"], names)
        []
    """
    if not reply:
        return []
    names = ccs_names or {}
    lookup: dict[str, str] = {}
    for code in candidates:
        lookup.setdefault(_normalize(names.get(code, code)), code)
    for code in candidates:
        lookup.setdefault(_normalize(code), code)

    def match(item: str) -> str | None:
        text = _normalize(item)
        if not text:
            return None
        if text in lookup:
            return lookup[text]
        for pattern, groups in ((_PREFIXED, (1, 2)), (_SUFFIXED, (2, 1))):
            found = pattern.match(text)
            if found:
                for group in groups:
                    part = _normalize(found.group(group))
                    if part in lookup:
                        return lookup[part]
        return None

    text = _THINK.sub(" ", str(reply))
    answers = list(_ANSWER.finditer(text))
    if answers:
        text = text[answers[-1].end():]
    items = [item for item in re.split(r"[,\n]", text)]

    parsed: list[str] = []
    i = 0
    while i < len(items):
        code, used = None, 1
        for width in range(1, min(4, len(items) - i) + 1):
            code = match(",".join(items[i : i + width]))
            if code is not None:
                used = width
                break
        if code is not None and code not in parsed:
            parsed.append(code)
        i += used
    return parsed


def merge_ranking(parsed: Sequence[str], fallback: Sequence[str]) -> list[str]:
    """Completes an LLM ranking with the guide's order.

    The parsed codes come first, followed by every other code of
    ``fallback`` in its order. An empty ``parsed`` gives ``fallback``.

    Args:
        parsed: Codes from :func:`parse_answer`.
        fallback: The guide's ranking of all eligible codes.

    Returns:
        The final ranking, without duplicates.

    Examples:
        >>> merge_ranking(["c", "a"], ["a", "b", "c", "d"])
        ['c', 'a', 'b', 'd']
    """
    ranking = list(dict.fromkeys(parsed))
    taken = set(ranking)
    ranking += [code for code in dict.fromkeys(fallback) if code not in taken]
    return ranking


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _novel_truth(
    targets: Sequence[Iterable[str]], histories: Sequence[Iterable[str]] | None
) -> list[set[str]]:
    if histories is None:
        return [set(t) for t in targets]
    if len(histories) != len(targets):
        raise ValueError("histories must have one entry per target.")
    return [set(t) - set(h) for t, h in zip(targets, histories)]


def _top_k(prediction: Sequence[str], k: int) -> set[str]:
    return set(list(dict.fromkeys(prediction))[:k])


def precision_at_k_visit(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[Iterable[str]],
    k: int,
    histories: Sequence[Iterable[str]] | None = None,
) -> float:
    """Visit-level precision at ``k`` (paper Appendix A, Eq. 11).

    ``mean over visits of |top_k(pred) & truth| / min(k, |truth|)``. With
    ``histories``, the novel variant: codes seen in the patient's earlier
    visits are removed from the truth, so only new codes count. Visits whose
    truth is empty are skipped.

    Args:
        predictions: Per visit, the ranked predicted codes.
        targets: Per visit, the true codes.
        k: The cut-off.
        histories: Optional, per visit, the codes of earlier visits.

    Returns:
        The score in ``[0, 1]`` (0.0 if no visit is scored).

    Examples:
        >>> predictions = [["a", "b", "c"], ["d", "e", "f"]]
        >>> targets = [["a", "c"], ["e", "x", "y", "z"]]
        >>> precision_at_k_visit(predictions, targets, k=2)  # (1/2 + 1/2) / 2
        0.5
        >>> precision_at_k_visit(predictions, targets, k=2,
        ...                      histories=[["a"], []])  # (0/1 + 1/2) / 2
        0.25
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    scores = []
    for prediction, truth in zip(predictions, _novel_truth(targets, histories)):
        if not truth:
            continue
        scores.append(len(_top_k(prediction, k) & truth) / min(k, len(truth)))
    return float(np.mean(scores)) if scores else 0.0


def accuracy_at_k_code(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[Iterable[str]],
    k: int,
    histories: Sequence[Iterable[str]] | None = None,
) -> float:
    """Code-level accuracy at ``k`` (paper Appendix A, Eq. 12).

    ``sum of |top_k(pred) & truth| / sum of |truth|`` over all visits. With
    ``histories``, the novel variant (see :func:`precision_at_k_visit`).

    Args:
        predictions: Per visit, the ranked predicted codes.
        targets: Per visit, the true codes.
        k: The cut-off.
        histories: Optional, per visit, the codes of earlier visits.

    Returns:
        The score in ``[0, 1]`` (0.0 if there is no true code).

    Examples:
        >>> predictions = [["a", "b", "c"], ["d", "e", "f"]]
        >>> targets = [["a", "c"], ["e", "x", "y", "z"]]
        >>> accuracy_at_k_code(predictions, targets, k=2)  # (1 + 1) / (2 + 4)
        0.3333333333333333
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    hits = total = 0
    for prediction, truth in zip(predictions, _novel_truth(targets, histories)):
        hits += len(_top_k(prediction, k) & truth)
        total += len(truth)
    return hits / total if total else 0.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class EviCarePrediction:
    """The result of :meth:`EviCare.predict` for one sample.

    Attributes:
        patient_id: The patient.
        visit_id: The predicted (target) visit.
        ranking: The final ranking of eligible codes (LLM first, then guide).
        guide_ranking: The guide's ranking of eligible codes.
        candidates: The candidates shown to the LLM.
        targets: The true codes, if the sample has labels.
        history: The patient's historical CCS codes.
        prompt: The user prompt, if an LLM was used.
        reply: The LLM reply, if any.
        n_parsed: How many codes of ``ranking`` came from the LLM.
        error: The LLM error (``"Type: message"``), if the call failed.

    Examples:
        >>> EviCarePrediction("p1", "v2", ["108"], ["108"], ["108"]).n_parsed
        0
    """

    patient_id: Any
    visit_id: Any
    ranking: list[str]
    guide_ranking: list[str]
    candidates: list[str]
    targets: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    prompt: str | None = None
    reply: str | None = None
    n_parsed: int = 0
    error: str | None = None


class EviCare(BaseModel):
    """EviCare: a guide model plus evidence-driven LLM re-ranking.

    Wraps a guide model that scores every CCS label (RETAIN with embedding
    size 16 by default, as in the paper's EviCare(R)). :meth:`forward` is the
    guide's forward pass, so the guide is trained with the usual
    :class:`~pyhealth.trainer.Trainer`. After training:

    1. :meth:`fit_cooccurrence` builds the co-occurrence graph from the
       training patients' code sets (``task.patient_code_sets``).
    2. :meth:`predict` selects candidates, builds the prompt with the
       prioritized history and relational evidence, calls the LLM and
       parses its ranking (falling back to the guide's on any failure).
       Without an LLM it returns the guide's ranking (the deep-model
       baseline).
    3. :meth:`evaluate` computes visit-level P@k and code-level Acc@k.

    The dataset must come from
    :class:`~pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3` or
    :class:`~pyhealth.tasks.EviCareDiagnosisPredictionMIMIC4` (or have the
    same fields), and ``setting`` must match the task's.

    Args:
        dataset: The sample dataset (one ``multilabel`` output).
        guide_model: None (RETAIN), a :class:`~pyhealth.models.BaseModel`
            subclass (built as ``guide_model(dataset, **guide_kwargs)``) or a
            model instance. The guide must return ``"logit"`` with one
            column per label code.
        setting: ``"overall"`` or ``"novel"``.
        k: Number of candidates. Default 50, as in the paper.
        ccs_names: CCS code -> name for the prompt, e.g.
            ``task.ccs_mapping.ccs_names``. Codes without a name are shown as
            codes.
        use_relations: Add relational evidence. Default True.
        max_prompt_visits: In the ``"overall"`` setting, show only the most
            recent visits in the prompt. Default None (all, as in the paper).
        embedding_dim: Embedding size of the default RETAIN guide. Ignored
            when ``guide_model`` is a class or an instance.
        **guide_kwargs: Extra arguments of the guide's constructor. Ignored
            when ``guide_model`` is an already-built instance.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import EviCare
        >>> samples = [
        ...     {"patient_id": "p1", "visit_id": "v2",
        ...      "conditions": [["98"]], "ccs_targets": ["108"],
        ...      "history_icd": [["4019"]],
        ...      "history_icd_names": [["Essential hypertension"]],
        ...      "history_ccs": [["98"]], "days_ago": [8.0]},
        ...     {"patient_id": "p2", "visit_id": "v3",
        ...      "conditions": [["106"]], "ccs_targets": ["98", "106"],
        ...      "history_icd": [["42731"]],
        ...      "history_icd_names": [["Atrial fibrillation"]],
        ...      "history_ccs": [["106"]], "days_ago": [30.0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"ccs_targets": "multilabel"},
        ... )
        >>> model = EviCare(dataset, k=3)
        >>> model.fit_cooccurrence({"p1": ["98", "108"], "p2": ["98", "106"]})
        >>> fake_llm = lambda system, prompt: "Answer: 108, 98"
        >>> predictions = model.predict(dataset, llm=fake_llm)
        >>> predictions[0].ranking[:2]
        ['108', '98']
        >>> sorted(model.evaluate(predictions, ks=(1,)))
        ['Acc@1', 'P@1']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        guide_model: BaseModel | type[BaseModel] | None = None,
        setting: Setting = "overall",
        k: int = 50,
        ccs_names: Mapping[str, str] | None = None,
        use_relations: bool = True,
        max_prompt_visits: int | None = None,
        embedding_dim: int = 16,
        **guide_kwargs: Any,
    ) -> None:
        super().__init__(dataset=dataset)
        if setting not in ("overall", "novel"):
            raise ValueError(f"setting must be 'overall' or 'novel', got {setting!r}.")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}.")
        if len(self.label_keys) != 1 or self.mode != "multilabel":
            raise ValueError("EviCare needs exactly one 'multilabel' output.")
        self.label_key = self.label_keys[0]
        self.setting = setting
        self.k = k
        self.ccs_names = dict(ccs_names or {})
        self.use_relations = use_relations
        self.max_prompt_visits = max_prompt_visits
        vocab = self.label_processor.label_vocab
        self.label_codes = [str(c) for c, _ in sorted(vocab.items(), key=lambda x: x[1])]
        self.cooccurrence: CooccurrenceGraph | None = None

        if guide_model is None:
            guide_model = RETAIN(dataset, embedding_dim=embedding_dim, **guide_kwargs)
        elif isinstance(guide_model, type):
            guide_model = guide_model(dataset, **guide_kwargs)
        if not isinstance(guide_model, nn.Module):
            raise TypeError("guide_model must be a model class or instance.")
        self.guide = guide_model

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Runs the guide model.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`.

        Returns:
            The guide's outputs: ``logit``, ``y_prob`` and, with labels,
            ``loss`` and ``y_true``.
        """
        return self.guide(**kwargs)

    def _decode_labels(self, value: Any) -> list[list[str]]:
        """Turns a batch of multi-hot labels (or code lists) into code lists."""
        if isinstance(value, torch.Tensor):
            rows = value.detach().cpu().numpy()
            return [[self.label_codes[j] for j in np.flatnonzero(row)] for row in rows]
        return [[str(c) for c in codes] for codes in value]

    def fit_cooccurrence(
        self, patient_codes: Mapping[Any, Iterable[str]] | Iterable[Iterable[str]]
    ) -> None:
        """Builds the co-occurrence graph ``G = A^T A`` from training patients.

        ``A`` has one row per training patient with all of their CCS codes,
        over all of their admissions. Build the rows with
        :meth:`~pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3.patient_code_sets`
        (or the MIMIC-IV task's), which reads the patients themselves. The
        samples are not enough: in the ``"novel"`` setting their labels
        drop codes already seen, and patients whose admissions add no new
        code give no sample at all. Use training patients only, so that
        test labels do not leak into the prompt.

        The graph's vocabulary is the guide's label codes plus every code
        of the given patients, so each candidate has a row (all zeros if no
        training patient has it).

        Args:
            patient_codes: Patient ID -> CCS codes, or one code list per
                patient.

        Raises:
            TypeError: If given a sample dataset instead of patient codes.

        Examples:
            >>> model.fit_cooccurrence(  # doctest: +SKIP
            ...     task.patient_code_sets(base_dataset, patient_ids=train_ids)
            ... )
        """
        if isinstance(patient_codes, (SampleDataset, str)):
            raise TypeError(
                "fit_cooccurrence takes patient-level code sets, not samples: "
                "pass task.patient_code_sets(base_dataset, patient_ids=...) "
                "for the training patients."
            )
        if isinstance(patient_codes, Mapping):
            patient_codes = patient_codes.values()
        self.cooccurrence = CooccurrenceGraph.fit(
            patient_codes, vocab=self.label_codes
        )

    @torch.no_grad()
    def predict(
        self,
        dataset: SampleDataset,
        llm: LLM | None = None,
        batch_size: int = 64,
    ) -> list[EviCarePrediction]:
        """Predicts a ranking of CCS codes for every sample.

        Args:
            dataset: The samples to predict (for example the test split).
            llm: Any ``llm(system, prompt) -> str`` callable. None returns
                the guide's ranking (with the ``"novel"`` filter applied).
            batch_size: Batch size of the guide model.

        Returns:
            One :class:`EviCarePrediction` per sample, in dataset order.

        Raises:
            ValueError: If ``llm`` is given, relational evidence is on and
                :meth:`fit_cooccurrence` has not been called (the guide-only
                path never uses relations), or if the guide's logits do not
                match the label vocabulary.
        """
        if llm is not None and self.use_relations and self.cooccurrence is None:
            raise ValueError(
                "Call fit_cooccurrence(task.patient_code_sets(...)) before "
                "predict, or set use_relations=False."
            )
        self.eval()
        results: list[EviCarePrediction] = []
        for batch in get_dataloader(dataset, batch_size=batch_size, shuffle=False):
            logits = self.guide(**batch)["logit"]
            if logits.shape[-1] != len(self.label_codes):
                raise ValueError(
                    f"The guide returns {logits.shape[-1]} logits for "
                    f"{len(self.label_codes)} label codes."
                )
            targets = (
                self._decode_labels(batch[self.label_key])
                if self.label_key in batch
                else [[] for _ in range(len(logits))]
            )
            for i in range(len(logits)):
                results.append(
                    self._predict_one(
                        batch, i, rank_codes(logits[i], self.label_codes),
                        targets[i], llm,
                    )
                )
        return results

    def _predict_one(
        self,
        batch: Mapping[str, Any],
        i: int,
        ranking: list[str],
        targets: list[str],
        llm: LLM | None,
    ) -> EviCarePrediction:
        """Builds the evidence and asks the LLM for one sample of a batch."""
        history_ccs = batch["history_ccs"][i]
        history = list(dict.fromkeys(c for visit in history_ccs for c in visit))
        fallback = eligible_codes(ranking, history, self.setting)
        candidates = fallback[: self.k]
        result = EviCarePrediction(
            patient_id=batch["patient_id"][i],
            visit_id=batch["visit_id"][i] if "visit_id" in batch else None,
            ranking=list(fallback),
            guide_ranking=list(fallback),
            candidates=list(candidates),
            targets=list(targets),
            history=history,
        )
        if llm is None:
            return result

        position = {code: j for j, code in enumerate(ranking)}
        prioritized = sorted(history, key=lambda c: position.get(c, len(position)))
        relations = (
            relational_evidence(candidates, prioritized, self.cooccurrence)
            if self.use_relations
            else []
        )
        result.prompt = build_prompt(
            history_icd_names=batch["history_icd_names"][i],
            history_ccs=history_ccs,
            days_ago=batch["days_ago"][i],
            candidates=candidates,
            ranking=ranking,
            ccs_names=self.ccs_names,
            setting=self.setting,
            relations=relations,
            max_visits=self.max_prompt_visits,
        )
        try:
            result.reply = llm(SYSTEM_PROMPT, result.prompt)
        except Exception as error:  # noqa: BLE001 - any failure falls back
            result.error = f"{type(error).__name__}: {error}"
            logger.warning("EviCare LLM call failed, using the guide: %s", result.error)
            return result
        parsed = parse_answer(result.reply, candidates, self.ccs_names)
        result.n_parsed = len(parsed)
        result.ranking = merge_ranking(parsed, fallback)
        return result

    @staticmethod
    def evaluate(
        predictions: Sequence[EviCarePrediction], ks: Iterable[int] = (10, 20)
    ) -> dict[str, float]:
        """Computes ``P@k`` and ``Acc@k`` of predictions.

        The targets are the task's labels, so in the ``"novel"`` setting
        they are already the novel codes.

        Args:
            predictions: The output of :meth:`predict`.
            ks: The cut-offs. The paper uses 5 and 10 (novel) and 10 and 20
                (overall).

        Returns:
            ``{"P@k": ..., "Acc@k": ...}`` for each ``k``.
        """
        rankings = [p.ranking for p in predictions]
        targets = [p.targets for p in predictions]
        scores = {}
        for k in ks:
            scores[f"P@{k}"] = precision_at_k_visit(rankings, targets, k)
            scores[f"Acc@{k}"] = accuracy_at_k_code(rankings, targets, k)
        return scores
