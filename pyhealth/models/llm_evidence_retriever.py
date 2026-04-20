"""Zero-shot LLM-based evidence retriever for EHR clinical notes.

Paper:
    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Paper link:
    https://proceedings.mlr.press/v248/ahsan24a.html

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import json
import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from pyhealth.models import BaseModel
from pyhealth.tasks.evidence_retrieval_mimic3 import split_sentences

logger = logging.getLogger(__name__)


LLMCallable = Callable[[str], str]


@dataclass
class EvidenceSnippet:
    """Structured evidence snippet returned by the retriever.

    Attributes:
        note_id (str): Identifier of the source note.
        condition (str): Diagnosis condition the snippet addresses.
        decision (str): ``"yes"``, ``"no"``, or ``"uncertain"``.
        role (str): ``"risk"`` or ``"sign"`` evidence category.
        explanation (str): Natural-language summary of the snippet.
        source_sentence (Optional[str]): Extractive sentence from the
            note that best supports ``explanation`` (when available).
        confidence (Optional[float]): Optional log-likelihood-style
            confidence in ``[0, 1]`` when the backend exposes it.
        is_generated (bool): ``True`` when ``explanation`` was generated
            by the LLM rather than extracted verbatim.
    """

    note_id: str
    condition: str
    decision: str
    role: str
    explanation: str
    source_sentence: Optional[str] = None
    confidence: Optional[float] = None
    is_generated: bool = True


@dataclass
class LLMRetrieverConfig:
    """Runtime configuration for the LLM evidence retriever.

    Attributes:
        model_name (str): Free-form backend identifier. Not validated
            here; downstream backends can use it as a dispatch key.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens to request per LLM call.
        prompt_style (str): ``"sequential"`` or ``"single"``. Sequential
            runs note-classification first and only generates an
            explanation when the decision is affirmative; single emits
            both decision and explanation in one prompt.
        max_note_chars (int): Hard cap on note text length passed to
            the LLM. Mitigates cost/latency of long notes.
        use_log_probs (bool): When ``True``, backends that return
            likelihood metadata are queried for confidence scoring.
        cache_size (int): Size of the in-memory prompt->response cache
            shared across :meth:`LLMEvidenceRetriever.retrieve_evidence`
            calls. Set to ``0`` to disable caching. Defaults to
            ``1024``.
    """

    model_name: str = "stub"
    temperature: float = 0.0
    max_tokens: int = 256
    prompt_style: str = "sequential"
    max_note_chars: int = 4000
    use_log_probs: bool = False
    cache_size: int = 1024


class StubLLMBackend:
    """Deterministic, dependency-free LLM backend used for tests.

    The stub inspects the incoming prompt and returns a small JSON
    object that mimics the contract expected from a hosted API. It
    never performs any network I/O, which keeps the PyHealth test
    suite fast and reproducible.
    """

    def __init__(self, keyword_map: Optional[Dict[str, List[str]]] = None) -> None:
        """Initialize the stub backend.

        Args:
            keyword_map (Optional[Dict[str, List[str]]]): Condition to
                keyword list used to decide yes/no. When ``None`` a
                minimal default is used.
        """
        self.keyword_map = keyword_map or {
            "intracranial hemorrhage": [
                "intracranial hemorrhage",
                "intraparenchymal",
                "subdural",
                "hyperdense",
                "craniotomy",
            ],
            "stroke": [
                "stroke",
                "mca infarct",
                "restricted diffusion",
                "middle cerebral artery",
            ],
            "pneumonia": [
                "pneumonia",
                "consolidation",
                "productive cough",
            ],
        }

    def __call__(self, prompt: str) -> str:
        """Return a JSON response that matches the retriever's parser.

        Args:
            prompt (str): The full prompt, containing both the note and
                the question being asked.

        Returns:
            str: JSON-encoded response. Always valid JSON.
        """
        note = (_extract_between(prompt, "Note:\n", "\n\n") or "").lower()
        condition = _extract_between(prompt, "condition:", "\n")
        condition = (condition or "").strip().lower()
        keywords = self.keyword_map.get(condition, [condition])

        decision = "no"
        matched_keyword: Optional[str] = None
        for kw in keywords:
            if kw and kw in note:
                decision = "yes"
                matched_keyword = kw
                break

        role = "risk" if "risk" in note and decision == "yes" else "sign"
        explanation = ""
        if decision == "yes":
            explanation = (
                f"Note references {matched_keyword}, supporting a {role} "
                f"indicator for {condition}."
            )

        return json.dumps(
            {
                "decision": decision,
                "role": role,
                "explanation": explanation,
                "confidence": 0.9 if decision == "yes" else 0.7,
            }
        )


class LLMEvidenceRetriever(BaseModel):
    """LLM-based evidence retriever for EHR clinical notes.

    This model implements the paper's two-pass prompting strategy. A
    first prompt asks the backend a strict yes/no question
    ("Is the patient at risk of X?" or "Does the patient have X?").
    A second prompt is issued only on an affirmative first pass and
    asks the backend for a short natural-language explanation.

    A single-prompt ablation style is also supported to reproduce the
    paper's finding that sequential prompting reduces false positives.

    Args:
        dataset: The :class:`~pyhealth.datasets.SampleDataset` produced
            by :class:`~pyhealth.tasks.EvidenceRetrievalMIMIC3`.
        backend (LLMCallable): Callable accepting a prompt string and
            returning a model response string. Defaults to a local
            deterministic stub that keeps unit tests dependency-free.
            Replace with a hosted API wrapper to run against real
            LLMs.
        config (LLMRetrieverConfig): Runtime configuration controlling
            prompt style, temperature, token budgets, and confidence
            logging.

    Expected inputs (forward):
        - ``note_text``: iterable of raw note strings, one per batch
          element. The text is not tensorised; the model operates on
          strings directly.
        - ``condition``: iterable of condition query strings, one per
          batch element.
        - ``is_positive`` (optional): iterable of binary labels.
          Required for loss computation.

    Examples:
        >>> from pyhealth.datasets import SyntheticEHRNotesDataset
        >>> from pyhealth.tasks import EvidenceRetrievalMIMIC3
        >>> from pyhealth.models import LLMEvidenceRetriever, LLMRetrieverConfig
        >>> dataset = SyntheticEHRNotesDataset(root="./synthetic_notes")
        >>> task = EvidenceRetrievalMIMIC3()
        >>> samples = dataset.set_task(task)
        >>> model = LLMEvidenceRetriever(
        ...     dataset=samples,
        ...     config=LLMRetrieverConfig(prompt_style="sequential"),
        ... )
        >>> snippets = model.retrieve_evidence(
        ...     note_text="patient is on warfarin post craniotomy",
        ...     condition="intracranial hemorrhage",
        ...     note_id="n0001",
        ... )
    """

    def __init__(
        self,
        dataset: Any = None,
        backend: Optional[LLMCallable] = None,
        config: Optional[LLMRetrieverConfig] = None,
    ) -> None:
        """Initialize the LLM evidence retriever."""
        super().__init__(dataset=dataset)
        self.backend: LLMCallable = backend or StubLLMBackend()
        self.config: LLMRetrieverConfig = config or LLMRetrieverConfig()
        self.mode = "binary"
        self._prompt_cache: "OrderedDict[str, str]" = OrderedDict()

    def _cached_call(self, prompt: str) -> str:
        """Invoke the backend with a bounded, FIFO prompt cache.

        Args:
            prompt (str): The prompt to send to the backend.

        Returns:
            str: Backend response. Repeated calls with the same prompt
                hit the in-memory cache until ``config.cache_size`` is
                exceeded.
        """
        if self.config.cache_size <= 0:
            return self.backend(prompt)
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            self._prompt_cache.move_to_end(prompt)
            return cached
        response = self.backend(prompt)
        self._prompt_cache[prompt] = response
        while len(self._prompt_cache) > self.config.cache_size:
            self._prompt_cache.popitem(last=False)
        return response

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_classification_prompt(self, note_text: str, condition: str) -> str:
        """Build the Pass-1 yes/no classification prompt.

        Args:
            note_text (str): Note text, truncated to the configured
                character budget.
            condition (str): Target diagnosis condition.

        Returns:
            str: The prompt to send to the backend.
        """
        return (
            "You are a clinical assistant answering questions from a "
            "patient note.\n"
            f"Note:\n{note_text}\n\n"
            f"Condition: {condition}\n"
            "Question: Is the patient at risk of, or do they have, the "
            "above condition based strictly on the note? Respond as a "
            'JSON object with keys "decision" (yes/no), "role" '
            '(risk/sign), "explanation" (empty string for this pass), '
            'and "confidence" (0-1).'
        )

    def _build_explanation_prompt(self, note_text: str, condition: str) -> str:
        """Build the Pass-2 explanation prompt.

        Args:
            note_text (str): Note text, truncated to the configured
                character budget.
            condition (str): Target diagnosis condition.

        Returns:
            str: The prompt to send to the backend.
        """
        return (
            "You are a clinical assistant summarising evidence from a "
            "patient note.\n"
            f"Note:\n{note_text}\n\n"
            f"Condition: {condition}\n"
            "Produce a short natural-language explanation of the risk "
            "factor or sign that supports the condition, grounded in "
            'the note. Return JSON with keys "decision" (yes), "role" '
            '(risk/sign), "explanation", and "confidence" (0-1).'
        )

    def _build_single_prompt(self, note_text: str, condition: str) -> str:
        """Build the single-prompt ablation prompt.

        Args:
            note_text (str): Note text, truncated to the configured
                character budget.
            condition (str): Target diagnosis condition.

        Returns:
            str: The prompt to send to the backend.
        """
        return (
            "You are a clinical assistant answering questions from a "
            "patient note.\n"
            f"Note:\n{note_text}\n\n"
            f"Condition: {condition}\n"
            "Answer both whether the patient is at risk of or has the "
            "condition and give a short explanation in one response. "
            'Return JSON with keys "decision" (yes/no), "role" '
            '(risk/sign), "explanation", and "confidence" (0-1).'
        )

    # ------------------------------------------------------------------
    # Core retrieval API
    # ------------------------------------------------------------------
    def retrieve_evidence(
        self,
        note_text: str,
        condition: str,
        note_id: str = "",
    ) -> EvidenceSnippet:
        """Run the two-pass or single-pass retrieval on one note.

        Args:
            note_text (str): Raw note text.
            condition (str): Target diagnosis condition.
            note_id (str): Source note identifier to thread through the
                returned snippet.

        Returns:
            EvidenceSnippet: Structured snippet capturing the note id,
                condition, decision, role, explanation, source
                sentence, and confidence.
        """
        truncated = (note_text or "")[: self.config.max_note_chars]

        if self.config.prompt_style == "single":
            raw = self._cached_call(self._build_single_prompt(truncated, condition))
            parsed = _parse_json_safely(raw)
            decision = str(parsed.get("decision", "no")).lower()
            explanation = parsed.get("explanation", "") or ""
            role = str(parsed.get("role", "sign")).lower()
            confidence = _coerce_float(parsed.get("confidence"))
        else:
            raw1 = self._cached_call(
                self._build_classification_prompt(truncated, condition)
            )
            parsed1 = _parse_json_safely(raw1)
            decision = str(parsed1.get("decision", "no")).lower()
            role = str(parsed1.get("role", "sign")).lower()
            confidence = _coerce_float(parsed1.get("confidence"))
            explanation = ""
            if decision == "yes":
                explain_prompt = self._build_explanation_prompt(truncated, condition)
                raw2 = self._cached_call(explain_prompt)
                parsed2 = _parse_json_safely(raw2)
                explanation = parsed2.get("explanation", "") or ""
                explanation_confidence = _coerce_float(parsed2.get("confidence"))
                if explanation_confidence is not None:
                    confidence = (
                        explanation_confidence
                        if confidence is None
                        else min(confidence, explanation_confidence)
                    )

        source_sentence = _pick_source_sentence(truncated, explanation, condition)

        return EvidenceSnippet(
            note_id=note_id,
            condition=condition,
            decision=decision,
            role=role,
            explanation=explanation,
            source_sentence=source_sentence,
            confidence=confidence,
            is_generated=bool(explanation),
        )

    def retrieve_evidence_batch(
        self,
        notes: List[str],
        conditions: List[str],
        note_ids: Optional[List[str]] = None,
    ) -> List[EvidenceSnippet]:
        """Vectorized wrapper around :meth:`retrieve_evidence`.

        Args:
            notes (List[str]): One note per element.
            conditions (List[str]): One condition per element.
            note_ids (Optional[List[str]]): Optional note identifiers,
                defaulting to empty strings.

        Returns:
            List[EvidenceSnippet]: Retrieved snippets, in order.
        """
        if len(notes) != len(conditions):
            raise ValueError("notes and conditions must have equal length.")
        ids = note_ids or [""] * len(notes)
        return [
            self.retrieve_evidence(n, c, i)
            for n, c, i in zip(notes, conditions, ids)
        ]

    # ------------------------------------------------------------------
    # PyHealth forward API — maps yes/no decision to binary logits so
    # the model can be used with the standard trainer for the Pass-1
    # sub-task.
    # ------------------------------------------------------------------
    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Run the retriever and return PyHealth-style outputs.

        Args:
            **kwargs: Must contain ``note_text`` and ``condition`` as
                sequences of strings with equal length. When
                ``is_positive`` is provided, a binary cross-entropy
                loss is computed against it.

        Returns:
            Dict[str, torch.Tensor]: Keys ``logit``, ``y_prob``, and
            optionally ``loss`` and ``y_true``.
        """
        note_text = kwargs.get("note_text", [])
        condition = kwargs.get("condition", [])
        note_text = list(_flatten_text(note_text))
        condition = list(_flatten_text(condition))
        if len(note_text) != len(condition):
            raise ValueError(
                "note_text and condition must have the same batch size."
            )

        ids = kwargs.get("note_id", [""] * len(note_text))
        ids = list(_flatten_text(ids))
        if len(ids) != len(note_text):
            ids = [""] * len(note_text)

        snippets = self.retrieve_evidence_batch(note_text, condition, ids)
        probs = torch.tensor(
            [_decision_to_prob(s) for s in snippets],
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(-1)
        logits = torch.log(probs.clamp(min=1e-6) / (1.0 - probs).clamp(min=1e-6))

        outputs: Dict[str, torch.Tensor] = {"logit": logits, "y_prob": probs}
        outputs["snippets"] = snippets  # type: ignore[assignment]

        y_true_raw = kwargs.get("is_positive")
        if y_true_raw is not None:
            y_true = _coerce_binary_tensor(y_true_raw, self.device).unsqueeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_true
            )
            outputs["y_true"] = y_true
            outputs["loss"] = loss
        return outputs


# ----------------------------------------------------------------------
# Small, self-contained helpers kept private to this module.
# ----------------------------------------------------------------------
def _parse_json_safely(text: str) -> Dict[str, Any]:
    """Best-effort JSON parse that tolerates chatter around the payload.

    Args:
        text (str): Raw backend response.

    Returns:
        Dict[str, Any]: Parsed object, or an empty dict on failure.
    """
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_between(text: str, start: str, end: str) -> Optional[str]:
    """Return the substring between ``start`` and ``end``, if present."""
    lowered = text.lower()
    start_idx = lowered.find(start.lower())
    if start_idx < 0:
        return None
    start_idx += len(start)
    end_idx = lowered.find(end.lower(), start_idx)
    if end_idx < 0:
        return text[start_idx:]
    return text[start_idx:end_idx]


def _coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` as ``float`` when convertible, else ``None``."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _pick_source_sentence(
    note_text: str, explanation: str, condition: str
) -> Optional[str]:
    """Pick the note sentence most likely to support ``explanation``."""
    sentences = split_sentences(note_text)
    if not sentences:
        return None
    target = (explanation or condition or "").lower()
    if not target:
        return sentences[0]
    best = sentences[0]
    best_score = -1
    target_tokens = {t for t in re.findall(r"[a-zA-Z]+", target) if len(t) > 3}
    for sentence in sentences:
        tokens = {t.lower() for t in re.findall(r"[a-zA-Z]+", sentence) if len(t) > 3}
        score = len(target_tokens & tokens)
        if score > best_score:
            best_score = score
            best = sentence
    return best


def _decision_to_prob(snippet: EvidenceSnippet) -> float:
    """Map a snippet decision+confidence to a probability in ``[0, 1]``."""
    if snippet.decision == "yes":
        base = snippet.confidence if snippet.confidence is not None else 0.9
        return max(0.5, min(1.0, base))
    if snippet.decision == "uncertain":
        return 0.5
    base = snippet.confidence if snippet.confidence is not None else 0.9
    return max(0.0, min(0.5, 1.0 - base))


def _flatten_text(value: Any) -> List[str]:
    """Normalize ``value`` into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, torch.Tensor):
        return [str(v) for v in value.tolist()]
    return [str(value)]


def _coerce_binary_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert ``value`` (list/tensor/sequence) to a float binary tensor."""
    if isinstance(value, torch.Tensor):
        return value.float().to(device)
    if isinstance(value, (list, tuple)):
        return torch.tensor([float(v) for v in value], dtype=torch.float, device=device)
    return torch.tensor([float(value)], dtype=torch.float, device=device)
