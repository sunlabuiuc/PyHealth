"""MedLingo jargon expansion task (plain-language answer from a prompt).

Tied to *Diagnosing our datasets* (Jia, Sontag & Agrawal, CHIL 2025;
https://arxiv.org/abs/2505.15024). This task is a **multiclass shortcut** over
the string ``answer`` column; it does not reproduce the paper's open-ended
generation plus LLM-as-judge setup.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from ..data import Event, Patient
from .base_task import BaseTask

ShotMode = Literal["zero_shot", "one_shot"]


def _as_str(value: Any) -> Optional[str]:
    """Return a clean string or None if the value is unusable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


class MedLingoJargonExpansionTask(BaseTask):
    """Map each MedLingo row to a text prompt and a plain-language ``answer``.

    Ablation (``shot_mode``), aligned with the course rubric:

    - **one_shot**: Use the ``question`` field verbatim as ``prompt``. This
      matches the **released** MedLingo item (including any in-context demo
      baked into that string).
    - **zero_shot**: Do **not** use ``question``. Rebuild a minimal instruction
      from ``word1`` and ``word2`` only so the model never sees the released
      one-shot prompt (ICL demonstration stripped by construction).

    Attributes:
        task_name: Includes ``shot_mode`` so caches differ per configuration.
        shot_mode: Either ``\"zero_shot\"`` or ``\"one_shot\"``.
        input_schema: Single ``\"text\"`` field ``prompt`` for encoder models.
        output_schema: ``answer`` as ``\"multiclass\"`` over distinct strings.
    """

    input_schema: Dict[str, str] = {"prompt": "text"}
    output_schema: Dict[str, str] = {"answer": "multiclass"}

    def __init__(
        self,
        shot_mode: ShotMode = "one_shot",
        code_mapping: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        if shot_mode not in ("zero_shot", "one_shot"):
            raise ValueError(
                f"shot_mode must be 'zero_shot' or 'one_shot', got {shot_mode!r}"
            )
        super().__init__(code_mapping=code_mapping)
        self.shot_mode: ShotMode = shot_mode
        self.task_name = f"MedLingoJargonExpansionTask/{shot_mode}"

    def _build_prompt(self, event: Event) -> Optional[str]:
        """Build model input text for the current ``shot_mode``."""
        word1 = _as_str(event.word1)
        word2 = _as_str(event.word2)
        question = _as_str(event.question)

        if self.shot_mode == "one_shot":
            # Released conditioning: full CSV ``question`` (demo + query as
            # distributed).
            return question

        # zero_shot: ignore ``question`` entirely; ICL is not present by design.
        if word1 is None or word2 is None:
            return None
        return (
            "In plain language, define the medical jargon that connects "
            f'"{word1}" and "{word2}". Respond with the plain-language '
            "definition only."
        )

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Emit one sample per patient when fields are valid.

        Args:
            patient: Synthetic patient with a single ``questions`` event.

        Returns:
            A one-element list with ``id``, ``prompt``, and ``answer``, or
            empty if required fields are missing.
        """
        events = patient.get_events(event_type="questions")
        if len(events) != 1:
            return []
        event = events[0]
        answer = _as_str(event.answer)
        prompt = self._build_prompt(event)
        if prompt is None or answer is None:
            return []
        return [
            {
                "id": patient.patient_id,
                "prompt": prompt,
                "answer": answer,
            }
        ]
