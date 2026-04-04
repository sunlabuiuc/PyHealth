from typing import Any, Dict, List
import json

from ..data import Patient
from .base_task import BaseTask


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


class ClinicalJargonVerification(BaseTask):
    """Binary candidate-verification task for public clinical jargon benchmarks."""

    task_name: str = "ClinicalJargonVerification"
    input_schema: Dict[str, str] = {"paired_text": "text"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        benchmark: str = "all",
        casi_variant: str = "release62",
        medlingo_distractors: int = 3,
    ) -> None:
        if benchmark not in {"all", "medlingo", "casi"}:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        if casi_variant not in {"release62", "paper59"}:
            raise ValueError(f"Unsupported CASI variant: {casi_variant}")
        if medlingo_distractors not in {1, 2, 3}:
            raise ValueError("medlingo_distractors must be 1, 2, or 3")
        self.benchmark = benchmark
        self.casi_variant = casi_variant
        self.medlingo_distractors = medlingo_distractors

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="examples")
        if len(events) != 1:
            return []
        event = events[0]

        if self.benchmark != "all" and event.benchmark != self.benchmark:
            return []
        if event.benchmark == "casi" and self.casi_variant == "paper59":
            if not parse_bool(event.paper59_included):
                return []

        if event.benchmark == "casi" and self.casi_variant == "paper59":
            candidates = json.loads(event.candidate_expansions_paper59_json)
        else:
            candidates = json.loads(event.candidate_expansions_json)

        if event.benchmark == "medlingo":
            gold = event.gold_expansion
            negatives = [candidate for candidate in candidates if candidate != gold]
            candidates = [gold, *negatives[: self.medlingo_distractors]]

        samples: List[Dict[str, Any]] = []
        for candidate in candidates:
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.sample_id,
                    "sample_id": event.sample_id,
                    "benchmark": event.benchmark,
                    "abbreviation": event.abbreviation,
                    "candidate_expansion": candidate,
                    "surface_form_group": event.surface_form_group,
                    "paired_text": (
                        f"Question: {event.question}\n"
                        f"Candidate expansion: {candidate}\n"
                        "Is this the correct expansion?"
                    ),
                    "label": int(candidate == event.gold_expansion),
                }
            )
        return samples
