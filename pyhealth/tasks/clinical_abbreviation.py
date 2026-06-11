import re
from typing import Any, Dict

class ClinicalAbbreviationTask:
    """
    Task for clinical abbreviation interpretation.

    Contributor: 
        Tedra Birch (tbirch2@illinois.edu)

    Paper: 
        Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?
        https://arxiv.org/abs/2505.15024


    This task converts Medlingo samples into model-ready input/label pairs.

    If `use_context` is False, the task uses the abbreviation directly.
    If `use_context` is True, the task attempts to extract a likely
    abbreviation from the clinical context.

    """

    task_name: str = "clinical_abbreviation"
    input_schema = {"input": "str"}
    output_schema = {"label": "str"}

    def __init__(self, use_context: bool = False) -> None:
        self.use_context = use_context

    def extract_abbreviation(self, text: str) -> str:
        """
        Extract a likely clinical abbreviation from text.

        Priority:
            1. uppercase abbreviations like SOB, BP, CHF, HTN
            2. mixed-case shorthand like Hx, Dx, Rx

        Args:
            text: The clinical context from which to extract an abbreviation.

        Returns:
            The extracted abbreviation, or an empty string if none is found.
        """
        # First, try to find uppercase abbreviations (2+ letters)
        upper_match = re.search(r"\b([A-Z]{2,})\b", text)
        if upper_match:
            return upper_match.group(0)

        # Then, try to find mixed-case shorthand (2+ letters)
        mixed_match = re.search(r"\b([A-Z][a-z]{1,})\b", text)
        if mixed_match:
            return mixed_match.group(0)

        return ""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert a MedLingo sample into task-ready format.

        Args:
            sample: A MedLingo sample containing abbreviation, context, and label.

        Returns:
            A dictionary with model input and expected label.
        """
        context = sample.get("context", "").strip()

        if self.use_context and context:
            extracted = self.extract_abbreviation(context)
            model_input = extracted if extracted else sample["abbr"]
        else:
            model_input = sample["abbr"]

        return {
            "input": model_input,
            "label": sample["label"],
        }