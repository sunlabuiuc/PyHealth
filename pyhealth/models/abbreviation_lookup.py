import re
from typing import Any


class AbbreviationLookupModel:
    """
    Simple rule-based model for clinical abbreviation interpretation.

    Contributor:
        Tedra Birch (tbirch2@illinois.edu)

    Paper:
        Diagnosing Our Datasets: How Does My Language Model Learn Clinical Information?
        https://arxiv.org/abs/2505.15024

    This model builds a dictionary mapping clinical abbreviations to their
    expanded labels. It optionally normalizes input by stripping punctuation
    and converting text to uppercase.

    Args:
        normalize: Whether to normalize inputs before lookup.

    Example:
        >>> model = AbbreviationLookupModel()
        >>> model.fit(samples)
        >>> model.predict("SOB")
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self.lookup: dict[str, str] = {}

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for rule-based lookup.

        Args:
            text: Input abbreviation string.

        Returns:
            Normalized abbreviation string.
        """
        text = text.strip()
        if self.normalize:
            text = re.sub(r"[^A-Za-z0-9/]+", "", text)
            text = text.upper()
        return text

    def fit(self, samples: list[dict[str, Any]]) -> None:
        """
        Fit the lookup table from abbreviation-label pairs.

        Args:
            samples: List of samples containing 'abbr' and 'label' keys.
        """
        for sample in samples:
            key = self._normalize_text(sample["abbr"])
            self.lookup[key] = sample["label"]

    def predict(self, input_text: str) -> str:
        """
        Predict the expanded meaning of an abbreviation.

        Args:
            input_text: Abbreviation string.

        Returns:
            Expanded abbreviation label if found, otherwise 'unknown'.
        """
        key = self._normalize_text(input_text)
        return self.lookup.get(key, "unknown")
    
    