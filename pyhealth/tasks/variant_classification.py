"""Variant classification tasks for PyHealth.

This module provides tasks for classifying genetic variants based on
their clinical significance using ClinVar and COSMIC datasets.
"""

from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class VariantClassificationClinVar(BaseTask):
    """Task for classifying variant clinical significance using ClinVar data.

    This task predicts the clinical significance of genetic variants
    (e.g., Pathogenic, Benign, Uncertain significance) based on variant
    features from the ClinVar database.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
        output_schema (Dict[str, str]): The output schema specifying outputs.
        CLINICAL_SIGNIFICANCE_CATEGORIES (Dict[str, str]): Mapping of raw values
            to standardized clinical significance labels.

    Note:
        Variants with conflicting interpretations or non-standard clinical
        significance values are excluded from the output samples.

    Examples:
        >>> from pyhealth.datasets import ClinVarDataset
        >>> from pyhealth.tasks import VariantClassificationClinVar
        >>> dataset = ClinVarDataset(root="/path/to/clinvar")
        >>> task = VariantClassificationClinVar()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "VariantClassificationClinVar"
    input_schema: Dict[str, str] = {
        "gene_symbol": "text",
        "variant_type": "text",
        "chromosome": "text",
    }
    output_schema: Dict[str, str] = {"clinical_significance": "multiclass"}

    # Standard clinical significance categories (ACMG/AMP guidelines)
    CLINICAL_SIGNIFICANCE_CATEGORIES: Dict[str, str] = {
        "pathogenic": "Pathogenic",
        "likely pathogenic": "Likely pathogenic",
        "benign": "Benign",
        "likely benign": "Likely benign",
        "uncertain significance": "Uncertain significance",
        "vus": "Uncertain significance",
    }

    def _normalize_clinical_significance(
        self, raw_value: Optional[str]
    ) -> Optional[str]:
        """Normalize clinical significance to standard ACMG/AMP categories.

        Args:
            raw_value: Raw clinical significance string from ClinVar.

        Returns:
            Normalized category string, or None if value is invalid or
            does not map to a standard category.
        """
        if raw_value is None or raw_value == "" or str(raw_value) == "nan":
            return None

        value_lower = str(raw_value).lower()

        # Check for exact or partial matches
        if "likely pathogenic" in value_lower:
            return self.CLINICAL_SIGNIFICANCE_CATEGORIES["likely pathogenic"]
        elif "pathogenic" in value_lower and "likely" not in value_lower:
            return self.CLINICAL_SIGNIFICANCE_CATEGORIES["pathogenic"]
        elif "likely benign" in value_lower:
            return self.CLINICAL_SIGNIFICANCE_CATEGORIES["likely benign"]
        elif "benign" in value_lower and "likely" not in value_lower:
            return self.CLINICAL_SIGNIFICANCE_CATEGORIES["benign"]
        elif "uncertain" in value_lower or "vus" in value_lower:
            return self.CLINICAL_SIGNIFICANCE_CATEGORIES["uncertain significance"]

        # Conflicting or unrecognized categories
        return None

    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely convert value to string, handling None and NaN.

        Args:
            value: Value to convert.
            default: Default value if conversion fails.

        Returns:
            String representation of value or default.
        """
        if value is None or str(value) == "nan":
            return default
        return str(value)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a variant record to extract features and label.

        Args:
            patient: A patient object containing variant data.
                In ClinVar, each "patient" represents a single variant record.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with
            variant features and clinical significance label. Returns an empty
            list if the variant has no events, invalid clinical significance,
            or conflicting interpretations.

        Note:
            Returns empty list for variants with:
            - No variant events
            - Missing or empty clinical significance
            - Conflicting interpretations
            - Non-standard clinical significance values
        """
        events = patient.get_events(event_type="variants")

        if len(events) == 0:
            return []

        event = events[0]

        # Normalize clinical significance
        raw_clinical_sig = getattr(event, "clinical_significance", None)
        label = self._normalize_clinical_significance(raw_clinical_sig)

        if label is None:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "gene_symbol": self._safe_str(getattr(event, "gene_symbol", "")),
                "variant_type": self._safe_str(getattr(event, "variant_type", "")),
                "chromosome": self._safe_str(getattr(event, "chromosome", "")),
                "clinical_significance": label,
            }
        ]


class MutationPathogenicityPrediction(BaseTask):
    """Task for predicting mutation pathogenicity using COSMIC data.

    This task predicts whether a somatic mutation is pathogenic or neutral
    based on FATHMM predictions and mutation features from the COSMIC database.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
        output_schema (Dict[str, str]): The output schema specifying outputs.
        VALID_FATHMM_PREDICTIONS (tuple): Valid FATHMM prediction values.

    Note:
        Only mutations with valid FATHMM predictions (PATHOGENIC or NEUTRAL)
        are included in the output samples.

    Examples:
        >>> from pyhealth.datasets import COSMICDataset
        >>> from pyhealth.tasks import MutationPathogenicityPrediction
        >>> dataset = COSMICDataset(root="/path/to/cosmic")
        >>> task = MutationPathogenicityPrediction()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MutationPathogenicityPrediction"
    input_schema: Dict[str, str] = {
        "gene_name": "text",
        "mutation_description": "text",
        "primary_site": "text",
    }
    output_schema: Dict[str, str] = {"fathmm_prediction": "binary"}

    # Valid FATHMM prediction categories
    VALID_FATHMM_PREDICTIONS: tuple = ("PATHOGENIC", "NEUTRAL")

    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely convert value to string, handling None and NaN.

        Args:
            value: Value to convert.
            default: Default value if conversion fails.

        Returns:
            String representation of value or default.
        """
        if value is None or str(value) == "nan":
            return default
        return str(value)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process mutation records to extract features and pathogenicity label.

        Args:
            patient: A patient object containing mutation data.
                In COSMIC, each "patient" represents a sample with mutations.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, one per valid mutation,
            each containing mutation features and binary pathogenicity label.
            Returns an empty list if no mutations have valid FATHMM predictions.

        Note:
            Only mutations with FATHMM predictions of "PATHOGENIC" or "NEUTRAL"
            are included. Mutations with missing or other prediction values
            are excluded.
        """
        events = patient.get_events(event_type="mutations")
        samples: List[Dict[str, Any]] = []

        for event in events:
            fathmm = getattr(event, "fathmm_prediction", None)

            if fathmm not in self.VALID_FATHMM_PREDICTIONS:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "gene_name": self._safe_str(getattr(event, "gene_name", "")),
                    "mutation_description": self._safe_str(
                        getattr(event, "mutation_description", "")
                    ),
                    "primary_site": self._safe_str(getattr(event, "primary_site", "")),
                    "fathmm_prediction": 1 if fathmm == "PATHOGENIC" else 0,
                }
            )

        return samples
