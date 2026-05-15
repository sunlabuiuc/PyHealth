"""Cancer survival prediction tasks for PyHealth.

This module provides tasks for predicting cancer patient survival outcomes
using multi-omics data from TCGA datasets.
"""

from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class CancerSurvivalPrediction(BaseTask):
    """Task for predicting cancer patient survival outcomes.

    This task predicts whether a cancer patient is alive or deceased based on
    their mutation profile and clinical features from TCGA datasets.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
        output_schema (Dict[str, str]): The output schema specifying outputs.
        VITAL_STATUS_DEAD (tuple): Values indicating deceased status.
        VITAL_STATUS_ALIVE (tuple): Values indicating alive status.

    Note:
        Patients without clinical data or with unknown vital status are
        excluded from the output samples.

    Examples:
        >>> from pyhealth.datasets import TCGAPRADDataset
        >>> from pyhealth.tasks import CancerSurvivalPrediction
        >>> dataset = TCGAPRADDataset(root="/path/to/tcga_prad")
        >>> task = CancerSurvivalPrediction()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "CancerSurvivalPrediction"
    input_schema: Dict[str, str] = {
        "mutations": "sequence",
        "age_at_diagnosis": "tensor",
        "gleason_score": "tensor",
    }
    output_schema: Dict[str, str] = {"vital_status": "binary"}

    # Vital status category mappings
    VITAL_STATUS_DEAD: tuple = ("dead", "deceased", "1")
    VITAL_STATUS_ALIVE: tuple = ("alive", "living", "0")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, handling None and NaN.

        Args:
            value: Value to convert.
            default: Default value if conversion fails.

        Returns:
            Float representation of value or default.
        """
        if value is None or str(value) == "nan":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _extract_genes(self, mutations: List[Any]) -> List[str]:
        """Extract valid gene symbols from mutation events.

        Args:
            mutations: List of mutation event objects.

        Returns:
            List of gene symbol strings, excluding None and NaN values.
        """
        genes: List[str] = []
        for mut in mutations:
            gene = getattr(mut, "hugo_symbol", None)
            if gene is not None and str(gene) != "nan":
                genes.append(str(gene))
        return genes

    def _parse_vital_status(self, raw_value: Any) -> Optional[int]:
        """Parse vital status to binary label.

        Args:
            raw_value: Raw vital status value from clinical data.

        Returns:
            1 for deceased, 0 for alive, None if value is invalid or unknown.
        """
        if raw_value is None or str(raw_value) == "nan":
            return None

        value_lower = str(raw_value).lower()

        if value_lower in self.VITAL_STATUS_DEAD:
            return 1
        elif value_lower in self.VITAL_STATUS_ALIVE:
            return 0

        return None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process patient mutation and clinical data for survival prediction.

        Args:
            patient: A patient object containing mutation and clinical data.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with
            patient features and survival label. Returns an empty list if
            clinical data is missing or vital status is unknown.

        Note:
            Returns empty list for patients with:
            - No clinical events
            - Missing or null vital status
            - Unrecognized vital status values
        """
        mutations = patient.get_events(event_type="mutations")
        clinical = patient.get_events(event_type="clinical")

        if len(clinical) == 0:
            return []

        clin = clinical[0]

        # Parse vital status
        vital_status = self._parse_vital_status(
            getattr(clin, "vital_status", None)
        )
        if vital_status is None:
            return []

        # Extract features
        mutated_genes = self._extract_genes(mutations)
        age = self._safe_float(getattr(clin, "age_at_diagnosis", None))
        gleason = self._safe_float(getattr(clin, "gleason_score", None))

        return [
            {
                "patient_id": patient.patient_id,
                "mutations": mutated_genes,
                "age_at_diagnosis": age,
                "gleason_score": gleason,
                "vital_status": vital_status,
            }
        ]


class CancerMutationBurden(BaseTask):
    """Task for predicting high vs low tumor mutation burden.

    This task classifies patients based on their tumor mutation burden (TMB),
    which is associated with immunotherapy response. TMB is approximated by
    counting the number of mutated genes.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
        output_schema (Dict[str, str]): The output schema specifying outputs.
        TMB_THRESHOLD (int): Mutation count threshold for high TMB classification.

    Note:
        This is a simplified TMB calculation based on gene count. Clinical TMB
        is typically measured as mutations per megabase of sequenced DNA.

    Examples:
        >>> from pyhealth.datasets import TCGAPRADDataset
        >>> from pyhealth.tasks import CancerMutationBurden
        >>> dataset = TCGAPRADDataset(root="/path/to/tcga_prad")
        >>> task = CancerMutationBurden()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "CancerMutationBurden"
    input_schema: Dict[str, str] = {
        "mutations": "sequence",
        "age_at_diagnosis": "tensor",
    }
    output_schema: Dict[str, str] = {"high_tmb": "binary"}

    # TMB threshold (number of mutated genes for high TMB classification)
    TMB_THRESHOLD: int = 10

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, handling None and NaN.

        Args:
            value: Value to convert.
            default: Default value if conversion fails.

        Returns:
            Float representation of value or default.
        """
        if value is None or str(value) == "nan":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _extract_genes(self, mutations: List[Any]) -> List[str]:
        """Extract valid gene symbols from mutation events.

        Args:
            mutations: List of mutation event objects.

        Returns:
            List of gene symbol strings, excluding None and NaN values.
        """
        genes: List[str] = []
        for mut in mutations:
            gene = getattr(mut, "hugo_symbol", None)
            if gene is not None and str(gene) != "nan":
                genes.append(str(gene))
        return genes

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process patient data to predict tumor mutation burden.

        Args:
            patient: A patient object containing mutation data.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with
            patient features and TMB classification label.

        Note:
            High TMB is defined as having >= TMB_THRESHOLD mutated genes.
            All patients with mutation data are included in the output.
        """
        mutations = patient.get_events(event_type="mutations")
        clinical = patient.get_events(event_type="clinical")

        # Extract mutated genes
        mutated_genes = self._extract_genes(mutations)

        # Classify TMB based on mutation count
        high_tmb = 1 if len(mutated_genes) >= self.TMB_THRESHOLD else 0

        # Get age if clinical data available
        age = 0.0
        if len(clinical) > 0:
            age = self._safe_float(getattr(clinical[0], "age_at_diagnosis", None))

        return [
            {
                "patient_id": patient.patient_id,
                "mutations": mutated_genes,
                "age_at_diagnosis": age,
                "high_tmb": high_tmb,
            }
        ]
