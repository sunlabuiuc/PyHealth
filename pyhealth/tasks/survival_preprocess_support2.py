import logging
from typing import Any, Dict, List, Optional

from .base_task import BaseTask

logger = logging.getLogger(__name__)


class SurvivalPreprocessSupport2(BaseTask):
    """Preprocessing task for survival probability prediction models using SUPPORT2 dataset.

    This task extracts features and labels from raw patient data to prepare samples
    for model training. It extracts patient demographics, diagnoses, clinical
    measurements, and vital signs, and pairs them with ground truth survival
    probabilities from the dataset (surv2m or surv6m fields).

    The task performs feature extraction and data structuring:
    - Extracts features from raw patient data (demographics, vitals, labs, scores, etc.)
    - Extracts ground truth survival probabilities from surv2m/surv6m fields
    - Structures data into samples ready for model training

    The SUPPORT2 dataset contains data on seriously ill hospitalized adults,
    with each patient represented by a single record at admission.

    Task Schema:
        Input:
            - demographics: sequence of demographic features (age, sex, race, education, income)
            - disease_codes: sequence of disease group and class codes
            - vitals: sequence of vital signs (meanbp, hrt, resp, temp, pafi)
            - labs: sequence of lab values (wblc, alb, bili, crea, sod, ph, glucose, bun)
            - scores: sequence of clinical scores (sps, aps, scoma)
            - comorbidities: sequence of comorbidity indicators (diabetes, dementia, ca)
        Output:
            - survival_probability: regression label (0-1, ground truth survival probability)

    Args:
        time_horizon (str): Which survival probability to extract.
            Options: "2m" (2 months) or "6m" (6 months). Default is "2m".

    Returns:
        List[Dict[str, Any]]: A list containing a single sample per patient with:
            - patient_id: The patient identifier
            - demographics: List of demographic feature strings (e.g., ["age_62.85", "sex_male"])
            - disease_codes: List of disease code strings
            - vitals: List of vital sign strings
            - labs: List of lab value strings
            - scores: List of clinical score strings
            - comorbidities: List of comorbidity indicator strings
            - survival_probability: Float between 0 and 1 (ground truth label)

    Examples:
        >>> from pyhealth.datasets import Support2Dataset
        >>> from pyhealth.tasks import SurvivalPreprocessSupport2
        >>>
        >>> # Step 1: Load SUPPORT2 dataset
        >>> print("Step 1: Load SUPPORT2 Dataset")
        >>> # For real usage, use your dataset path:
        >>> # dataset = Support2Dataset(
        >>> #     root="/path/to/support2/data",
        >>> #     tables=["support2"]
        >>> # )
        >>> # For local testing with test data:
        >>> from pathlib import Path
        >>> test_data_path = Path("test-resources/core/support2")
        >>> dataset = Support2Dataset(
        ...     root=str(test_data_path),
        ...     tables=["support2"]
        ... )
        >>> print(f"Loaded dataset with {len(dataset.unique_patient_ids)} patients\n")
        >>>
        >>> # Step 2: Apply preprocessing task to extract features and labels
        >>> print("Step 2: Apply Survival Preprocessing Task")
        >>> task = SurvivalPreprocessSupport2(time_horizon="2m")
        >>> sample_dataset = dataset.set_task(task=task)
        >>> print(f"Generated {len(sample_dataset)} samples")
        >>> print(f"Input schema: {sample_dataset.input_schema}")
        >>> print(f"Output schema: {sample_dataset.output_schema}\n")
        >>>
        >>> # Helper function to decode tensor indices to feature strings
        >>> def decode_features(tensor, processor):
        ...     if processor is None or not hasattr(processor, 'code_vocab'):
        ...         return [str(idx.item()) for idx in tensor]
        ...     reverse_vocab = {idx: token for token, idx in processor.code_vocab.items()}
        ...     return [reverse_vocab.get(idx.item(), f"<unk:{idx.item()}>") for idx in tensor]
        >>>
        >>> # Step 3: Display features for one sample
        >>> print("Step 3: Examine Preprocessed Samples")
        >>> sample = sample_dataset[0]
        >>> print(f"Patient {sample['patient_id']}:")
        >>> print(f"Demographics tensor shape: {sample['demographics'].shape}")
        >>> print(f"Disease codes tensor shape: {sample['disease_codes'].shape}")
        >>> print(f"Vitals tensor shape: {sample['vitals'].shape}")
        >>> print(f"Labs tensor shape: {sample['labs'].shape}")
        >>> print(f"Scores tensor shape: {sample['scores'].shape}")
        >>> print(f"Comorbidities tensor shape: {sample['comorbidities'].shape}")
        >>>
        >>> # Decode and display features for this sample
        >>> demographics_decoded = decode_features(
        ...     sample['demographics'],
        ...     sample_dataset.input_processors.get('demographics')
        ... )
        >>> print(f"  Demographics: {', '.join(demographics_decoded)}")
        >>> disease_codes_decoded = decode_features(
        ...     sample['disease_codes'],
        ...     sample_dataset.input_processors.get('disease_codes')
        ... )
        >>> print(f"  Disease Codes: {', '.join(disease_codes_decoded)}")
        >>> vitals_decoded = decode_features(
        ...     sample['vitals'],
        ...     sample_dataset.input_processors.get('vitals')
        ... )
        >>> print(f"  Vitals: {', '.join(vitals_decoded)}")
        >>> print(f"  Survival Probability (2m): {sample['survival_probability'].item():.4f}")
        >>>
        >>> # For a complete working example displaying all feature groups for all samples,
        >>> # see: examples/survival_preprocess_support2_demo.py

    Note:
        - Each patient produces exactly one sample (single-row-per-patient dataset)
        - Missing values in labs/vitals are handled by excluding None values
        - Survival probabilities are ground truth labels extracted from surv2m/surv6m fields
        - Processors will automatically convert string features to tensors for model training
    """

    task_name: str = "SurvivalPreprocessSupport2"
    input_schema: Dict[str, str] = {
        "demographics": "sequence",
        "disease_codes": "sequence",
        "vitals": "sequence",
        "labs": "sequence",
        "scores": "sequence",
        "comorbidities": "sequence",
    }
    output_schema: Dict[str, str] = {"survival_probability": "regression"}

    def __init__(self, time_horizon: str = "2m"):
        """Initialize the SurvivalPreprocessSupport2 preprocessing task.

        Args:
            time_horizon (str): Which survival probability to extract as the label.
                Options: "2m" (2 months) or "6m" (6 months). Default is "2m".
        """
        super().__init__()
        self.time_horizon = time_horizon
        if time_horizon == "2m":
            self.survival_field = "surv2m"
            self.task_name = "SurvivalPreprocessSupport2_2m"
        elif time_horizon == "6m":
            self.survival_field = "surv6m"
            self.task_name = "SurvivalPreprocessSupport2_6m"
        else:
            raise ValueError(
                f"time_horizon must be '2m' or '6m', got {time_horizon}"
            )

    def _clean_value(self, value: Any) -> Optional[float]:
        """Clean a value by converting to float, handling None and empty strings.

        Args:
            value: The value to clean

        Returns:
            Optional[float]: Cleaned float value, or None if invalid
        """
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() == "none":
                return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_attr_safe(self, event: Any, attr: str) -> Any:
        """Safely get an attribute from an event.

        Args:
            event: The event object
            attr: The attribute name

        Returns:
            The attribute value or None if not present
        """
        try:
            return getattr(event, attr, None)
        except AttributeError:
            try:
                return event[attr]
            except (KeyError, TypeError):
                return None

    def _extract_numeric_features(
        self, event: Any, features: Dict[str, str]
    ) -> List[str]:
        """Extract numeric features from an event and format them.

        Args:
            event: The event object
            features: Dict mapping prefix to attribute name (e.g., {"age": "age"})

        Returns:
            List of formatted feature strings (e.g., ["age_62.85"])
        """
        result = []
        for prefix, attr in features.items():
            value = self._clean_value(self._get_attr_safe(event, attr))
            if value is not None:
                result.append(f"{prefix}_{value}")
        return result

    def _extract_string_features(
        self, event: Any, features: Dict[str, str]
    ) -> List[str]:
        """Extract string features from an event and format them.

        Args:
            event: The event object
            features: Dict mapping prefix to attribute name (e.g., {"sex": "sex"})

        Returns:
            List of formatted feature strings (e.g., ["sex_male"])
        """
        result = []
        for prefix, attr in features.items():
            value = self._get_attr_safe(event, attr)
            if value is not None:
                result.append(f"{prefix}_{str(value)}")
        return result

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extracts features and labels from a patient for survival probability prediction models.

        This is a preprocessing step that structures raw patient data into
        features (demographics, vitals, labs, etc.) and ground truth labels
        (survival probabilities) ready for model training.

        Args:
            patient (Any): A Patient object containing SUPPORT2 data.

        Returns:
            List[Dict[str, Any]]: A list containing a single sample per patient
                with extracted features and survival probability label.
        """
        # Get the single support2 event per patient
        events = patient.get_events(event_type="support2")
        
        if len(events) == 0:
            return []
        
        # Should be exactly one event per patient (single-row-per-patient dataset)
        if len(events) > 1:
            logger.warning(
                f"Patient {patient.patient_id} has {len(events)} support2 events, "
                "expected 1. Using first event."
            )

        event = events[0]

        # Extract demographics (mixed numeric and string features)
        demographics = []
        demographics.extend(
            self._extract_numeric_features(event, {"age": "age", "edu": "edu"})
        )
        demographics.extend(
            self._extract_string_features(event, {"sex": "sex", "race": "race", "income": "income"})
        )

        # Extract disease codes
        disease_codes = self._extract_string_features(
            event, {"dzgroup": "dzgroup", "dzclass": "dzclass"}
        )

        # Extract vital signs
        vitals = self._extract_numeric_features(
            event, {"meanbp": "meanbp", "hrt": "hrt", "resp": "resp", "temp": "temp", "pafi": "pafi"}
        )

        # Extract lab values
        labs = self._extract_numeric_features(
            event,
            {
                "wblc": "wblc",
                "alb": "alb",
                "bili": "bili",
                "crea": "crea",
                "sod": "sod",
                "ph": "ph",
                "glucose": "glucose",
                "bun": "bun",
            },
        )

        # Extract clinical scores
        scores = self._extract_numeric_features(
            event, {"sps": "sps", "aps": "aps", "scoma": "scoma"}
        )

        # Extract comorbidities (mixed numeric and string features)
        comorbidities = []
        comorbidities.extend(
            self._extract_numeric_features(event, {"diabetes": "diabetes", "dementia": "dementia"})
        )
        comorbidities.extend(self._extract_string_features(event, {"ca": "ca"}))

        # Extract ground truth survival probability label from dataset
        # (surv2m or surv6m field contains pre-computed survival probabilities)
        survival_prob = self._clean_value(self._get_attr_safe(event, self.survival_field))
        
        # Skip if survival probability is missing
        if survival_prob is None:
            return []

        # Ensure survival probability is in valid range [0, 1]
        survival_prob = max(0.0, min(1.0, survival_prob))

        # Create single sample per patient
        sample = {
            "patient_id": patient.patient_id,
            "demographics": demographics,
            "disease_codes": disease_codes,
            "vitals": vitals,
            "labs": labs,
            "scores": scores,
            "comorbidities": comorbidities,
            "survival_probability": survival_prob,
        }

        return [sample]
