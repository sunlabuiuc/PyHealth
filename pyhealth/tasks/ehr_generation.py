from typing import Any, Dict, List

from pyhealth.tasks.base_task import BaseTask


class EHRGenerationMIMIC3(BaseTask):
    """Task for training synthetic EHR generative models using MIMIC-III.

    Transforms longitudinal patient records into a visit-sequence representation
    suitable for generative modeling. Each sample corresponds to one patient and
    captures the complete temporal trajectory of ICD-9 diagnosis codes across
    admissions.

    Two downstream representations can be derived from the output:

    * **Sequential** (PromptEHR, HALO, GPT): the nested ``conditions`` list
      retains full visit boundaries and ordering.
    * **Matrix / flattened** (MedGAN, CorGAN): flatten ``conditions`` into a
      single list (binary presence) or count vector per patient.

    For standardised evaluation, every synthetic or real record should be
    converted to a long-format schema of ``(subject_id, visit_id, code)``
    triplets as recommended by the paper *Accelerating Reproducible Research in
    Synthetic EHR Generation*.

    Attributes:
        task_name (str): Identifier for this task.
        input_schema (Dict[str, str]): ``{"conditions": "nested_sequence"}`` –
            tells PyHealth's processor to serialise the variable-length nested
            visit list correctly (same convention as
            ``DrugRecommendationMIMIC3``).
        output_schema (Dict[str, str]): ``{}`` – no supervised label is
            produced.
        min_visits (int): Minimum number of valid visits a patient must have
            to be included. Defaults to ``1``.
        truncate_icd (bool): When ``True``, ICD-9 codes are truncated to the
            first 3 characters (e.g. ``"250.40"`` → ``"250"``), reducing the
            vocabulary from ~6,955 to 1,071 codes. The paper recommends
            keeping ``False`` for full clinical fidelity. Defaults to
            ``False``.

    Note:
        A full end-to-end training example using a GPT-2 style decoder can be
        found at ``examples/ehr_generation/ehr_generation_mimic3_transformer.py``.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = EHRGenerationMIMIC3()
        >>> samples = dataset.set_task(task)
        >>> # Each sample: {patient_id, conditions, num_visits}
        >>> # conditions is a list of visits; each visit is a list of ICD-9 codes
        >>> # e.g. [["250.00", "401.9"], ["272.0", "428.0"]]
    """

    task_name: str = "EHRGenerationMIMIC3"
    input_schema: Dict[str, str] = {"conditions": "nested_sequence"}
    output_schema: Dict[str, str] = {}

    def __init__(
        self,
        min_visits: int = 1,
        truncate_icd: bool = False,
    ) -> None:
        """Initialise the task.

        Args:
            min_visits (int): Patients with fewer than ``min_visits`` valid
                admissions (i.e. admissions that contain at least one ICD-9
                code) are excluded. Defaults to ``1``.
            truncate_icd (bool): Truncate ICD-9 codes to 3-digit prefixes.
                Useful for reproducing prior work that caps the vocabulary at
                1,071 codes. Defaults to ``False`` (full 6,955-code vocabulary).
        """
        self.min_visits = min_visits
        self.truncate_icd = truncate_icd

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient and return a list with one generation sample.

        Each returned sample represents the patient's full longitudinal record
        as a nested list of ICD-9 diagnosis code sequences, one inner list per
        hospital admission (visit).

        Admissions with no ICD-9 codes are silently skipped. Patients with
        fewer valid visits than ``self.min_visits`` return an empty list.

        Args:
            patient: A PyHealth ``Patient`` object providing a
                ``get_events(event_type, filters)`` interface.

        Returns:
            A list containing a single dict with:

            * ``patient_id`` (str): MIMIC-III ``subject_id``.
            * ``conditions`` (List[List[str]]): Nested list of ICD-9 diagnosis
              codes, grouped by admission. Outer index = visit order (chronological);
              inner index = code index within that visit.
              The number of visits can be derived as ``len(conditions)``.
        """
        admissions = patient.get_events(event_type="admissions")

        visit_sequences: List[List[str]] = []

        for admission in admissions:
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            codes = [event.icd9_code for event in diagnoses if event.icd9_code]

            if self.truncate_icd:
                codes = [code[:3] for code in codes]

            # Deduplicate while preserving order
            seen: set = set()
            unique_codes: List[str] = []
            for code in codes:
                if code not in seen:
                    seen.add(code)
                    unique_codes.append(code)
            codes = unique_codes

            if not codes:
                continue

            visit_sequences.append(codes)

        if len(visit_sequences) < self.min_visits:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "conditions": visit_sequences,
            }
        ]

