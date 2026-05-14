"""EHR Fact Checking task for PyHealth.

Implements the claim-verification task introduced by:
    Zhang et al., "Dossier: Fact Checking in Electronic Health Records
    while Preserving Patient Privacy", MLHC 2024.

The task takes natural language claims about a patient's ICU stay and
assigns a veracity label: True (T), False (F), or Not-Enough-Information (N).
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from pyhealth.tasks.base_task import BaseTask

STANCE_TO_INT: Dict[str, int] = {"T": 0, "F": 1, "N": 2}
INT_TO_STANCE: Dict[int, str] = {0: "T", 1: "F", 2: "N"}

_REQUIRED_COLS = {"HADM_ID", "claim", "t_C", "label"}


class EHRFactCheckingMIMIC3(BaseTask):
    """EHR claim-verification task on MIMIC-III.

    Given a natural language claim about a patient's hospital stay, predict
    whether the claim is True (T), False (F), or Not-Enough-Information (N).

    This task is designed to pair with :class:`~pyhealth.models.DOSSIERPipeline`,
    which converts claims to SQL queries executed against evidence tables
    (Admission, Lab, Vital, Input) optionally augmented with a biomedical
    knowledge graph (SemMedDB).

    Unlike standard PyHealth tasks, the "features" here are symbolic patient
    tables (Pandas DataFrames) rather than fixed-size tensors.  The DOSSIER
    pipeline accesses those tables internally via MIMIC-III CSVs, so each
    sample produced by this task contains only the identifiers needed to look
    up the right patient and the associated claim metadata.

    Args:
        claims_df: DataFrame with at minimum the columns
            ``HADM_ID`` (int), ``claim`` (str), ``t_C`` (float, hours after
            admission), and ``label`` (str, one of "T" / "F" / "N").
            Additional columns (``lower``, ``upper``, ``stance``) are
            forwarded transparently and stored in each sample.
        code_mapping: Unused; kept for API consistency with BaseTask.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.tasks import EHRFactCheckingMIMIC3
        >>> claims = pd.DataFrame({
        ...     "HADM_ID": [100001, 100001],
        ...     "claim": [
        ...         "pt was given a blood thinner in the past 24 hours",
        ...         "systolic BP exceeded 140 since t=20",
        ...     ],
        ...     "t_C": [70.0, 100.0],
        ...     "label": ["T", "F"],
        ... })
        >>> task = EHRFactCheckingMIMIC3(claims_df=claims)
    """

    task_name: str = "EHRFactCheckingMIMIC3"

    # DOSSIER does not pass tensors through a DataLoader; the output_schema
    # entry drives metric selection in an evaluate() call.
    input_schema: Dict[str, str] = {}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        claims_df: pd.DataFrame,
        code_mapping: Optional[Dict] = None,
    ) -> None:
        """Initialise the task and validate the claims DataFrame.

        Args:
            claims_df: DataFrame containing claim records.  Must include the
                columns ``HADM_ID``, ``claim``, ``t_C``, and ``label``.
                Extra columns are forwarded into each produced sample.
            code_mapping: Unused; accepted for API compatibility with
                :class:`~pyhealth.tasks.base_task.BaseTask`.

        Raises:
            ValueError: If any of the required columns are absent from
                ``claims_df``.
        """
        super().__init__(code_mapping=code_mapping)
        missing = _REQUIRED_COLS - set(claims_df.columns)
        if missing:
            raise ValueError(
                f"claims_df is missing required columns: {missing}. "
                f"Got: {set(claims_df.columns)}"
            )
        self.claims_df = claims_df.copy()
        self.claims_df["HADM_ID"] = self.claims_df["HADM_ID"].astype(int)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient and return one sample per claim.

        Args:
            patient: PyHealth Patient object exposing ``get_events`` and a
                ``patient_id`` attribute.

        Returns:
            List of dicts (one per claim associated with any of this patient's
            admissions).  Each dict contains:

            * ``patient_id`` (str)
            * ``hadm_id`` (int) – MIMIC-III hospital admission ID
            * ``claim`` (str) – natural language claim text
            * ``t_C`` (float) – claim timestamp in hours relative to admission
            * ``label`` (int) – 0 = True, 1 = False, 2 = NEI
            * any extra columns from ``claims_df`` (e.g. ``lower``, ``upper``,
              ``stance``)
        """
        samples: List[Dict[str, Any]] = []
        patient_id = str(patient.patient_id)

        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        extra_cols = [
            c for c in self.claims_df.columns if c not in _REQUIRED_COLS
        ]

        for adm in admissions:
            hadm_id = int(adm.hadm_id)
            patient_claims = self.claims_df[self.claims_df["HADM_ID"] == hadm_id]
            if patient_claims.empty:
                continue

            for _, row in patient_claims.iterrows():
                label_str = str(row["label"]).strip().upper()
                if label_str not in STANCE_TO_INT:
                    continue

                sample: Dict[str, Any] = {
                    "patient_id": patient_id,
                    "hadm_id": hadm_id,
                    "claim": str(row["claim"]),
                    "t_C": float(row["t_C"]),
                    "label": STANCE_TO_INT[label_str],
                }
                for col in extra_cols:
                    sample[col] = row[col]

                samples.append(sample)

        return samples
