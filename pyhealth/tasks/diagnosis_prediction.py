"""
Name: Hyunsoo Lee
NetId: hyunsoo2

Description: Task for next-visit diagnosis prediction

The example is implemented in examples/shy_mimic3_demo.py
"""

from typing import Any, Dict, List
import polars as pl
from .base_task import BaseTask

class DiagnosisPredictionMIMIC3(BaseTask):
    """Task for next-visit diagnosis prediction using the MIMIC-III dataset.

    This task follows a standard next-visit prediction setting:
    given the clinical information from the *current* admission
    (diagnoses, procedures, medications), predict the set of
    diagnosis codes that will appear in the *next* admission.

    Each sample corresponds to one admission with at least one
    *subsequent* admission for the same patient.

    Attributes
    ----------
    task_name:
        Name of the task.
    input_schema:
        Schema of input features produced by this task:
        - ``conditions``: flat list of diagnosis codes for the current visit.
        - ``procedures``: flat list of procedure codes for the current visit.
        - ``drugs``: flat list of drug codes for the current visit.
    output_schema:
        Schema of output labels:
        - ``label``: multi-label set of diagnosis codes for the *next* visit.
    """

    task_name: str = "DiagnosisPredictionMIMIC3"

    # Flat lists of codes for the current visit.
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }

    # Multi-label diagnosis codes for the *next* visit.
    output_schema: Dict[str, str] = {"label": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient into diagnosis prediction samples.

        Parameters
        ----------
        patient
            A :class:`~pyhealth.data.Patient` object with admissions,
            diagnoses, procedures, and prescriptions available.

        Returns
        -------
        List[Dict[str, Any]]
            A list of samples. Each sample has the following keys:

            - ``patient_id``: patient identifier.
            - ``visit_id``: current admission ID (HADM_ID).
            - ``conditions``: list of diagnosis codes for current visit.
            - ``procedures``: list of procedure codes for current visit.
            - ``drugs``: list of drug codes (ATC-3) for current visit.
            - ``label``: list of diagnosis codes for *next* visit.

            If the patient has fewer than two valid visits (current + next),
            an empty list is returned.
        """
        samples: List[Dict[str, Any]] = []

        # Retrieve all admissions for this patient (MIMIC-III uses ICD-9 codes)
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            # Need at least 2 admissions to form (current, next) pairs
            return []

        # Iterate over consecutive admission pairs (current, next)
        for idx in range(len(admissions) - 1):
            current_admission = admissions[idx]
            next_admission = admissions[idx + 1]

            # Extract current visit features
            # Diagnoses (ICD-9 codes) for current admission
            current_hadm_id = current_admission.hadm_id
            diagnoses_df = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", current_hadm_id)],
                return_df=True,
            )
            conditions = (
                diagnoses_df.select(pl.col("diagnoses_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Procedures (ICD-9 codes) for current admission
            procedures_df = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", current_hadm_id)],
                return_df=True,
            )
            procedures = (
                procedures_df.select(pl.col("procedures_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Prescriptions for current admission (ATC-3: first 4 characters)
            prescriptions_df = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", current_hadm_id)],
                return_df=True,
            )
            drugs = (
                prescriptions_df.select(pl.col("prescriptions/drug"))
                .to_series()
                .to_list()
            )
            drugs = [drug_code[:4] for drug_code in drugs if drug_code]

            # Extract next visit labels
            next_hadm_id = next_admission.hadm_id
            next_diag_df = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", next_hadm_id)],
                return_df=True,
            )
            next_labels = (
                next_diag_df.select(pl.col("diagnoses_icd/icd9_code"))
                .to_series()
                .to_list()
            )

            # Skip visits with missing key features
            if not (conditions and procedures and drugs and next_labels):
                continue

            samples.append(
                {
                    "visit_id": current_hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "label": next_labels,
                }
            )

        return samples


class DiagnosisPredictionMIMIC4(BaseTask):
    """Task for next-visit diagnosis prediction using the MIMIC-IV dataset.

    Similar to :class:`DiagnosisPredictionMIMIC3`, but tailored to the
    MIMIC-IV schema, where diagnosis and procedure codes are stored as
    ``icd_version`` + ``icd_code``, and prescriptions use NDC codes.

    We concatenate the version and code (e.g., ``"9_41071"`` or ``"10_I21"``)
    to form stable categorical tokens.

    Attributes
    ----------
    task_name:
        Name of the task.
    input_schema:
        Schema of input features produced by this task:
        - ``conditions``: flat list of diagnosis tokens for current visit.
        - ``procedures``: flat list of procedure tokens for current visit.
        - ``drugs``: flat list of drug codes (ATC-3 from NDC) for current visit.
    output_schema:
        Schema of output labels:
        - ``label``: multi-label set of diagnosis tokens for the *next* visit.
    """

    task_name: str = "DiagnosisPredictionMIMIC4"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"label": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient into diagnosis prediction samples.

        Parameters
        ----------
        patient
            A :class:`~pyhealth.data.Patient` object built from MIMIC-IV,
            with events ``admissions``, ``diagnoses_icd``, ``procedures_icd``,
            and ``prescriptions``.

        Returns
        -------
        List[Dict[str, Any]]
            A list of samples. Each sample has the following keys:

            - ``patient_id``: patient identifier.
            - ``visit_id``: current admission ID (HADM_ID).
            - ``conditions``: list of diagnosis tokens for current visit
              (``icd_version`` + ``icd_code``).
            - ``procedures``: list of procedure tokens for current visit
              (``icd_version`` + ``icd_code``).
            - ``drugs``: list of drug codes (ATC-3) for current visit.
            - ``label``: list of diagnosis tokens for *next* visit.

            If the patient has fewer than two valid visits (current + next),
            an empty list is returned.
        """
        samples: List[Dict[str, Any]] = []

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        # Iterate over consecutive admission pairs (current, next)
        for idx in range(len(admissions) - 1):
            current_admission = admissions[idx]
            next_admission = admissions[idx + 1]

            # Extract current visit features
            # Diagnoses: concat icd_version + icd_code (e.g., "9_41071")
            current_hadm_id = current_admission.hadm_id
            diagnoses_df = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", current_hadm_id)],
                return_df=True,
            )
            conditions = (
                diagnoses_df.select(
                    pl.concat_str(
                        [
                            "diagnoses_icd/icd_version",
                            "diagnoses_icd/icd_code",
                        ],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            # Procedures: concat icd_version + icd_code
            procedures_df = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", current_hadm_id)],
                return_df=True,
            )
            procedures = (
                procedures_df.select(
                    pl.concat_str(
                        [
                            "procedures_icd/icd_version",
                            "procedures_icd/icd_code",
                        ],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            # Prescriptions: NDC -> ATC-3 (first 4 characters)
            # Note: when using code_mapping={"NDC": "ATC"}, this column
            # will already be mapped to ATC codes in the dataset.
            prescriptions_df = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", current_hadm_id)],
                return_df=True,
            )
            drugs = (
                prescriptions_df.select(pl.col("prescriptions/ndc"))
                .to_series()
                .to_list()
            )
            drugs = [drug_code[:4] for drug_code in drugs if drug_code]

            # Extract next visit labels
            next_hadm_id = next_admission.hadm_id
            next_diag_df = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", next_hadm_id)],
                return_df=True,
            )
            next_labels = (
                next_diag_df.select(
                    pl.concat_str(
                        [
                            "diagnoses_icd/icd_version",
                            "diagnoses_icd/icd_code",
                        ],
                        separator="_",
                    )
                )
                .to_series()
                .to_list()
            )

            # Skip visits with missing key features
            if not (conditions and procedures and drugs and next_labels):
                continue

            samples.append(
                {
                    "visit_id": current_hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "label": next_labels,
                }
            )

        return samples