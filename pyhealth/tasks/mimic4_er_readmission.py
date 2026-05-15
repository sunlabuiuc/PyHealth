"""
Name: Ranjithkumar Rajendran
NetID: rr54
Paper: KEEP (CHIL 2025) — Elhussein et al.
"""
from datetime import datetime, timedelta
from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask


class ERReadmissionMIMIC4(BaseTask):
    """ER-Specific Readmission prediction on MIMIC-IV.

    Predicts whether an emergency-room patient will be
    readmitted within a specified window (default 30 days)
    based on clinical information from the current ER visit.

    Only visits whose ``admission_location`` is
    ``'EMERGENCY ROOM'`` are considered.  Diagnosis codes
    are prefixed with their ICD version (``"9_"`` or
    ``"10_"``) to match the format used by
    :class:`~pyhealth.tasks.ReadmissionPredictionMIMIC4`.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict[str, str]): Input schema.
        output_schema (Dict[str, str]): Output schema.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import ERReadmissionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = ERReadmissionMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ERReadmissionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(
        self, window: timedelta = timedelta(days=30)
    ) -> None:
        """Initialise the task.

        Args:
            window: If a subsequent admission occurs within
                this window of an ER discharge, it is
                labelled as a readmission.  Defaults to
                30 days per KEEP (2025).
        """
        self.window = window

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generate binary samples for one patient.

        Visits with no diagnoses are skipped.  Only visits
        where ``admission_location == 'EMERGENCY ROOM'``
        are processed.

        Args:
            patient: A PyHealth patient object.

        Returns:
            A list of sample dicts, each containing
            ``visit_id``, ``patient_id``, ``conditions``
            (list of versioned ICD strings), and
            ``readmission`` (0 or 1).
        """
        admissions: List[Event] = patient.get_events(
            event_type="admissions"
        )
        if len(admissions) < 2:
            return []

        samples = []
        for i in range(len(admissions) - 1):
            adm = admissions[i]
            loc = getattr(adm, "admission_location", "")
            if loc != "EMERGENCY ROOM":
                continue

            filt = ("hadm_id", "==", adm.hadm_id)

            diagnoses = []
            for ev in patient.get_events(
                event_type="diagnoses_icd",
                filters=[filt],
            ):
                ver = getattr(ev, "icd_version", "10")
                diagnoses.append(
                    f"{ver}_{ev.icd_code}"
                )

            if not diagnoses:
                continue

            try:
                disch = datetime.strptime(
                    adm.dischtime,
                    "%Y-%m-%d %H:%M:%S",
                )
            except ValueError:
                disch = datetime.strptime(
                    adm.dischtime, "%Y-%m-%d"
                )

            readmit = int(
                (admissions[i + 1].timestamp - disch)
                < self.window
            )

            samples.append(
                {
                    "visit_id": adm.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": diagnoses,
                    "readmission": readmit,
                }
            )

        return samples
