from typing import Any, Dict, List

from .base_task import BaseTask


class FutureSeverityPredictionEICU(BaseTask):
    """Task for predicting near-future ICU severity using eICU data.

    This task asks: based on the current ICU stay, how severe the patient's
    condition will become in the near future.

    This implementation uses a simple proxy label based on the amount of
    future clinical activity in the next ICU stay(s).

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import FutureSeverityPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu",
        ...     tables=["diagnosis", "medication", "physicalexam"],
        ... )
        >>> task = FutureSeverityPredictionEICU(future_window=1)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "FutureSeverityPredictionEICU"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }

    output_schema: Dict[str, str] = {"future_severity": "multiclass"}

    def __init__(self, future_window: int = 1) -> None:
        """Initializes the future severity prediction task.

        Args:
            future_window: Number of future ICU stays to look ahead.

        Raises:
            ValueError: If future_window is less than 1.
        """
        if future_window < 1:
            raise ValueError("future_window must be at least 1")

        self.future_window = future_window

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generates task samples for one patient.

        Args:
            patient: A patient object with ICU stay and event information.

        Returns:
            A list of task samples. Each sample contains the current visit
            features and a multiclass future severity label.
        """
        samples: List[Dict[str, Any]] = []
        patient_stays = patient.get_events(event_type="patient")

        if len(patient_stays) <= self.future_window:
            return []

        for i in range(len(patient_stays) - self.future_window):
            stay = patient_stays[i]
            future_stay = patient_stays[i + self.future_window]

            stay_id = str(getattr(stay, "patientunitstayid", ""))
            future_stay_id = str(getattr(future_stay, "patientunitstayid", ""))

            diagnoses = patient.get_events(
                event_type="diagnosis",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            physical_exams = patient.get_events(
                event_type="physicalexam",
                filters=[("patientunitstayid", "==", stay_id)],
            )
            medications = patient.get_events(
                event_type="medication",
                filters=[("patientunitstayid", "==", stay_id)],
            )

            conditions = [
                getattr(event, "icd9code", "")
                for event in diagnoses
                if getattr(event, "icd9code", None)
            ]
            procedures = [
                getattr(event, "physicalexampath", "")
                for event in physical_exams
                if getattr(event, "physicalexampath", None)
            ]
            drugs = [
                getattr(event, "drugname", "")
                for event in medications
                if getattr(event, "drugname", None)
            ]

            if not conditions or not procedures or not drugs:
                continue

            future_events = (
                patient.get_events(
                    event_type="diagnosis",
                    filters=[("patientunitstayid", "==", future_stay_id)],
                )
                + patient.get_events(
                    event_type="physicalexam",
                    filters=[("patientunitstayid", "==", future_stay_id)],
                )
                + patient.get_events(
                    event_type="medication",
                    filters=[("patientunitstayid", "==", future_stay_id)],
                )
            )

            future_severity = self._get_future_severity_label(
                future_event_count=len(future_events)
            )

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "future_severity": future_severity,
                }
            )

        if samples:
            label_counts = {0: 0, 1: 0, 2: 0}
            for sample in samples:
                label_counts[sample["future_severity"]] += 1

            dominant_label = max(label_counts, key=label_counts.get)
            label_names = {
                0: "low",
                1: "moderate",
                2: "high",
            }

            print(
                "[FutureSeverityPredictionEICU] "
                f"patient={patient.patient_id} "
                f"samples={len(samples)} "
                f"low={label_counts[0]} "
                f"moderate={label_counts[1]} "
                f"high={label_counts[2]} "
                f"overall={label_names[dominant_label]}"
            )
        
        return samples

    @staticmethod
    def _get_future_severity_label(future_event_count: int) -> int:
        """Maps future event count to a multiclass severity label.

        Args:
            future_event_count: Number of events in the future stay.

        Returns:
            An integer severity label:
                0 = low severity
                1 = moderate severity
                2 = high severity
        """
        if future_event_count <= 2:
            return 0
        if future_event_count <= 8:
            return 1
        return 2