from typing import Dict, List
from pyhealth.data import Patient
from pyhealth.tasks import BaseTask


class MIMICCXRLongitudinalClassificationTask(BaseTask):
    """Longitudinal task for HIST-AID: pairing images with report history.

    Args:
        max_history (int): Maximum number of past reports to include.
        **kwargs: Additional keyword arguments for BaseTask.
    """

    def __init__(self, max_history: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_history = max_history

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes patient history into longitudinal multi-modal samples."""
        samples = []
        report_history = []
        # BaseDataset ensures visits are sorted by encounter_time
        visits = sorted(patient.visits, key=lambda v: v.encounter_time)

        for visit in visits:
            img_event = visit.get_event_by_table("metadata")
            label_event = visit.get_event_by_table("chexpert")

            if img_event and label_event:
                samples.append({
                    "patient_id": patient.patient_id,
                    "visit_id": visit.visit_id,
                    "image": img_event[0]["dicom_id"],
                    "history": list(report_history),
                    "label": label_event[0],
                })

            report_event = visit.get_event_by_table("reports")
            if report_event:
                report_history.append(report_event[0]["report_text"])
                if len(report_history) > self.max_history:
                    report_history.pop(0)

        return samples