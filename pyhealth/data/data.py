from typing import List, Optional


class Visit:
    """ Contains information about a single visit """

    def __init__(
            self,
            visit_id: str,
            patient_id: str,
            conditions: List[str],
            procedures: List[str],
            drugs: List[str],
            labs: List[str] = None,
            physicalExams: List[str] = None,
            visit_info: Optional[dict] = None
    ):
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.conditions = conditions
        self.procedures = procedures
        self.labs = labs
        self.physicalExams = physicalExams
        self.drugs = drugs
        self.visit_info = visit_info

    def __str__(self):
        return f"Visit {self.visit_id} of patient {self.patient_id}"


class Patient:
    """ Contains information about a single patient """

    def __init__(
            self,
            patient_id: str,
            visits: List[Visit],
            patient_info: Optional[dict] = None
    ):
        self.patient_id = patient_id
        self.visits = visits
        self.patient_info = patient_info

    def __str__(self):
        return f"Patient {self.patient_id}"
