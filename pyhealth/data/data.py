from typing import List, Optional


class Event:
    """Contains information about a single event"""

    def __init__(
        self,
        code: str,
        time: float,
    ):
        self.code = code
        self.time = time

    def __str__(self):
        return f"{self.code} at {self.time}"


class Visit:
    """Contains information about a single visit"""

    def __init__(
        self,
        visit_id: str,
        patient_id: str,
        encounter_time: float = 0.0,
        duration: float = 0.0,
        mortality_status: bool = False,
        conditions: List[Event] = [],
        procedures: List[Event] = [],
        drugs: List[Event] = [],
        labs: List[Event] = [],
        physicalExams: List[Event] = [],
    ):
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.encounter_time = encounter_time
        self.duration = duration
        self.mortality_status = mortality_status
        self.conditions = conditions
        self.procedures = procedures
        self.labs = labs
        self.physicalExams = physicalExams
        self.drugs = drugs

    def __str__(self):
        return f"Visit {self.visit_id} of patient {self.patient_id}"


class Patient:
    """Contains information about a single patient"""

    def __init__(
        self,
        patient_id: str,
        visits=None,
    ):
        self.patient_id = patient_id
        self.visits = visits

    def __str__(self):
        return f"Patient {self.patient_id}"
