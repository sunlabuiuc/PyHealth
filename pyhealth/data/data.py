from typing import List, Optional


# TODO: domain and event type are used interchangeably, should be consistent
# TODO: conditions and condition are used interchangeably, should be consistent


class Event:
    """Contains information about a single event.

    For example, a single event can be a lab test, a procedure, a drug, or a condition.
    """

    def __init__(
            self,
            code: str,
            domain: str,
            vocabulary: str,
            timestamp: float,
    ):
        """
        Args:
            code: str, code of the event
            domain: str, domain of the code (e.g., 'condition', 'procedure', 'drug', 'lab')
            vocabulary: str, vocabulary of the code (e.g., 'ICD9', 'ICD10', 'NDC')
            timestamp: float, timestamp of the event
        """
        self.code = code
        self.domain = domain
        self.vocabulary = vocabulary
        self.timestamp = timestamp

    def __str__(self):
        return f"{self.code} at {self.timestamp}"


class Visit:
    """Contains information about a single visit.

    A visit is a period of time in which a patient is admitted to a hospital.
    Each visit is associated with a patient and contains a list of different events.
    """

    def __init__(
            self,
            visit_id: str,
            patient_id: str,
            encounter_time: float,
            discharge_time: float,
            mortality_status: bool,
            conditions: Optional[List[Event]] = None,
            procedures: Optional[List[Event]] = None,
            drugs: Optional[List[Event]] = None,
            labs: Optional[List[Event]] = None,
    ):
        """
        Args:
            visit_id: str, unique identifier of the visit
            patient_id: str, unique identifier of the patient
            encounter_time: float, encounter timestamp of the visit
            discharge_time: float, discharge timestamp of the visit
            mortality_status: bool, whether the patient died during the visit
            conditions: List[Event], list of condition events
            procedures: List[Event], list of procedure events
            drugs: List[Event], list of drug events
            labs: List[Event], list of lab events
        """

        self.visit_id = visit_id
        self.patient_id = patient_id
        self.encounter_time = encounter_time
        self.discharge_time = discharge_time
        self.mortality_status = mortality_status
        self.conditions = conditions if conditions is not None else []
        self.procedures = procedures if procedures is not None else []
        self.drugs = drugs if drugs is not None else []
        self.labs = labs if labs is not None else []

    def __str__(self):
        return f"Visit {self.visit_id} of patient {self.patient_id}"


class Patient:
    """Contains information about a single patient.

    A patient is a person who is admitted at least once to a hospital. Each patient is
    associated with a list of visits.
    """

    def __init__(
            self,
            patient_id: str,
            birth_date: float,
            death_date: float,
            mortality_status: bool,
            gender: str,
            ethnicity: str,
            visit_ids: List[str],
    ):
        """
        Args:
            patient_id: str, unique identifier of the patient
            birth_date: float, timestamp of the birth date
            death_date: float, timestamp of the death date
            mortality_status: bool, whether the patient died
            gender: str, gender of the patient
            ethnicity: str, ethnicity of the patient
            visit_ids: List[str], list of visit_ids
        """

        self.patient_id = patient_id
        self.birth_date = birth_date
        self.death_date = death_date
        self.mortality_status = mortality_status
        self.gender = gender
        self.ethnicity = ethnicity
        self.visit_ids = visit_ids

    def __str__(self):
        return f"Patient {self.patient_id}"
