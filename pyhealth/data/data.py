from datetime import datetime
from typing import List, Optional


class Event:
    """Contains information about a single event.

    An event can be a lab test, a procedure, a drug, or a diagnosis happened to a patient during a hospital visit
    at a specific time.

    Args:
        code: str, code of the event (e.g., "428.0" for heart failure)
        event_type: str, type of the event (e.g., 'condition', 'procedure', 'drug', 'lab')
        vocabulary: str, vocabulary of the code (e.g., 'ICD9CM', 'ICD10CM', 'NDC')
        timestamp: Optional[datetime], timestamp of the event. Defaults to None.
        **attr, optional attributes of the event. Attributes to add to visit as key=value pairs.

    Attributes:
        attr_dict: dict, dictionary of event attributes. Each key is an attribute name and each value is
            the attribute's value.
    """

    def __init__(
            self,
            code: str,
            event_type: str,
            vocabulary: str,
            timestamp: Optional[datetime] = None,
            **attr,
    ):
        self.code = code
        self.event_type = event_type
        self.vocabulary = vocabulary
        self.timestamp = timestamp
        self.attr_dict = dict()
        self.attr_dict.update(attr)

    def __str__(self):
        return f"Event {self.code} of type {self.event_type}"


class Visit:
    """Contains information about a single visit.

    A visit is a period of time in which a patient is admitted to a hospital.
    Each visit is associated with a patient and contains a list of different events.

    Args:
        visit_id: str, unique identifier of the visit
        patient_id: str, unique identifier of the patient
        encounter_time: Optional[datetime], timestamp of visit's encounter. Defaults to None.
        discharge_time: Optional[datetime], timestamp of visit's discharge. Defaults to None.
        discharge_status: Optional[str], patient's status upon discharge. E.g., "Alive", "Dead". Defaults to None.
        **attr, optional attributes of the visit. Attributes to add to visit as key=value pairs.

    Attributes:
        attr_dict: dict, dictionary of visit attributes. Each key is an attribute name and each value is
            the attribute's value.
        event_list_dict: dict, dictionary of event lists. Each key is an event type and each value is a list of
            Event objects of that type ordered by timestamp.
    """

    def __init__(
            self,
            visit_id: str,
            patient_id: str,
            encounter_time: Optional[datetime] = None,
            discharge_time: Optional[datetime] = None,
            discharge_status: Optional[str] = None,
            **attr,
    ):
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.encounter_time = encounter_time
        self.discharge_time = discharge_time
        self.discharge_status = discharge_status
        self.attr_dict = dict()
        self.attr_dict.update(attr)
        self.event_list_dict = dict()

    def add_event(self, event_type: str, event: Event):
        """Adds an event to the visit.

        Args:
            event_type: str, type of event to add
            event: Event, event to add
        """
        if event_type not in self.event_list_dict:
            self.event_list_dict[event_type] = list()
        self.event_list_dict[event_type].append(event)

    def __str__(self):
        return f"Visit {self.visit_id} of patient {self.patient_id}"


class Patient:
    """Contains information about a single patient.

    A patient is a person who is admitted at least once to a hospital.
    Each patient is associated with a list of visits.

    Args:
        patient_id: str, unique identifier of the patient.
        birth_datetime: Optional[datetime], timestamp of patient's birth. Defaults to None.
        death_datetime: Optional[datetime], timestamp of patient's death. Defaults to None.
        gender: Optional[str], gender of the patient. E.g., "M", "F", "Unknown". Defaults to None.
        ethnicity: Optional[str], ethnicity of the patient. E.g., "White", "Black or African American",
            "American Indian or Alaska Native", "Asian", "Native Hawaiian or Other Pacific Islander".
            Defaults to None.
        visits: Optional[List[Visit]], list of Visit objects, ordered by visit's encounter time.
            Defaults to empty list.
        attr: optional attributes of the patient. Attributes to add to patient as key=value pairs.

    Attributes:
        attr_dict: dict, dictionary of patient attributes. Each key is an attribute name and each value is
            the attribute's value.
    """

    def __init__(
            self,
            patient_id: str,
            birth_datetime: Optional[datetime] = None,
            death_datetime: Optional[datetime] = None,
            gender: Optional[str] = None,
            ethnicity: Optional[str] = None,
            visits: Optional[List[Visit]] = None,
            **attr,
    ):
        if visits is None:
            visits = list()
        self.patient_id = patient_id
        self.birth_datetime = birth_datetime
        self.death_datetime = death_datetime
        self.gender = gender
        self.ethnicity = ethnicity
        self.visits = visits
        self.attr_dict = dict()
        self.attr_dict.update(attr)

    def __str__(self):
        return f"Patient {self.patient_id} with {len(self.visits)} visit(s)"


if __name__ == "__main__":
    patient = Patient(patient_id="1", attr="attr")
    print(patient)
    print(patient.attr_dict)
    visit = Visit(visit_id="1", patient_id="1", attr="attr")
    print(visit)
    print(visit.attr_dict)
    event = Event(code="428.0", event_type="condition", vocabulary="ICD9CM", attr="attr")
    print(event)
    print(event.attr_dict)
