from collections import OrderedDict
from datetime import datetime
from typing import Optional


class Event:
    """Contains information about a single event.

    An event can be a diagnosis, a procedure, a drug, a lab that happened to a patient in a visit at a specific time.

    Args:
        code: str, code of the event (e.g., "428.0" for heart failure).
        event_type: str, type of the event. This corresponds to the table name in the raw data (e.g., "DIAGNOSES_ICD").
        vocabulary: str, vocabulary of the code (e.g., 'ICD9CM', 'ICD10CM', 'NDC').
        visit_id: str, unique identifier of the visit.
        patient_id: str, unique identifier of the patient.
        timestamp: Optional[datetime], timestamp of the event. Defaults to None.
        **attr: optional attributes of the event. Attributes to add to visit as key=value pairs.

    Attributes:
        attr_dict: dict, dictionary of event attributes. Each key is an attribute name and each value is
            the attribute's value.
    """

    def __init__(
        self,
        code: str,
        event_type: str,
        vocabulary: str,
        visit_id: str,
        patient_id: str,
        timestamp: Optional[datetime] = None,
        **attr,
    ):
        self.code = code
        self.event_type = event_type
        self.vocabulary = vocabulary
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.timestamp = timestamp
        self.attr_dict = dict()
        self.attr_dict.update(attr)

    def __str__(self):
        return (
            f"Event of type {self.event_type} with {self.vocabulary} code {self.code}"
        )


class Visit:
    """Contains information about a single visit.

    A visit is a period of time in which a patient is admitted to a hospital.
    Each visit is associated with a patient and contains a list of different events.

    Args:
        visit_id: str, unique identifier of the visit.
        patient_id: str, unique identifier of the patient.
        encounter_time: Optional[datetime], timestamp of visit's encounter. Defaults to None.
        discharge_time: Optional[datetime], timestamp of visit's discharge. Defaults to None.
        discharge_status: Optional[str], patient's status upon discharge. E.g., "Alive", "Dead". Defaults to None.
        **attr, optional attributes of the visit. Attributes to add to visit as key=value pairs.

    Attributes:
        attr_dict: dict, dictionary of visit attributes. Each key is an attribute name and each value is
            the attribute's value.
        event_list_dict: dict, dictionary of event lists. Each key is an event type and each value is a list of
            events of that type ordered by timestamp.
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

    def add_event(self, event: Event):
        """Adds an event to the visit.

        Args:
            event: Event, event to add.
        """
        event_type = event.event_type
        if event_type not in self.event_list_dict:
            self.event_list_dict[event_type] = list()
        self.event_list_dict[event_type].append(event)

    def get_event_list(self, event_type: str) -> list:
        """Returns a list of events of a specific type.

        If no events of that type are found, returns an empty list.

        Args:
            event_type: str, type of events to return.

        Returns:
            list, list of events of the specified type.
        """
        # TODO: ensure that events are sorted by timestamp
        if event_type in self.event_list_dict:
            return self.event_list_dict[event_type]
        else:
            return list()

    @property
    def event_types(self) -> list:
        """Returns a list of event types in the visit.

        Returns:
            list, list of event types in the visit.
        """
        return list(self.event_list_dict.keys())

    def __str__(self):
        return f"Visit {self.visit_id} with {len(self.event_types)} types of events"


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
        attr: optional attributes of the patient. Attributes to add to patient as key=value pairs.

    Attributes:
        attr_dict: dict, dictionary of patient attributes. Each key is an attribute name and each value is
            the attribute's value.
        visits: OrderedDict[str, Visit], an ordered dictionary of visits. Each key is a visit id and each value is
            a visit.
        index_to_visit: dict, dictionary that maps the index of a visit in the visits list to the visit id.
    """

    def __init__(
        self,
        patient_id: str,
        birth_datetime: Optional[datetime] = None,
        death_datetime: Optional[datetime] = None,
        gender: Optional[str] = None,
        ethnicity: Optional[str] = None,
        **attr,
    ):
        self.patient_id = patient_id
        self.birth_datetime = birth_datetime
        self.death_datetime = death_datetime
        self.gender = gender
        self.ethnicity = ethnicity
        self.attr_dict = dict()
        self.attr_dict.update(attr)
        self.visits = OrderedDict()
        self.index_to_visit = dict()

    def add_visit(self, visit: Visit):
        """Adds a visit to the patient.

        Args:
            visit: Visit, visit to add.
        """
        self.visits[visit.visit_id] = visit
        self.index_to_visit[len(self.visits) - 1] = visit.visit_id  # incrementing index

    def add_event(self, event: Event):
        """Adds an event to the patient.

        Args:
            event: Event, event to add.
        """
        visit_id = event.visit_id
        if visit_id not in self.visits:
            raise KeyError(f"Visit {visit_id} not found in patient {self.patient_id}")
        self.get_visit_by_id(visit_id).add_event(event)

    def get_visit_by_id(self, visit_id: str):
        """Returns a visit by visit id.

        Args:
            visit_id: str, unique identifier of the visit.

        Returns:
            Visit, visit with the given visit id.
        """
        return self.visits[visit_id]

    def get_visit_by_index(self, index: int):
        """Returns a visit by its index.

        Args:
            index: int, index of the visit to return.

        Returns:
            Visit, visit with the given index.
        """
        if index not in self.index_to_visit:
            raise IndexError(
                f"Visit index {index} not found in patient {self.patient_id}"
            )
        visit_id = self.index_to_visit[index]
        return self.get_visit_by_id(visit_id)

    def __str__(self):
        return f"Patient {self.patient_id} with {len(self)} visits"

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, index):
        return self.get_visit_by_index(index)


if __name__ == "__main__":
    patient = Patient(patient_id="1", attr="attr")
    print(patient)
    print(patient.attr_dict)
    for v in patient:
        print(v)

    visit = Visit(visit_id="1", patient_id="1", attr="attr")
    print(visit)
    print(visit.attr_dict)

    event = Event(
        code="428.0",
        visit_id="1",
        patient_id="1",
        event_type="condition",
        vocabulary="ICD9CM",
        attr="attr",
    )
    print(event)
    print(event.attr_dict)
