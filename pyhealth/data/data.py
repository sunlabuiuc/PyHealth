from collections import OrderedDict
from datetime import datetime
from typing import Optional, List


class Event:
    """Contains information about a single event.

    An event can be a diagnosis, a procedure, a drug or a lab that happened
        in a visit of a patient at a specific time.

    Args:
        code: str, code of the event. E.g., "428.0" for heart failure.
        table: str, name of the table where the event is recorded.
            E.g., "DIAGNOSES_ICD"
        vocabulary: str, vocabulary of the code. E.g., "ICD9CM", "ICD10CM", "NDC".
        visit_id: str, unique identifier of the visit.
        patient_id: str, unique identifier of the patient.
        timestamp: Optional[datetime], timestamp of the event. Defaults to None.
        **attr: optional attributes of the event. Attributes to add to visit as
            key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of event attributes. Each key is an attribute
            name and each value is the attribute's value.
    """

    def __init__(
            self,
            code: str,
            table: str,
            vocabulary: str,
            visit_id: str,
            patient_id: str,
            timestamp: Optional[datetime] = None,
            **attr,
    ):
        assert timestamp is None or isinstance(timestamp, datetime), \
            "timestamp must be a datetime object"
        self.code = code
        self.table = table
        self.vocabulary = vocabulary
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.timestamp = timestamp
        self.attr_dict = dict()
        self.attr_dict.update(attr)

    def __repr__(self):
        return (
            f"Event with {self.vocabulary} code {self.code} from table {self.table}"
        )

    def __str__(self):
        line = f"Event with {self.vocabulary} code {self.code} " \
               f"from table {self.table} " \
               f"at time {self.timestamp}"
        return line


class Visit:
    """Contains information about a single visit.

    A visit is a period of time in which a patient is admitted to a hospital. Each
        visit is associated with a patient and contains a list of different events.

    Args:
        visit_id: str, unique identifier of the visit.
        patient_id: str, unique identifier of the patient.
        encounter_time: Optional[datetime], timestamp of visit's encounter.
            Defaults to None.
        discharge_time: Optional[datetime], timestamp of visit's discharge.
            Defaults to None.
        discharge_status: Optional, patient's status upon discharge.
            E.g., "Alive", "Dead". Defaults to None.
        **attr: optional attributes of the visit. Attributes to add to visit as
            key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of visit attributes. Each key is an attribute
            name and each value is the attribute's value.
        event_list_dict: Dict[str, List[Event]], dictionary of event lists.
            Each key is a table name and each value is a list of events from that
            table ordered by timestamp.
    """

    def __init__(
            self,
            visit_id: str,
            patient_id: str,
            encounter_time: Optional[datetime] = None,
            discharge_time: Optional[datetime] = None,
            discharge_status=None,
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

    def add_event(self, event: Event) -> None:
        """Adds an event to the visit.

        If the event's table is not in the visit's event list dictionary, it is
            added as a new key. The event is then added to the list of events of
            that table.

        As for now, there is no check on the order of the events. The new event
            is simply appended to the list of events.

        Args:
            event: Event, event to add.
        """
        # TODO: ensure that events are sorted by timestamp
        table = event.table
        if table not in self.event_list_dict:
            self.event_list_dict[table] = list()
        self.event_list_dict[table].append(event)

    def get_event_list(self, table: str) -> List[Event]:
        """Returns a list of events from a specific table.

        If the table is not in the visit's event list dictionary, an empty list
            is returned.

        As for now, there is no check on the order of the events. The list of
            events is simply returned as is.

        Args:
            table: str, name of the table.

        Returns:
            List[Event], list of events from the specified table.
        """
        # TODO: ensure that events are sorted by timestamp
        if table in self.event_list_dict:
            return self.event_list_dict[table]
        else:
            return list()

    def get_code_list(
            self,
            table: str,
            remove_duplicate: Optional[bool] = True
    ) -> List[str]:
        """Returns a list of codes from a specific table.

        If the table is not in the visit's event list dictionary, an empty list
            is returned.

        As for now, there is no check on the order of the codes. The list of
            codes is simply returned as is.

        Args:
            table: str, name of the table.
            remove_duplicate: Optional[bool], whether to remove duplicate codes
                (but keep the order). Default is True.

        Returns:
            List[str], list of codes from the specified table.
        """
        # TODO: ensure that codes are sorted by timestamp
        event_list = self.get_event_list(table)
        code_list = [event.code for event in event_list]
        if remove_duplicate:
            # remove duplicate codes but keep the order
            code_list = list(dict.fromkeys(code_list))
        return code_list

    def set_event_list(self, table: str, event_list: List[Event]) -> None:
        """Sets the list of events from a specific table.

        Note that this function will overwrite any existing list of events from
            the specified table.

        As for now, there is no check on the order of the events. The list of
            events is simply set as is.

        Args:
            table: str, name of the table.
            event_list: List[Event], list of events to set.
        """
        # TODO: ensure that events are sorted by timestamp
        self.event_list_dict[table] = event_list

    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the visit.

        Returns:
            List[str], list of available tables.
        """
        return list(self.event_list_dict.keys())

    @property
    def num_events(self) -> int:
        """Returns the total number of events in the visit.

        Returns:
            int, total number of events.
        """
        return sum([len(event_list) for event_list in self.event_list_dict.values()])

    def __repr__(self):
        return f"Visit {self.visit_id} " \
               f"from patient {self.patient_id} " \
               f"with {self.num_events} events " \
               f"from tables {self.available_tables}"

    def __str__(self):
        lines = list()
        lines.append(f"Visit {self.visit_id} from patient {self.patient_id} "
                     f"with {self.num_events} events:")
        lines.append(f"Encounter time: {self.encounter_time}")
        lines.append(f"Discharge time: {self.discharge_time}")
        lines.append(f"Discharge status: {self.discharge_status}")
        lines.append(f"Available tables: {self.available_tables}")
        for k, v in self.attr_dict.items():
            lines.append(f"{k}: {v}")
        lines.append("Events:")
        for table, event_list in self.event_list_dict.items():
            for event in event_list:
                lines.append(f"\t {event}")
        return "\n".join(lines)


class Patient:
    """Contains information about a single patient.

    A patient is a person who is admitted at least once to a hospital. Each patient
        is associated with a list of visits.

    Args:
        patient_id: str, unique identifier of the patient.
        birth_datetime: Optional[datetime], timestamp of patient's birth.
            Defaults to None.
        death_datetime: Optional[datetime], timestamp of patient's death.
            Defaults to None.
        gender: Optional, gender of the patient. E.g., "M", "F".
            Defaults to None.
        ethnicity: Optional, ethnicity of the patient. E.g., "White",
            "Black or African American", "American Indian or Alaska Native",
            "Asian", "Native Hawaiian or Other Pacific Islander".
            Defaults to None.
        attr: optional attributes of the patient. Attributes to add to patient as
            key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of patient attributes. Each key is an attribute
            name and each value is the attribute's value.
        visits: OrderedDict[str, Visit], an ordered dictionary of visits. Each key
            is a visit id and each value is a visit.
        index_to_visit_id: Dict[int, str], dictionary that maps the index of a visit in the
            visits list to the corresponding visit id.
    """

    def __init__(
            self,
            patient_id: str,
            birth_datetime: Optional[datetime] = None,
            death_datetime: Optional[datetime] = None,
            gender=None,
            ethnicity=None,
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
        self.index_to_visit_id = dict()

    def add_visit(self, visit: Visit) -> None:
        """Adds a visit to the patient.

        Note that if the visit's id is already in the patient's visits dictionary,
            it will be overwritten by the new visit.

        As for now, there is no check on the order of the visits. The new visit
            is simply added to the end of the ordered dictionary of visits.

        Args:
            visit: Visit, visit to add.
        """
        self.visits[visit.visit_id] = visit
        # incrementing index
        self.index_to_visit_id[len(self.visits) - 1] = visit.visit_id

    def add_event(self, event: Event) -> None:
        """Adds an event to the patient.

        If the event's visit id is not in the patient's visits dictionary, this
            function will raise KeyError.

        As for now, there is no check on the order of the events. The new event
            is simply appended to the list of events of the corresponding visit.

        Args:
            event: Event, event to add.
        """
        visit_id = event.visit_id
        if visit_id not in self.visits:
            raise KeyError(
                f"Visit with id {visit_id} not found in patient {self.patient_id}"
            )
        self.get_visit_by_id(visit_id).add_event(event)

    def get_visit_by_id(self, visit_id: str) -> Visit:
        """Returns a visit by visit id.

        Args:
            visit_id: str, unique identifier of the visit.

        Returns:
            Visit, visit with the given visit id.
        """
        return self.visits[visit_id]

    def get_visit_by_index(self, index: int) -> Visit:
        """Returns a visit by its index.

        Args:
            index: int, index of the visit to return.

        Returns:
            Visit, visit with the given index.
        """
        if index not in self.index_to_visit_id:
            raise IndexError(
                f"Visit with  index {index} not found in patient {self.patient_id}"
            )
        visit_id = self.index_to_visit_id[index]
        return self.get_visit_by_id(visit_id)

    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the patient.

        Returns:
            List[str], list of available tables.
        """
        tables = []
        for visit in self:
            tables.extend(visit.available_tables)
        return list(set(tables))

    def __repr__(self):
        return f"Patient {self.patient_id} with {len(self)} visits"

    def __str__(self):
        lines = list()
        # patient info
        lines.append(f"Patient {self.patient_id} with {len(self)} visits:")
        lines.append(f"\t Birth datetime: {self.birth_datetime}")
        lines.append(f"\t Death datetime: {self.death_datetime}")
        lines.append(f"\t Gender: {self.gender}")
        lines.append(f"\t Ethnicity: {self.ethnicity}")
        for k, v in self.attr_dict.items():
            lines.append(f"\t {k}: {v}")
        lines.append("")
        # visit info
        for visit in self:
            lines.append(f"{visit}")
            lines.append("")
        return "\n".join(lines)

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, index) -> Visit:
        """Returns a visit by its index."""
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
        table="condition",
        vocabulary="ICD9CM",
        attr="attr",
    )
    print(event)
    print(event.attr_dict)
