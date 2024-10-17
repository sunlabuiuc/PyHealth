from collections import OrderedDict
from datetime import datetime
from typing import Optional, List

class Event:
    """Contains information about a single event.

    An event can be anything from a diagnosis to a prescription or a lab test
    that happened in a visit of a patient at a specific time.

    Args:
        code: code of the event. E.g., "428.0" for congestive heart failure.
        table: name of the table where the event is recorded. This corresponds
            to the raw csv file name in the dataset. E.g., "DIAGNOSES_ICD".
        vocabulary: vocabulary of the code. E.g., "ICD9CM" for ICD-9 diagnosis codes.
        visit_id: unique identifier of the visit.
        patient_id: unique identifier of the patient.
        timestamp: timestamp of the event. Default is None.
        event_type: type of the event.
        item_id: unique identifier of the item.
        **attr: optional attributes to add to the event as key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of event attributes. Each key is an attribute
            name and each value is the attribute's value.
    """

    def __init__(
        self,
        code: str = None,
        table: str = None,
        vocabulary: str = None,
        visit_id: str = None,
        patient_id: str = None,
        timestamp: Optional[datetime] = None,
        item_id: str = None,
        **attr,
    ):
        assert timestamp is None or isinstance(
            timestamp, datetime
        ), "timestamp must be a datetime object"
        self.code = code 
        self.table = table
        self.vocabulary = vocabulary
        self.visit_id = visit_id # we can remove the explicity of it if need be.
        self.patient_id = patient_id
        self.timestamp = timestamp
        self.item_id = item_id
        self.attr_dict = dict() # Event(...,visit_id=), don't make it explicit 
        self.attr_dict.update(attr)

    def __repr__(self):
        return f"Event with {self.vocabulary} code {self.code} from table {self.table}"

    def __str__(self):
        lines = list()
        lines.append(f"Event from patient {self.patient_id} visit {self.visit_id}:")
        lines.append(f"\t- Code: {self.code}")
        lines.append(f"\t- Table: {self.table}")
        lines.append(f"\t- Vocabulary: {self.vocabulary}")
        lines.append(f"\t- Timestamp: {self.timestamp}")
        lines.append(f"\t- Item ID: {self.item_id}")
        for k, v in self.attr_dict.items():
            lines.append(f"\t- {k}: {v}")
        return "\n".join(lines)

class Patient:
    """Contains information about a single patient.

    A patient is a person who is admitted at least once to a hospital or
    a specific department. Each patient is associated with a list of events.

    Args:
        patient_id: unique identifier of the patient.
        birth_datetime: timestamp of patient's birth. Default is None.
        death_datetime: timestamp of patient's death. Default is None.
        gender: gender of the patient. Default is None.
        ethnicity: ethnicity of the patient. Default is None.
        **attr: optional attributes to add to the patient as key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of patient attributes. Each key is an attribute
            name and each value is the attribute's value.
        events: Dict[str, List[Event]], dictionary of event lists.
            Each key is a table name and each value is a list of events from that table.
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
        self.events = [] # Nested Dataframe -> PyArrow?

    def add_event(self, event: Event) -> None:
        """Adds an event to the patient.

        If the event's table is not in the patient's event dictionary, it is
        added as a new key. The event is then added to the list of events of
        that table.

        Args:
            event: event to add.
        """
        assert event.patient_id == self.patient_id, "patient_id unmatched"
        # table = event.table
        # if table not in self.events:
        #     self.events[table] = list()
        # self.events[table].append(event)
        self.events.append(event) # 

    def get_event_list(self, table: str) -> List[Event]:
        """Returns a list of events from a specific table.

        If the table is not in the patient's event dictionary, an empty list
        is returned.

        Args:
            table: name of the table.

        Returns:
           List of events from the specified table.
        """
        return [event for event in self.events if event.table == table]

    def get_code_list(
        self, table: str, remove_duplicate: Optional[bool] = True
    ) -> List[str]:
        """Returns a list of codes from a specific table.

        If the table is not in the patient's event dictionary, an empty list
        is returned.

        Args:
            table: name of the table.
            remove_duplicate: whether to remove duplicate codes
                (but keep the relative order). Default is True.

        Returns:
            List of codes from the specified table.
        """
        event_list = self.get_event_list(table)
        code_list = [event.code for event in event_list]
        if remove_duplicate:
            # remove duplicate codes but keep the order
            code_list = list(dict.fromkeys(code_list))
        return code_list

    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the patient.

        Returns:
            List of available tables.
        """
        tables = set()
        for event in self.events:
            tables.add(event.table)
        return list(tables)

    @property
    def num_events(self) -> int:
        """Returns the total number of events for the patient.

        Returns:
            Total number of events.
        """
        return len(self.events)

    def __repr__(self):
        return f"Patient {self.patient_id} with {self.num_events} events"

    def __str__(self):
        lines = list()
        lines.append(f"Patient {self.patient_id} with {self.num_events} events:")
        lines.append(f"\t- Birth datetime: {self.birth_datetime}")
        lines.append(f"\t- Death datetime: {self.death_datetime}")
        lines.append(f"\t- Gender: {self.gender}")
        lines.append(f"\t- Ethnicity: {self.ethnicity}")
        lines.append(f"\t- Available tables: {self.available_tables}")
        for k, v in self.attr_dict.items():
            lines.append(f"\t- {k}: {v}")
        for event in self.events:
           
            event_str = str(event).replace("\n", "\n\t")
            lines.append(f"\t- {event_str}")
        return "\n".join(lines)