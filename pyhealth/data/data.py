from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class Event:
    """Contains information about a single event.

    An event can be anything from a diagnosis to a prescription or lab test
    that happened for a patient at a specific time.

    Args:
        type: type of the event (e.g., "diagnosis", "prescription", "lab_test").
        timestamp: timestamp of the event.
        attr_dict: event attributes as a dictionary.
    """
    type: str
    timestamp: Optional[datetime] = None
    attr_dict: Dict[str, any] = field(default_factory=dict)


@dataclass
class Patient:
    """Contains information about a single patient and their events.

    A patient is a person who has a sequence of events over time, each associated
    with specific health data.

    Args:
        patient_id: unique identifier for the patient.
        attr_dict: patient attributes as a dictionary.
        events: list of events for the patient.
    """
    patient_id: str
    attr_dict: Dict[str, any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        """Adds an event to the patient's event sequence, maintaining order by event_time.

        Events without a timestamp are placed at the end of the list.
        """
        self.events.append(event)
        # Sort events, placing those with None timestamps at the end
        self.events.sort(key=lambda e: (e.timestamp is None, e.timestamp))

    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Retrieve events of a specific type."""
        return [event for event in self.events if event.type == event_type]
