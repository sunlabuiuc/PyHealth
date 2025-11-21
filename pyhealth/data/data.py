import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Union, Any

import dask.dataframe as dd
import pandas as pd

@dataclass(frozen=True)
class Event:
    """Event class representing a single clinical event.

    Attributes:
        event_type (str): Type of the clinical event (e.g., 'medication', 'diagnosis')
        timestamp (datetime): When the event occurred
        attr_dict (Mapping[str, any]): Dictionary containing event-specific attributes
    """

    event_type: str
    timestamp: datetime
    attr_dict: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, event_type: str, timestamp: datetime | None = None, **kwargs):
        """Initialize an Event instance.

        Args:
            event_type (str): Type of the clinical event
            timestamp (datetime, optional): When the event occurred.
                If not provided, current time will be used.
            **kwargs: Additional attributes to store in attr_dict
        """
        # Create a mutable copy of kwargs to manipulate
        attr_dict = dict(kwargs)

        # Extract existing attr_dict if provided in kwargs
        if "attr_dict" in attr_dict:
            existing_attr_dict = attr_dict.pop("attr_dict")
            # Merge with remaining kwargs, with kwargs taking precedence
            attr_dict = {**existing_attr_dict, **attr_dict}

        # Set timestamp to current time if not provided
        if timestamp is None:
            timestamp = datetime.now()

        # Use object.__setattr__ since the dataclass is frozen
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "attr_dict", attr_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Create an Event instance from a dictionary.

        Args:
            d (Dict[str, Any]): Dictionary containing event data.
        Returns:
            Event: An instance of the Event class.
        """
        timestamp: datetime = d["timestamp"]
        event_type: str = d["event_type"]
        attr_dict: Dict[str, Any] = {
            k.split("/", 1)[1]: v for k, v in d.items() if k.split("/")[0] == event_type
        }
        return cls(event_type=event_type, timestamp=timestamp, attr_dict=attr_dict)

    def __getitem__(self, key: str) -> Any:
        """Get an attribute by key.

        Args:
            key (str): The key of the attribute to retrieve.

        Returns:
            any: The value of the attribute.
        """
        if key == "timestamp":
            return self.timestamp
        elif key == "event_type":
            return self.event_type
        else:
            return self.attr_dict[key]

    def __contains__(self, key: str) -> bool:
        """Check if an attribute exists by key.

        Args:
            key (str): The key of the attribute to check.

        Returns:
            bool: True if the attribute exists, False otherwise.
        """
        if key == "timestamp" or key == "event_type":
            return True
        return key in self.attr_dict

    def __getattr__(self, key: str) -> Any:
        """Get an attribute using dot notation.

        Args:
            key (str): The key of the attribute to retrieve.

        Returns:
            any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if key == "timestamp" or key == "event_type":
            return getattr(self, key)
        if key in self.attr_dict:
            return self.attr_dict[key]
        raise AttributeError(f"'Event' object has no attribute '{key}'")


class Patient:
    """Patient class representing a sequence of events.

    Attributes:
        patient_id (str): Unique patient identifier.
        data_source (dd.DataFrame): Dask DataFrame containing all events.
    """

    def __init__(self, patient_id: str, data_source: dd.DataFrame) -> None:
        """Initialize a Patient instance.

        Args:
            patient_id (str): Unique patient identifier.
            data_source (dd.DataFrame): DataFrame containing all events.
        """
        self.patient_id = patient_id
        self.data_source = data_source

    def _filter_by_time_range(self, df: dd.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> dd.DataFrame:
        """Filter events by time range using lazy Dask operations."""
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        return df

    def _filter_by_event_type(self, df: dd.DataFrame, event_type: Optional[str]) -> dd.DataFrame:
        """Filter by event type if provided."""
        if event_type:
            df = df[df["event_type"] == event_type]
        return df

    def _apply_attribute_filters(
        self, df: dd.DataFrame, event_type: str, filters: List[tuple]
    ) -> dd.DataFrame:
        """Apply attribute-level filters to the DataFrame."""
        op_map = {
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        mask = None
        for filt in filters:
            if not (isinstance(filt, tuple) and len(filt) == 3):
                raise ValueError(
                    f"Invalid filter format: {filt} (must be tuple of (attr, op, value))"
                )
            attr, op, val = filt
            if op not in op_map:
                raise ValueError(f"Unsupported operator: {op} in filter {filt}")
            col_name = f"{event_type}/{attr}"
            if col_name not in df.columns:
                raise KeyError(f"Column '{col_name}' not found in dataset")
            col = df[col_name]
            condition = op_map[op](col, val)
            mask = condition if mask is None else mask & condition
        if mask is not None:
            df = df[mask]
        return df

    def get_events(
        self,
        event_type: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        filters: Optional[List[tuple]] = None,
        return_df: bool = False,
    ) -> Union[dd.DataFrame, List[Event]]:
        """Get events with optional type and time filters.

        Args:
            event_type (Optional[str]): Type of events to filter.
            start (Optional[datetime]): Start time for filtering events.
            end (Optional[datetime]): End time for filtering events.
            return_df (bool): Whether to return a pandas DataFrame or a list of
                Event objects.
            filters (Optional[List[tuple]]): Additional filters as [(attr, op, value), ...], e.g.:
                [("attr1", "!=", "abnormal"), ("attr2", "!=", 1)]. Filters are applied after type
                and time filters. The logic is "AND" between different filters.

        Returns:
            Union[dd.DataFrame, List[Event]]: Filtered events as a Dask DataFrame
            or a list of Event objects.
        """
        df = self._filter_by_event_type(self.data_source, event_type)
        df = self._filter_by_time_range(df, start, end)

        active_filters = filters or []
        if active_filters:
            assert event_type is not None, "event_type must be provided if filters are provided"
            df = self._apply_attribute_filters(df, event_type, active_filters)

        if return_df:
            return df
        # Dask DataFrames do not expose .to_dict on lazy expressions; compute to pandas first.
        records = df.compute().to_dict("records")
        return [Event.from_dict(d) for d in records]
