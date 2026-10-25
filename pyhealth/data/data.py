import operator
from dataclasses import dataclass, field
from datetime import datetime
from functools import reduce
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Event:
    """Event class representing a single clinical event.

    Attributes:
        event_type (str): Type of the clinical event (e.g., 'medication', 'diagnosis')
        timestamp (Optional[datetime]): When the event occurred, or ``None`` if
            the event has no time (e.g. rows of a demographics table such as
            MIMIC's ``patients``)
        attr_dict (Mapping[str, any]): Dictionary containing event-specific attributes

    Examples:
        >>> from datetime import datetime
        >>> from pyhealth.data import Event
        >>> event = Event("admissions", timestamp=datetime(2164, 10, 23), hadm_id="1")
        >>> event.hadm_id
        '1'
        >>> Event("patients", gender="F").timestamp is None
        True
    """

    event_type: str
    timestamp: datetime | None
    attr_dict: Mapping[str, any] = field(default_factory=dict)

    def __init__(
        self, event_type: str, timestamp: datetime | None = None, **kwargs
    ):
        """Initialize an Event instance.

        Args:
            event_type (str): Type of the clinical event
            timestamp (datetime, optional): When the event occurred. Defaults to
                ``None``, meaning the event has no time.
            **kwargs: Additional attributes to store in attr_dict
        """
        # Create a mutable copy of kwargs to manipulate
        attr_dict = dict(kwargs)

        # Extract existing attr_dict if provided in kwargs
        if "attr_dict" in attr_dict:
            existing_attr_dict = attr_dict.pop("attr_dict")
            # Merge with remaining kwargs, with kwargs taking precedence
            attr_dict = {**existing_attr_dict, **attr_dict}

        # Use object.__setattr__ since the dataclass is frozen
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "attr_dict", attr_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, any]) -> "Event":
        """Create an Event instance from a dictionary.

        Args:
            d (Dict[str, any]): Dictionary containing event data.

        Returns:
            Event: An instance of the Event class.
        """
        timestamp: datetime = d["timestamp"]
        event_type: str = d["event_type"]
        attr_dict: Dict[str, any] = {
            k.split("/", 1)[1]: v for k, v in d.items() if k.split("/")[0] == event_type
        }
        return cls(event_type=event_type, timestamp=timestamp, attr_dict=attr_dict)

    def __getitem__(self, key: str) -> any:
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

    def __getattr__(self, key: str) -> any:
        """Get an attribute using dot notation.

        Args:
            key (str): The key of the attribute to retrieve.

        Returns:
            any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        # Only called when normal lookup fails. copy and pickle build the object
        # without __init__ and probe for dunder methods before attr_dict exists,
        # so never touch self.attr_dict here (it would recurse forever).
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        attr_dict = self.__dict__.get("attr_dict")
        if attr_dict is not None and key in attr_dict:
            return attr_dict[key]
        raise AttributeError(f"'Event' object has no attribute '{key}'")


class Patient:
    """Patient class representing a sequence of events.

    Events are ordered by timestamp. Events that share a timestamp are grouped
    by table, sorted by ``event_type``, and within a table keep the order of
    their rows in the source file (for example all diagnoses of a MIMIC
    admission, which carry its discharge time), using the ``_source_row``
    column that dataset loading adds. The order is the same on every cache
    build, except among rows that a ``join`` in the dataset config copies
    (one-to-many) or adds without a match (``right`` or ``outer``).

    Attributes:
        patient_id (str): Unique patient identifier.
        data_source (pl.DataFrame): DataFrame containing all events, sorted by timestamp.
        event_type_partitions (Dict[str, pl.DataFrame]): Dictionary mapping event types to their respective DataFrame partitions.

    Examples:
        >>> from datetime import datetime
        >>> import polars as pl
        >>> from pyhealth.data import Patient
        >>> t = datetime(2020, 1, 1)
        >>> events = pl.DataFrame({
        ...     "patient_id": ["p1"] * 3,
        ...     "event_type": ["diagnoses"] * 3,
        ...     "timestamp": [t, t, t],
        ...     "_source_row": [2, 0, 1],
        ...     "diagnoses/code": ["c", "a", "b"],
        ... })
        >>> patient = Patient(patient_id="p1", data_source=events)
        >>> [e.code for e in patient.get_events(event_type="diagnoses")]
        ['a', 'b', 'c']
    """

    def __init__(self, patient_id: str, data_source: pl.DataFrame) -> None:
        """
        Initialize a Patient instance.

        Args:
            patient_id (str): Unique patient identifier.
            data_source (pl.DataFrame): DataFrame containing all events.
        """
        self.patient_id = patient_id
        # Events that share a timestamp are ordered by table and then by their
        # row in the source file, so the order does not depend on how the cache
        # was built. (event_type, _source_row) identifies a source row, so this
        # sort needs no stability (copies made by a one-to-many join share a
        # row and have no fixed order). Older caches lack the source-row
        # column; there the existing row order is kept, which needs a stable
        # sort.
        if "_source_row" in data_source.columns:
            self.data_source = data_source.sort(
                ["timestamp", "event_type", "_source_row"]
            )
        else:
            self.data_source = data_source.sort("timestamp", maintain_order=True)
        self.event_type_partitions = self.data_source.partition_by("event_type", maintain_order=True, as_dict=True)

    def _filter_by_time_range_regular(self, df: pl.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> pl.DataFrame:
        """Regular filtering by time. Time complexity: O(n)."""
        if start is not None:
            df = df.filter(pl.col("timestamp") >= start)
        if end is not None:
            df = df.filter(pl.col("timestamp") <= end)
        return df

    def _filter_by_time_range_fast(self, df: pl.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> pl.DataFrame:
        """Fast filtering by time using binary search on sorted timestamps. Time complexity: O(log n)."""
        if start is None and end is None:
            return df
        df = df.filter(pl.col("timestamp").is_not_null())
        ts_col = df["timestamp"].to_numpy()
        start_idx = 0
        end_idx = len(ts_col)
        if start is not None:
            start_idx = np.searchsorted(ts_col, np.datetime64(start, "ms"), side="left")
        if end is not None:
            end_idx = np.searchsorted(ts_col, np.datetime64(end, "ms"), side="right")
        return df.slice(start_idx, end_idx - start_idx)

    def _filter_by_event_type_regular(self, df: pl.DataFrame, event_type: Optional[str]) -> pl.DataFrame:
        """Regular filtering by event type. Time complexity: O(n)."""
        if event_type:
            df = df.filter(pl.col("event_type") == event_type)
        return df

    def _filter_by_event_type_fast(self, df: pl.DataFrame, event_type: Optional[str]) -> pl.DataFrame:
        """Fast filtering by event type using pre-built event type index. Time complexity: O(1)."""
        if event_type:
            return self.event_type_partitions.get((event_type,), df[:0])
        else:
            return df

    def get_events(
        self,
        event_type: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        filters: Optional[List[tuple]] = None,
        return_df: bool = False,
    ) -> Union[pl.DataFrame, List[Event]]:
        """Get events with optional type and time filters.

        Args:
            event_type (Optional[str]): Type of events to filter.
            start (Optional[datetime]): Start time for filtering events.
            end (Optional[datetime]): End time for filtering events.
            return_df (bool): Whether to return a DataFrame or a list of
                Event objects.
            filters (Optional[List[tuple]]): Additional filters as [(attr, op, value), ...], e.g.:
                [("attr1", "!=", "abnormal"), ("attr2", "!=", 1)]. Filters are applied after type
                and time filters. The logic is "AND" between different filters.

        Returns:
            Union[pl.DataFrame, List[Event]]: Filtered events as a DataFrame
            or a list of Event objects.
        """
        # faster filtering (by default)
        df = self._filter_by_event_type_fast(self.data_source, event_type)
        df = self._filter_by_time_range_fast(df, start, end)

        # regular filtering (commented out by default)
        # df = self._filter_by_event_type_regular(self.data_source, event_type)
        # df = self._filter_by_time_range_regular(df, start, end)

        if filters:
            assert event_type is not None, "event_type must be provided if filters are provided"
        else:
            filters = []
        exprs = []
        for filt in filters:
            if not (isinstance(filt, tuple) and len(filt) == 3):
                raise ValueError(
                    f"Invalid filter format: {filt} (must be tuple of (attr, op, value))"
                )
            attr, op, val = filt
            col_expr = pl.col(f"{event_type}/{attr}")
            # Build operator expression
            if op == "==":
                exprs.append(col_expr == val)
            elif op == "!=":
                exprs.append(col_expr != val)
            elif op == "<":
                exprs.append(col_expr < val)
            elif op == "<=":
                exprs.append(col_expr <= val)
            elif op == ">":
                exprs.append(col_expr > val)
            elif op == ">=":
                exprs.append(col_expr >= val)
            else:
                raise ValueError(f"Unsupported operator: {op} in filter {filt}")
        if exprs:
            df = df.filter(reduce(operator.and_, exprs))
        if return_df:
            return df
        return [Event.from_dict(d) for d in df.to_dicts()]
