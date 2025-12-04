from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl
import copy

from .base_task import BaseTask


class TextExtractionMIMIC4(BaseTask):
    """Task for extracting text from MIMIC-IV EHR tables.

    This class extracts structured text from MIMIC-IV EHR tables (e.g.,
    labevents, prescriptions) and formats it for use with embedding models.
    The extracted text can be tokenized and fed to language models for
    generating embeddings or other downstream tasks.

    The class supports configurable field extraction and event filtering:
    - Field extraction: Select which fields to include in the output text
    - Event filtering: Include/exclude events based on field values

    This task is typically used with MIMIC4Dataset to extract text samples
    from patient records. The extracted samples can then be processed with
    embedding models (e.g., BioBERT, ClinicalBERT) for representation learning.

    Attributes:
        task_name (str): The name of the task, set to "TextExtractionMIMIC4".
        input_schema (Dict[str, str]): Schema defining input data structure.
            Contains "text" key with "text" type.
        output_schema (Dict[str, str]): Schema defining output data structure.
            Contains "text" (extracted text) and "event_type" (metadata).
        DEFAULT_TABLE_CONFIG (Dict[str, Dict[str, Any]]): Default configuration
            for table processing, including field extraction and filtering rules.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import TextExtractionMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="path/to/mimic-iv",
        ...     ehr_tables=["admissions", "labevents", "prescriptions"],
        ... )
        >>> task = TextExtractionMIMIC4(max_patients=100)
        >>> sample_dataset = dataset.set_task(task)
        >>> # Each sample contains: patient_id, visit_id, event_type, text
    """

    # BaseTask required attributes
    task_name: str = "TextExtractionMIMIC4"
    input_schema: Dict[str, str] = {
        "text": "text",
    }
    output_schema: Dict[str, str] = {
        "text": "text",
        # Use "raw" for metadata that doesn't need processing
        "event_type": "raw",
    }

    # Default table configuration
    # Each table config contains:
    #   - extract_fields: List of field names (str) for text extraction
    #     These are the fields that will be extracted and included in the output text
    #     Fields are included only if their value is not None
    #   - filters: Dict with "includes" and/or "excludes" lists for event filtering
    #     Each include/exclude is a dict with:
    #       - "field": field name to check (e.g., "category", "label", "drug")
    #       - "terms": list of terms to match (exact match for includes, substring match for excludes)
    #     Includes use OR logic (keep if any include rule matches)
    #     Excludes use OR logic (exclude if any exclude rule matches)
    DEFAULT_TABLE_CONFIG = {
        "labevents": {
            "extract_fields": [
                "label",
                "itemid",
                "category",
                "fluid",
                "value",
                "valuenum",
                "valueuom",
                "flag",
            ],
            "filters": {
                "includes": [
                    {"field": "category", "terms": ["Blood Gas"]},
                    {
                        "field": "label",
                        "terms": ["C-Reactive Protein", "High-Sensitivity CRP"],
                    },
                ]
            },
        },
        "prescriptions": {
            "extract_fields": [
                "drug",
                "prod_strength",
                "dose_val_rx",
                "dose_unit_rx",
                "route",
            ],
            "filters": {
                "excludes": [
                    {"field": "drug", "terms": ["tobramycin", "coq10"]},
                ]
            },
        },
    }

    def __init__(
        self,
        max_patients: Optional[int] = None,
        table_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the TextExtractionMIMIC4 task.

        Called when creating a new instance of TextExtractionMIMIC4. If
        table_config is provided, it replaces the default configuration entirely.
        If None, uses the default configuration.

        Args:
            max_patients (Optional[int]): Maximum number of patients to process.
                If None, processes all patients. Used to limit dataset size for
                development or testing purposes.
            table_config (Optional[Dict[str, Dict[str, Any]]]): Dictionary
                mapping table names (e.g., "labevents", "prescriptions") to
                their configuration dictionaries. Each table config should have:
                - "extract_fields": List[str] of field names to extract.
                  Fields are included in the output only if their value is not None.
                - "filters": Dict with optional "includes" and/or "excludes"
                  lists for event filtering. Each include/exclude rule is a dict
                  with:
                  - "field": str, field name to check (e.g., "category", "label")
                  - "terms": List[str], list of terms to match
                  Includes use OR logic (keep if any include rule matches).
                  Excludes use OR logic (exclude if any exclude rule matches).
                If None, uses DEFAULT_TABLE_CONFIG. If provided, replaces the
                default configuration entirely.

        Returns:
            None. Initializes the task instance with the provided or default configuration.
        """
        self.max_patients = max_patients

        if table_config is not None:
            # Replace default config entirely if table_config is provided
            self.table_config = copy.deepcopy(table_config)
        else:
            # Use default config if no table_config provided
            self.table_config = copy.deepcopy(self.DEFAULT_TABLE_CONFIG)

    def _get_field_value(self, event: Any, rule: Dict[str, Any]) -> Optional[Any]:
        """Extract field value from event based on rule configuration.

        Helper method to extract field_name and terms from a filter rule and
        get the corresponding value from the event object.

        Args:
            event (Any): Event object with attributes to check.
            rule (Dict[str, Any]): Filter rule dict with "field" and "terms" keys.

        Returns:
            Optional[Any]: Field value if rule is valid and field exists,
                None otherwise.
        """
        field_name = rule.get("field")
        terms = rule.get("terms", [])

        if not field_name or not terms:
            return None

        if not hasattr(event, field_name):
            return None

        return getattr(event, field_name)

    def _should_keep_event(self, event: Any, table_name: str) -> bool:
        """Check if an event should be kept based on table configuration filters.

        Called by `_process_table_events` to filter events before text extraction.
        Applies include/exclude rules from the table configuration to determine
        whether an event should be processed.

        Args:
            event (Any): Event object (e.g., lab event, prescription event) with
                attributes corresponding to fields specified in filter rules.
            table_name (str): Name of the table (e.g., "labevents",
                "prescriptions") used to look up filter configuration.

        Returns:
            bool: True if event should be kept and processed, False if event
                should be excluded. Returns True if table has no filters or if
                table is not in configuration.

        Note:
            Filter logic:
            - If includes are specified: event is kept if ANY include rule
              matches (OR logic). If includes are specified but none match,
              event is excluded.
            - If excludes are specified: event is excluded if ANY exclude rule
              matches (OR logic).
            - If both includes and excludes are specified: includes are checked
              first, then excludes.
            - Include rules use exact matching, exclude rules use case-insensitive
              substring matching.
        """
        if table_name not in self.table_config:
            # If table not in config, keep all events
            return True

        filters = self.table_config[table_name].get("filters", {})

        # If no filters, keep all
        if not filters:
            return True

        includes = filters.get("includes", [])
        excludes = filters.get("excludes", [])

        # Handle includes: keep if ANY include rule matches (OR logic)
        if includes:
            include_matched = False
            for include_rule in includes:
                field_value = self._get_field_value(event, include_rule)
                if field_value is None:
                    continue

                terms = include_rule.get("terms", [])
                # Exact match for includes
                if field_value in terms:
                    include_matched = True
                    break

            # If includes are specified but none matched, exclude the event
            if not include_matched:
                return False

        # Handle excludes: exclude if ANY exclude rule matches (OR logic)
        if excludes:
            for exclude_rule in excludes:
                field_value = self._get_field_value(event, exclude_rule)
                if not field_value:
                    continue

                terms = exclude_rule.get("terms", [])
                field_value_str = str(field_value).lower()

                # Substring match for excludes (case-insensitive)
                for term in terms:
                    if term.lower() in field_value_str:
                        return False

        # Event passed all filters
        return True

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter dataset to limit number of patients processed.

        Called by PyHealth dataset processing pipeline before task execution.
        If max_patients is set, filters the dataframe to include only the first
        N unique patients. This is useful for development and testing with
        smaller subsets of data.

        Args:
            df (pl.LazyFrame): Polars LazyFrame containing patient data with
                a "patient_id" column. This is the dataset to filter.

        Returns:
            pl.LazyFrame: Filtered LazyFrame containing only the first
                max_patients unique patients. If max_patients is None, returns
                the original dataframe unchanged.
        """
        if self.max_patients is None:
            return df

        unique_patient_ids = (
            df.select("patient_id")
            .unique()
            .head(self.max_patients)["patient_id"]
            .to_list()
        )
        filtered_df = df.filter(pl.col("patient_id").is_in(unique_patient_ids))

        return filtered_df

    def _extract_fields(
        self, obj: Any, table_name: str, field_config: List[str]
    ) -> List[str]:
        """Extract specified fields from an event object and format as text parts.

        Called by `_process_table_events` to extract field values from event
        objects (e.g., lab events, prescriptions) according to the field
        configuration. Fields are extracted only if their value is not None.

        Args:
            obj (Any): Event object (e.g., lab event, prescription event) with
                attributes to extract. Must have attributes matching field names
                in field_config.
            table_name (str): Name of the table (e.g., "labevents",
                "prescriptions") used as a prefix in the output text parts.
            field_config (List[str]): List of field names (str) to extract from
                obj. Each field is included only if its value is not None.

        Returns:
            List[str]: List of strings in format [table_name, field_name, value,
                table_name, field_name, value, ...]. Each field that is not None
                contributes three strings: table name, field name, and string
                representation of the value. Empty list if no fields have
                non-None values.
        """
        text_parts = []

        for field_name in field_config:
            if hasattr(obj, field_name):
                value = getattr(obj, field_name)

                # Include field only if value is not None
                if value is not None:
                    text_parts.extend([table_name, field_name, str(value)])

        return text_parts

    def _process_table_events(
        self, events: List[Any], table_name: str, visit_id: Any, patient_id: Any
    ) -> List[Dict[str, Any]]:
        """Process events for a specific table and extract text samples.

        Called by `__call__` to process events from a single table (e.g.,
        labevents, prescriptions) for a specific visit. Applies filtering,
        extracts fields, and creates text samples with metadata.

        Args:
            events (List[Any]): List of event objects (e.g., lab events,
                prescription events) to process for this table and visit.
            table_name (str): Name of the table (e.g., "labevents",
                "prescriptions") used to look up configuration and as event_type.
            visit_id (Any): Visit/admission ID (e.g., hadm_id) to associate
                with extracted samples. Can be None if not available.
            patient_id (Any): Patient ID to associate with extracted samples.

        Returns:
            List[Dict[str, Any]]: List of sample dictionaries, each containing:
                - "patient_id": Patient ID
                - "visit_id": Visit/admission ID
                - "event_type": Event type (table name, singularized if plural)
                - "text": Extracted text string (space-separated field values)
            Empty list if table is not in configuration or no events pass
            filtering and field extraction.
        """
        samples = []

        if table_name not in self.table_config:
            return samples

        table_config = self.table_config[table_name]
        extract_fields = table_config.get("extract_fields", [])

        # Apply filtering
        filtered_events = [
            event for event in events if self._should_keep_event(event, table_name)
        ]

        # Extract text from events
        for event in filtered_events:
            text_parts = self._extract_fields(event, table_name, extract_fields)

            if text_parts:
                # Space-separated: "table_name field value ..."
                text = " ".join(text_parts)
                # Use table name as event_type
                # Remove 's' from plural names for consistency
                event_type = (
                    table_name.rstrip("s") if table_name.endswith("s") else table_name
                )
                samples.append(
                    {
                        "patient_id": patient_id,
                        "visit_id": visit_id,
                        "event_type": event_type,
                        "text": text,
                    }
                )

        return samples

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to extract text from configured EHR tables.

        Processes all visits for a patient, extracting text from configured
        tables (e.g., labevents, prescriptions) within each visit's time window.

        Args:
            patient (Any): Patient object from PyHealth dataset with:
                - patient_id: Patient identifier
                - get_events(event_type, start, end): Method to retrieve events
                  filtered by type and time window

        Returns:
            List[Dict[str, Any]]: List of sample dictionaries, each containing:
                - "patient_id": Patient identifier
                - "visit_id": Visit/admission ID (e.g., hadm_id)
                - "event_type": Type of event (e.g., "labevent", "prescription")
                - "text": Extracted text string with space-separated field values
            Empty list if patient has no admissions or no events pass filtering.
        """
        samples = []

        # Get all admissions/visits for this patient
        admissions = patient.get_events(event_type="admissions")

        # Process by admission/visit
        for admission in admissions:
            visit_id = getattr(admission, "hadm_id", None)
            admission_time = admission.timestamp
            discharge_time_str = getattr(admission, "dischtime", None)

            # Convert discharge_time from string to datetime if needed
            discharge_time = None
            if discharge_time_str:
                try:
                    discharge_time = datetime.strptime(
                        discharge_time_str, "%Y-%m-%d %H:%M:%S"
                    )
                except (ValueError, AttributeError, TypeError):
                    # If conversion fails, use None
                    discharge_time = None

            # Process each table in the configuration
            for table_name in self.table_config.keys():
                # Skip admissions table (used for visit context only)
                if table_name == "admissions":
                    continue

                # Get events for this table using timestamp filtering
                events = patient.get_events(
                    event_type=table_name,
                    start=admission_time,
                    end=discharge_time,
                )

                # Process events for this table
                table_samples = self._process_table_events(
                    events, table_name, visit_id, patient.patient_id
                )
                samples.extend(table_samples)

        return samples
