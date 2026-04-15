from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class MortalityPredictionMIMIC3(BaseTask):
    """Task for predicting mortality using MIMIC-III dataset with text data.

    This task aims to predict whether the patient will decease in the next
    hospital visit based on clinical information from the current visit.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MortalityPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = MortalityPredictionMIMIC3()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MortalityPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task."""
        samples = []

        # We will drop the last visit
        visits = patient.get_events(event_type="admissions")

        if len(visits) <= 1:
            return []

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Check discharge status for mortality label - more robust handling
            if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(next_visit.hospital_expire_flag)

            # Get clinical codes using hadm_id-based filtering
            # (more precise than timestamp filtering)
            diagnoses = patient.get_events(
                event_type="diagnoses_icd", filters=[("hadm_id", "==", visit.hadm_id)]
            )
            procedures = patient.get_events(
                event_type="procedures_icd", filters=[("hadm_id", "==", visit.hadm_id)]
            )
            prescriptions = patient.get_events(
                event_type="prescriptions", filters=[("hadm_id", "==", visit.hadm_id)]
            )

            conditions = [event.icd9_code for event in diagnoses]
            procedures_list = [event.icd9_code for event in procedures]
            drugs = [event.ndc for event in prescriptions if event.ndc]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "hadm_id": visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                }
            )

        return samples


class MultimodalMortalityPredictionMIMIC3(BaseTask):
    """Task for predicting mortality using MIMIC-III dataset with text data.

    This task aims to predict whether the patient will decease in the next
    hospital visit based on clinical information from the current visit.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MultimodalMortalityPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions",
        ...             "noteevents"],
        ... )
        >>> task = MultimodalMortalityPredictionMIMIC3()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MultimodalMortalityPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
        "clinical_notes": "text",  # Added support for clinical notes
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task."""
        samples = []

        # We will drop the last visit
        visits = patient.get_events(event_type="admissions")

        if len(visits) <= 1:
            return []

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Check discharge status for mortality label - more robust handling
            if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(next_visit.hospital_expire_flag)

            # Get clinical codes using hadm_id-based filtering
            # (more precise than timestamp filtering)
            diagnoses = patient.get_events(
                event_type="diagnoses_icd", filters=[("hadm_id", "==", visit.hadm_id)]
            )
            procedures = patient.get_events(
                event_type="procedures_icd", filters=[("hadm_id", "==", visit.hadm_id)]
            )
            prescriptions = patient.get_events(
                event_type="prescriptions", filters=[("hadm_id", "==", visit.hadm_id)]
            )
            # Get clinical notes
            notes = patient.get_events(
                event_type="noteevents", filters=[("hadm_id", "==", visit.hadm_id)]
            )
            conditions = [event.icd9_code for event in diagnoses]
            procedures_list = [event.icd9_code for event in procedures]
            drugs = [event.ndc for event in prescriptions if event.ndc]
            # Extract note text - concatenate if multiple exist
            text = ""
            for note in notes:
                text += note.text

            # Heterogeneous problem. Some events may not have notes, procedures, prescriptions, or diagnoses.
            samples.append(
                {
                    "hadm_id": visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "clinical_notes": text,
                    "mortality": mortality_label,
                }
            )

        return samples


class MortalityPredictionMIMIC4(BaseTask):
    """Task for predicting mortality using MIMIC-IV EHR data only.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import MortalityPredictionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = MortalityPredictionMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_sequence(self, sequence: Optional[List[Any]]) -> List[str]:
        """
        Clean a sequence by:
        1. Removing None values
        2. Converting to strings
        3. Removing empty strings
        """
        if sequence is None:
            return []

        # Remove None, convert to strings, remove empty strings
        cleaned = [
            str(item).strip()
            for item in sequence
            if item is not None and str(item).strip()
        ]
        return cleaned

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task."""
        samples = []

        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]
        anchor_age = getattr(demographics, "anchor_age", None)

        # Safely check age - fix potential bug with non-numeric ages
        try:
            if anchor_age is not None and int(float(anchor_age)) < 18:
                return []  # Skip patients under 18
        except (ValueError, TypeError):
            # If age can't be determined, we'll include the patient
            pass

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            admission = admissions[i]
            next_admission = admissions[i + 1]

            # Check discharge status for mortality label - more robust handling
            if next_admission.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(next_admission.hospital_expire_flag)

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If date parsing fails, skip this admission
                print("Error parsing admission discharge time:", admission.dischtime)
                continue

            # Get clinical codes
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime,
            )

            # Extract relevant data
            conditions = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in diagnoses_icd]
            )
            procedures_list = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in procedures_icd]
            )
            drugs = self._clean_sequence(
                [getattr(event, "ndc", None) for event in prescriptions]
            )

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                }
            )
        return samples


class MultimodalMortalityPredictionMIMIC4(BaseTask):
    """Task for predicting patient-level mortality using MIMIC-IV multimodal data.

    This task combines multiple modalities for mortality prediction at the
    PATIENT LEVEL (not visit level). All core modalities are required for
    each sample.

    Required Modalities:
    - EHR codes: ICD diagnoses, procedures, AND prescriptions (all required)
    - Clinical notes: Discharge summaries OR radiology reports (at least one)
    - Lab events: 10-dimensional lab value vectors (time-series)
    - Chest X-rays: Must have an image path available

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     note_root="/path/to/mimic-iv-note/2.2",
        ...     cxr_root="/path/to/mimic-cxr/2.0.0",
        ...     ehr_tables=["diagnoses_icd", "procedures_icd",
        ...                 "prescriptions", "labevents"],
        ...     note_tables=["discharge", "radiology"],
        ...     cxr_tables=["metadata", "negbio"],
        ... )
        >>> task = MultimodalMortalityPredictionMIMIC4()
        >>> samples = dataset.set_task(task)

    Patient-Level Aggregation:
        - Mortality is determined iteratively by checking if the NEXT admission
          has the death flag
        - Admissions are included up to (but not including) any admission where
          the patient dies
        - For surviving patients: aggregate all events across all admissions
        - Returns ONE sample per patient with aggregated multimodal data

    Modality Coverage:
        - No modality requirements - returns all patients
        - Coverage analysis should be done downstream
        - Discharge and radiology notes are returned as lists (raw processor)
        - lab_values uses nested_sequence_floats processor for 10D vectors
        - lab_times is a separate list of time offsets (raw processor)

    Lab Processing:
        - 10-dimensional vectors (one per lab category)
        - Categories: Sodium, Potassium, Chloride, Bicarbonate, Glucose,
          Calcium, Magnesium, Anion Gap, Osmolality, Phosphate
        - Multiple itemids per category → take first observed value
        - Time intervals calculated from first admission start (hours)

    Image Processing:
        - Uses image_path from MIMIC-CXR metadata directly
        - Returns first available X-ray image path across all X-rays
    """

    task_name: str = "MultimodalMortalityPredictionMIMIC4"

    # Lab categories matching MortalityPredictionStageNetMIMIC4
    LAB_CATEGORIES: Dict[str, List[str]] = {
        "Sodium": ["50824", "52455", "50983", "52623"],
        "Potassium": ["50822", "52452", "50971", "52610"],
        "Chloride": ["50806", "52434", "50902", "52535"],
        "Bicarbonate": ["50803", "50804"],
        "Glucose": ["50809", "52027", "50931", "52569"],
        "Calcium": ["50808", "51624"],
        "Magnesium": ["50960"],
        "Anion Gap": ["50868", "52500"],
        "Osmolality": ["52031", "50964", "51701"],
        "Phosphate": ["50970"],
    }

    LAB_CATEGORY_NAMES: List[str] = [
        "Sodium",
        "Potassium",
        "Chloride",
        "Bicarbonate",
        "Glucose",
        "Calcium",
        "Magnesium",
        "Anion Gap",
        "Osmolality",
        "Phosphate",
    ]

    LABITEMS: List[str] = [
        item for itemids in LAB_CATEGORIES.values() for item in itemids
    ]

    def __init__(self, **kwargs):
        """Initialize the multimodal mortality prediction task.

        Args:
            **kwargs: Passed to :class:`~pyhealth.tasks.BaseTask`, e.g.
                ``code_mapping``.
        """
        self.input_schema: Dict[str, str] = {
            "conditions": "nested_sequence",  # Nested by visit
            "procedures": "nested_sequence",  # Nested by visit
            "drugs": "nested_sequence",  # Nested by visit
            "discharge": "raw",  # List of discharge notes
            "radiology": "raw",  # List of radiology notes
            "lab_values": "nested_sequence_floats",  # 10D lab vectors per timestamp
            "lab_times": "raw",  # Lab measurement times (hours from first admission)
            "negbio_findings": "sequence",  # NegBio X-ray findings
            "image_path": "text",  # Image path as text string
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}
        super().__init__(**kwargs)

    def _clean_sequence(self, sequence: Optional[List[Any]]) -> List[str]:
        """Clean a sequence by removing None values and converting to strings."""
        if sequence is None:
            return []

        # Remove None, convert to strings, remove empty strings
        cleaned = [
            str(item).strip()
            for item in sequence
            if item is not None and str(item).strip()
        ]
        return cleaned

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def _process_lab_events(
        self,
        patient: Any,
        admission_time: datetime,
        admission_dischtime: datetime,
        reference_time: Optional[datetime] = None,
    ) -> Optional[tuple]:
        """Process lab events into 10-dimensional vectors with timestamps.

        Args:
            patient: Patient object
            admission_time: Admission start time
            admission_dischtime: Admission discharge time
            reference_time: Reference time for calculating time offsets (default: admission_time)

        Returns:
            Tuple of (times_list, values_list) or None if no lab events
        """
        try:
            import polars as pl
        except ImportError:
            return None

        if reference_time is None:
            reference_time = admission_time

        # Use timestamp filtering for lab events (hadm_id not reliable)
        labevents_df = patient.get_events(
            event_type="labevents",
            start=admission_time,
            end=admission_dischtime,
            return_df=True,
        )

        if labevents_df is None or labevents_df.height == 0:
            return None

        # Filter to relevant lab items
        labevents_df = labevents_df.filter(
            pl.col("labevents/itemid").is_in(self.LABITEMS)
        )

        if labevents_df.height == 0:
            return None

        # Parse storetime and filter (matching stagenet implementation)
        labevents_df = labevents_df.with_columns(
            pl.col("labevents/storetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
        labevents_df = labevents_df.filter(
            pl.col("labevents/storetime") <= admission_dischtime
        )

        if labevents_df.height == 0:
            return None

        # Select relevant columns
        labevents_df = labevents_df.select(
            pl.col("timestamp"),
            pl.col("labevents/itemid"),
            pl.col("labevents/valuenum").cast(pl.Float64),
        )

        # Group by timestamp and aggregate into 10D vectors
        unique_timestamps = sorted(labevents_df["timestamp"].unique().to_list())

        lab_times = []
        lab_values = []

        for lab_ts in unique_timestamps:
            ts_labs = labevents_df.filter(pl.col("timestamp") == lab_ts)

            # Create 10-dimensional vector
            lab_vector = []
            for category_name in self.LAB_CATEGORY_NAMES:
                category_itemids = self.LAB_CATEGORIES[category_name]

                # Find first matching value for this category
                category_value = None
                for itemid in category_itemids:
                    matching = ts_labs.filter(pl.col("labevents/itemid") == itemid)
                    if matching.height > 0:
                        category_value = matching["labevents/valuenum"][0]
                        break

                lab_vector.append(category_value)

            # Calculate time from reference time (hours)
            time_from_reference = (lab_ts - reference_time).total_seconds() / 3600.0

            lab_times.append(time_from_reference)
            lab_values.append(lab_vector)

        if len(lab_values) == 0:
            return None

        return (lab_times, lab_values)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for patient-level multimodal mortality prediction.

        This task aggregates ALL modalities across visits at the patient level,
        supporting heterogeneous features (not all modalities required).

        Mortality is determined iteratively by checking if the NEXT admission
        has the death flag. Admissions are included up to (but not including)
        any admission where the patient dies.

        Returns ONE sample per patient with aggregated multimodal data.
        """
        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        # Determine which admissions to process iteratively
        # Check each admission's NEXT admission for mortality flag
        admissions_to_process = []
        mortality_label = 0

        for i, admission in enumerate(admissions):
            # Check if THIS admission has the death flag
            if admission.hospital_expire_flag in [1, "1"]:
                # Patient died in this admission - set mortality label
                # but don't include this admission's data
                mortality_label = 1
                break

            # Check if there's a next admission with death flag
            if i + 1 < len(admissions):
                next_admission = admissions[i + 1]
                if next_admission.hospital_expire_flag in [1, "1"]:
                    # Next admission has death - include current, set mortality
                    admissions_to_process.append(admission)
                    mortality_label = 1
                    break

            # No death in current or next - include this admission
            admissions_to_process.append(admission)

        if len(admissions_to_process) == 0:
            return []

        # Get first admission time as reference for lab time calculations
        first_admission_time = admissions_to_process[0].timestamp

        # Aggregated data across all admissions
        all_conditions = []
        all_procedures = []
        all_drugs = []
        all_discharge_notes = []  # List of individual discharge notes
        all_radiology_notes = []  # List of individual radiology notes
        all_lab_times = []
        all_lab_values = []
        all_negbio_findings = []
        image_path = ""  # Empty string instead of None for serialization

        # Get X-ray data (patient-level, not admission-specific)
        # Note: event types match table names in mimic4_cxr.yaml (negbio, metadata)
        negbio_events = patient.get_events(event_type="negbio")
        metadata_events = patient.get_events(event_type="metadata")

        # Process X-ray findings (aggregate across all X-rays)
        # NegBio findings attributes (from mimic4_cxr.yaml negbio table)
        negbio_finding_names = [
            "no finding",
            "enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "lung lesion",
            "edema",
            "consolidation",
            "pneumonia",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "pleural other",
            "fracture",
            "support devices",
        ]
        for xray in negbio_events:
            try:
                for finding_name in negbio_finding_names:
                    try:
                        value = getattr(xray, finding_name, None)
                        if value is not None and float(value) > 0:
                            all_negbio_findings.append(finding_name)
                    except (ValueError, TypeError, AttributeError):
                        pass
            except Exception:
                pass

        # Get first available image path from metadata
        for event in metadata_events:
            try:
                if event.image_path:
                    image_path = event.image_path
                    break  # Use first valid image
            except AttributeError:
                pass

        # Process each admission and aggregate data
        for admission in admissions_to_process:
            # Parse admission discharge time for lab events filtering
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If we can't parse discharge time, skip this admission
                continue

            # Skip if discharge is before admission (data quality issue)
            if admission_dischtime < admission.timestamp:
                continue

            # Get clinical codes using hadm_id filtering
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            # Get notes using hadm_id filtering
            discharge_notes = patient.get_events(
                event_type="discharge", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology", filters=[("hadm_id", "==", admission.hadm_id)]
            )

            # Extract clinical codes per visit (nested structure)
            conditions = self._clean_sequence(
                [event.icd_code for event in diagnoses_icd]
            )
            procedures_list = self._clean_sequence(
                [event.icd_code for event in procedures_icd]
            )
            drugs = self._clean_sequence([event.ndc for event in prescriptions])

            # Append as nested lists (one list per visit) for nested_sequence
            all_conditions.append(conditions)
            all_procedures.append(procedures_list)
            all_drugs.append(drugs)

            # Extract and aggregate notes as individual items in lists
            # Note: attribute is "text" (from mimic4_note.yaml), not "discharge"/"radiology"
            for note in discharge_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_discharge_notes.append(note_text)
                except AttributeError:
                    pass

            for note in radiology_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_radiology_notes.append(note_text)
                except AttributeError:
                    pass

            # Process lab events with reference to first admission time
            labs_data = self._process_lab_events(
                patient,
                admission.timestamp,
                admission_dischtime,
                reference_time=first_admission_time,
            )

            if labs_data is not None:
                lab_times, lab_values = labs_data
                all_lab_times.extend(lab_times)
                all_lab_values.extend(lab_values)

        # ===== MODALITY REQUIREMENTS =====
        # Check that all required modalities are present before returning sample
        # Required: EHR codes (conditions, procedures, drugs), notes (discharge OR radiology),
        # labs, and image_path

        # Check EHR codes - need at least one code in each category across all visits
        has_conditions = any(len(codes) > 0 for codes in all_conditions)
        has_procedures = any(len(codes) > 0 for codes in all_procedures)
        has_drugs = any(len(codes) > 0 for codes in all_drugs)

        # Check notes - need at least one discharge OR radiology note
        has_notes = len(all_discharge_notes) > 0 or len(all_radiology_notes) > 0

        # Check labs - need at least one lab measurement
        has_labs = len(all_lab_times) > 0

        # Check image - need a valid image path
        has_image = bool(image_path)

        # Return empty list if any required modality is missing
        if not (
            has_conditions
            and has_procedures
            and has_drugs
            and has_notes
            and has_labs
            and has_image
        ):
            return []

        # Sort lab events by time and create aggregated labs data
        # Use nested_sequence_floats processor for lab_values (handles None values)
        if all_lab_times:
            sorted_indices = sorted(
                range(len(all_lab_times)), key=lambda k: all_lab_times[k]
            )
            sorted_lab_times = [all_lab_times[i] for i in sorted_indices]
            sorted_lab_values = [all_lab_values[i] for i in sorted_indices]
        else:
            sorted_lab_times = []
            sorted_lab_values = []

        # Deduplicate negbio findings (flat sequence)
        unique_negbio = list(dict.fromkeys(all_negbio_findings))

        # Return single patient-level sample with heterogeneous features
        # Note: conditions/procedures/drugs are nested lists (one list per visit)
        # Note: discharge and radiology are lists (passed through by raw processor)
        # Note: lab_values uses nested_sequence_floats processor (handles None values)
        return [
            {
                "patient_id": patient.patient_id,
                "conditions": all_conditions,  # Nested: [[visit1_codes], [visit2_codes], ...]
                "procedures": all_procedures,  # Nested: [[visit1_codes], [visit2_codes], ...]
                "drugs": all_drugs,  # Nested: [[visit1_codes], [visit2_codes], ...]
                "discharge": all_discharge_notes,  # List of discharge notes
                "radiology": all_radiology_notes,  # List of radiology notes
                "lab_values": sorted_lab_values,  # Nested floats: [[10D vector], ...]
                "lab_times": sorted_lab_times,  # List of times (hours from first admission)
                "negbio_findings": unique_negbio,  # NegBio X-ray findings
                "image_path": image_path,  # Image path as string
                "mortality": mortality_label,
            }
        ]


class MortalityPredictionEICU(BaseTask):
    """Task for predicting mortality using eICU dataset.

    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.

    Features key-value pairs:
    - using diagnosis table (ICD9CM and ICD10CM) as condition codes
    - using physicalexam table as procedure codes
    - using medication table as drugs codes

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import MortalityPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalExam"],
        ... )
        >>> task = MortalityPredictionEICU()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MortalityPredictionEICU"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task.

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a dict with
                patient_id, visit_id, conditions, procedures, drugs and mortality.
        """
        samples = []

        # Get patient stays (each row in patient table is an ICU stay)
        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) <= 1:
            return []

        for i in range(len(patient_stays) - 1):
            stay = patient_stays[i]
            next_stay = patient_stays[i + 1]

            # Check discharge status for mortality label
            # In eICU, hospitaldischargestatus indicates "Alive" or "Expired"
            discharge_status = getattr(next_stay, "hospitaldischargestatus", None)
            if discharge_status not in ["Alive", "Expired"]:
                mortality_label = 0
            else:
                mortality_label = 0 if discharge_status == "Alive" else 1

            # Get the patientunitstayid for filtering
            stay_id = str(getattr(stay, "patientunitstayid", ""))

            # Get clinical codes using patientunitstayid-based filtering
            diagnoses = patient.get_events(
                event_type="diagnosis",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            physical_exams = patient.get_events(
                event_type="physicalexam",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            medications = patient.get_events(
                event_type="medication",
                filters=[("patientunitstayid", "==", stay_id)]
            )

            # Extract codes - use icd9code for diagnoses, physicalexampath for exams, drugname for meds
            conditions = [
                getattr(event, "icd9code", "") for event in diagnoses
                if getattr(event, "icd9code", None)
            ]
            procedures_list = [
                getattr(event, "physicalexampath", "") for event in physical_exams
                if getattr(event, "physicalexampath", None)
            ]
            drugs = [
                getattr(event, "drugname", "") for event in medications
                if getattr(event, "drugname", None)
            ]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            # TODO: Exclude visits with age < 18

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                }
            )

        return samples


class MortalityPredictionEICU2(BaseTask):
    """Task for predicting mortality using eICU dataset with alternative coding.

    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.

    Similar to MortalityPredictionEICU, but with different code mapping:
    - using admissiondx table and diagnosisstring under diagnosis table as condition codes
    - using treatment table as procedure codes

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import MortalityPredictionEICU2
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "treatment", "admissionDx"],
        ... )
        >>> task = MortalityPredictionEICU2()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MortalityPredictionEICU2"
    input_schema: Dict[str, str] = {"conditions": "sequence", "procedures": "sequence"}
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task.

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a dict with
                patient_id, visit_id, conditions, procedures and mortality.
        """
        samples = []

        # Get patient stays (each row in patient table is an ICU stay)
        patient_stays = patient.get_events(event_type="patient")
        if len(patient_stays) <= 1:
            return []

        for i in range(len(patient_stays) - 1):
            stay = patient_stays[i]
            next_stay = patient_stays[i + 1]

            # Check discharge status for mortality label
            discharge_status = getattr(next_stay, "hospitaldischargestatus", None)
            if discharge_status not in ["Alive", "Expired"]:
                mortality_label = 0
            else:
                mortality_label = 0 if discharge_status == "Alive" else 1

            # Get the patientunitstayid for filtering
            stay_id = str(getattr(stay, "patientunitstayid", ""))

            # Get clinical codes using patientunitstayid-based filtering
            admission_dx = patient.get_events(
                event_type="admissiondx",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            diagnosis_events = patient.get_events(
                event_type="diagnosis",
                filters=[("patientunitstayid", "==", stay_id)]
            )
            treatments = patient.get_events(
                event_type="treatment",
                filters=[("patientunitstayid", "==", stay_id)]
            )

            # Get diagnosis strings from diagnosis events
            diagnosis_strings = list(
                set(
                    [
                        getattr(event, "diagnosisstring", "")
                        for event in diagnosis_events
                        if getattr(event, "diagnosisstring", None)
                    ]
                )
            )

            # Get admission diagnosis codes
            admission_dx_codes = [
                getattr(event, "admitdxpath", "") for event in admission_dx
                if getattr(event, "admitdxpath", None)
            ]
            
            # Get treatment codes
            treatment_codes = [
                getattr(event, "treatmentstring", "") for event in treatments
                if getattr(event, "treatmentstring", None)
            ]

            # Combine admission diagnoses and diagnosis strings
            conditions = admission_dx_codes + diagnosis_strings

            # Exclude visits without sufficient codes
            if len(conditions) * len(treatment_codes) == 0:
                continue

            # TODO: Exclude visits with age < 18

            samples.append(
                {
                    "visit_id": stay_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": treatment_codes,
                    "mortality": mortality_label,
                }
            )

        return samples


class MortalityPredictionOMOP(BaseTask):
    """Task for predicting mortality using OMOP CDM dataset.

    This task predicts whether a patient has a death record (binary
    mortality prediction) based on clinical information from each visit.
    Unlike visit-specific mortality tasks, this provides a patient-level
    mortality indicator (whether the patient died at any point).

    The task processes visits sequentially and extracts clinical codes
    (conditions, procedures, drugs) for each visit. Clinical events are
    linked to visits via visit_occurrence_id, following OMOP CDM
    conventions.

    Features:
        - Uses OMOP CDM standard tables (condition_occurrence,
          procedure_occurrence, drug_exposure)
        - Links clinical events to visits via visit_occurrence_id
        - Uses OMOP concept_ids as medical codes
        - Binary mortality label (1 if patient has death record, 0
          otherwise)

    Task Schema:
        Input:
            - conditions: sequence of condition_concept_id codes
            - procedures: sequence of procedure_concept_id codes
            - drugs: sequence of drug_concept_id codes
        Output:
            - mortality: binary label (0: no death record, 1: death record)

    Args:
        patient (Patient): A Patient object containing OMOP CDM data.

    Returns:
        List[Dict[str, Any]]: A list of samples, where each sample
            contains:
            - visit_id: The visit_occurrence_id
            - patient_id: The person_id
            - conditions: List of condition_concept_id codes
            - procedures: List of procedure_concept_id codes
            - drugs: List of drug_concept_id codes
            - mortality: Binary label (0 or 1)

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> from pyhealth.tasks import MortalityPredictionOMOP
        >>>
        >>> # Load OMOP dataset
        >>> dataset = OMOPDataset(
        ...     root="/path/to/omop/data",
        ...     tables=["condition_occurrence", "procedure_occurrence",
        ...             "drug_exposure"],
        ... )
        >>>
        >>> # Create mortality prediction task
        >>> task = MortalityPredictionOMOP()
        >>> sample_dataset = dataset.set_task(task=task)
        >>>
        >>> # Access samples
        >>> print(f"Generated {len(sample_dataset)} samples")
        >>> sample = sample_dataset.samples[0]
        >>> print(f"Conditions: {sample['conditions']}")
        >>> print(f"Mortality: {sample['mortality']}")

    Note:
        - Visits without any clinical codes (conditions, procedures, or
          drugs) are excluded
        - The last visit is excluded as there is no "next visit" to
          predict for
        - Clinical events are filtered by visit_occurrence_id, not by
          timestamp ranges, following OMOP best practices
    """

    task_name: str = "MortalityPredictionOMOP"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task.

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a
                dict with patient_id, visit_id, conditions, procedures,
                drugs and mortality.
        """
        samples = []

        # Get visits and death events
        visits = patient.get_events(event_type="visit_occurrence")
        death_events = patient.get_events(event_type="death")

        if len(visits) <= 1:
            return []

        # Get death datetime if exists
        death_datetime = None
        if len(death_events) > 0:
            death_datetime = death_events[0].timestamp

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Determine mortality label
            # Check if patient has death record (died at some point)
            # This is binary mortality prediction, not time-specific
            mortality_label = 1 if death_datetime is not None else 0

            # Get visit end datetime for filtering events
            visit_end_str = getattr(visit, "visit_end_datetime", None)

            # Parse visit_end_datetime if it's a string
            visit_end = None
            if visit_end_str is not None:
                if isinstance(visit_end_str, str):
                    try:
                        visit_end = datetime.strptime(
                            visit_end_str, "%Y-%m-%d %H:%M:%S"
                        )
                    except (ValueError, TypeError):
                        visit_end = None
                else:
                    visit_end = visit_end_str

            # Fallback to next visit start if visit_end not available
            if visit_end is None:
                visit_end = next_visit.timestamp

            # Get visit_occurrence_id for filtering
            visit_occurrence_id = str(getattr(visit, "visit_occurrence_id", None))

            # Get clinical codes within this visit using visit_occurrence_id
            # In OMOP, clinical events are linked to visits by
            # visit_occurrence_id
            if visit_occurrence_id:
                conditions = patient.get_events(
                    event_type="condition_occurrence",
                    filters=[("visit_occurrence_id", "==", visit_occurrence_id)],
                )
                procedures = patient.get_events(
                    event_type="procedure_occurrence",
                    filters=[("visit_occurrence_id", "==", visit_occurrence_id)],
                )
                drugs = patient.get_events(
                    event_type="drug_exposure",
                    filters=[("visit_occurrence_id", "==", visit_occurrence_id)],
                )

            # Extract concept IDs as codes
            condition_codes = [
                str(getattr(event, "condition_concept_id", ""))
                for event in conditions
                if getattr(event, "condition_concept_id", None) is not None
            ]
            procedure_codes = [
                str(getattr(event, "procedure_concept_id", ""))
                for event in procedures
                if getattr(event, "procedure_concept_id", None) is not None
            ]
            drug_codes = [
                str(getattr(event, "drug_concept_id", ""))
                for event in drugs
                if getattr(event, "drug_concept_id", None) is not None
            ]

            # Exclude visits without any clinical codes
            total_codes = len(condition_codes) + len(procedure_codes) + len(drug_codes)
            if total_codes == 0:
                continue

            samples.append(
                {
                    "visit_id": visit_occurrence_id,
                    "patient_id": patient.patient_id,
                    "conditions": condition_codes,
                    "procedures": procedure_codes,
                    "drugs": drug_codes,
                    "mortality": mortality_label,
                }
            )

        return samples
