from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class MortalityPredictionMIMIC3(BaseTask):
    """Task for predicting mortality using MIMIC-III dataset with text data.

    This task aims to predict whether the patient will decease in the next
    hospital visit based on clinical information from the current visit.
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
            drugs = [event.drug for event in prescriptions]

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
            drugs = [event.drug for event in prescriptions]
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
    """Task for predicting mortality using MIMIC-IV EHR data only."""

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
                [getattr(event, "drug", None) for event in prescriptions]
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
    """Task for predicting mortality using MIMIC-IV multimodal data.

    This task combines multiple modalities for mortality prediction:
    - EHR codes: ICD diagnoses, procedures, and prescriptions
    - Clinical notes: Discharge summaries and radiology reports
    - Lab events: 10-dimensional lab value vectors (time-series)
    - Chest X-rays: CXR images and NegBio findings

    This is a TRUE multimodal task requiring ALL modalities to be present,
    making it suitable for showcasing PyHealth's multimodal data loading
    capabilities.

    Lab Processing:
        - 10-dimensional vectors (one per lab category)
        - Categories: Sodium, Potassium, Chloride, Bicarbonate, Glucose,
          Calcium, Magnesium, Anion Gap, Osmolality, Phosphate
        - Multiple itemids per category → take first observed value
        - Time intervals calculated from admission start (hours)

    Image Processing:
        - Uses "image" processor for chest X-ray loading
        - Images are loaded from MIMIC-CXR dataset paths
        - Returns first available X-ray image per admission
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

    def __init__(self, cxr_root: Optional[str] = None):
        """Initialize the multimodal mortality prediction task.

        Args:
            cxr_root: Root directory for MIMIC-CXR images. If provided,
                image paths will be prefixed with this root.
        """
        self.cxr_root = cxr_root
        self.input_schema: Dict[str, str] = {
            "conditions": "sequence",
            "procedures": "sequence",
            "drugs": "sequence",
            "discharge": "text",
            "radiology": "text",
            "labs": "stagenet_tensor",  # 10D lab vectors with time
            "xrays_negbio": "sequence",
            "image_path": "text",  # returns the image_path
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

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
        """Clean text by stripping whitespace and returning None if empty."""
        if text is None:
            return None

        cleaned_text = str(text).strip()
        return cleaned_text if cleaned_text else None

    def _construct_image_path(
        self, subject_id: str, study_id: str, dicom_id: str
    ) -> str:
        """Constructs the path to a MIMIC-CXR image file.

        Args:
            subject_id: The patient/subject ID (e.g., "10000032")
            study_id: The study ID (e.g., "50414267")
            dicom_id: The DICOM ID (e.g., "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014")

        Returns:
            The path to the image file
        """
        # Extract first two characters of patient_id for parent folder
        patient_id_clean = subject_id.replace("p", "")
        parent_folder = f"p{patient_id_clean[:2]}"
        patient_folder = f"p{patient_id_clean}"

        # Construct path
        relative_path = (
            f"files/{parent_folder}/{patient_folder}/s{study_id}/{dicom_id}.jpg"
        )

        if self.cxr_root:
            return f"{self.cxr_root}/{relative_path}"
        return relative_path

    def _process_lab_events(
        self, patient: Any, admission_time: datetime, admission_dischtime: datetime
    ) -> Optional[tuple]:
        """Process lab events into 10-dimensional vectors with timestamps.

        Args:
            patient: Patient object
            admission_time: Admission start time
            admission_dischtime: Admission discharge time

        Returns:
            Tuple of (times_list, values_list) or None if no lab events
        """
        try:
            import polars as pl
        except ImportError:
            return None

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

            # Calculate time from admission start (hours)
            time_from_admission = (
                lab_ts - admission_time
            ).total_seconds() / 3600.0

            lab_times.append(time_from_admission)
            lab_values.append(lab_vector)

        if len(lab_values) == 0:
            return None

        return (lab_times, lab_values)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the multimodal mortality prediction task.

        This task requires ALL modalities to be present for a sample to be valid:
        - Conditions (ICD diagnoses)
        - Procedures (ICD procedures)
        - Drugs (prescriptions)
        - Discharge notes
        - Radiology notes
        - Lab events
        - Chest X-ray images
        - X-ray NegBio findings
        """
        samples = []

        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]
        anchor_age = getattr(demographics, "anchor_age", None)

        # Check age - filter out patients under 18 when possible
        try:
            if anchor_age is not None and int(float(anchor_age)) < 18:
                return []
        except (ValueError, TypeError):
            pass

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            admission = admissions[i]
            next_admission = admissions[i + 1]

            # Check discharge status for mortality label
            if next_admission.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(next_admission.hospital_expire_flag)

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

            # Get clinical codes using hadm_id filtering (more robust than timestamps)
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)]
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)]
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", admission.hadm_id)]
            )

            # Get notes using hadm_id filtering
            discharge_notes = patient.get_events(
                event_type="discharge",
                filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology",
                filters=[("hadm_id", "==", admission.hadm_id)]
            )

            # Get X-ray data
            xrays_negbio = patient.get_events(event_type="xrays_negbio")
            xrays_metadata = patient.get_events(event_type="xrays_metadata")

            # Extract clinical codes
            conditions = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in diagnoses_icd]
            )
            procedures_list = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in procedures_icd]
            )
            drugs = self._clean_sequence(
                [getattr(event, "drug", None) for event in prescriptions]
            )

            # Extract note text
            discharge_text = self._clean_text(
                " ".join([getattr(note, "discharge", "") for note in discharge_notes])
            )
            radiology_text = self._clean_text(
                " ".join([getattr(note, "radiology", "") for note in radiology_notes])
            )

            # Process lab events using timestamp filtering
            labs_data = self._process_lab_events(
                patient, admission.timestamp, admission_dischtime
            )

            # Process X-ray findings
            xray_negbio_features = []
            for xray in xrays_negbio:
                try:
                    findings = []
                    for finding in [
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
                    ]:
                        try:
                            value = getattr(xray, f"{finding}", None)
                            if value is not None:
                                try:
                                    numeric_value = float(value)
                                    if numeric_value > 0:
                                        findings.append(finding)
                                except (ValueError, TypeError):
                                    pass
                        except Exception:
                            pass

                    if findings:
                        xray_negbio_features.extend(findings)
                except Exception:
                    pass

            # Generate image path (use first available X-ray)
            image_path = None
            for xray in xrays_metadata:
                try:
                    study_id = getattr(xray, "study_id", None)
                    dicom_id = getattr(xray, "dicom_id", None)

                    if study_id and dicom_id:
                        image_path = self._construct_image_path(
                            patient.patient_id, study_id, dicom_id
                        )
                        break  # Use first valid image
                except Exception:
                    pass

            # ===== MULTIMODAL REQUIREMENT =====
            # Require ALL modalities to be present for this sample
            # This ensures we have truly multimodal data for each sample

            # Check EHR codes
            if len(conditions) == 0:
                continue
            if len(procedures_list) == 0:
                continue
            if len(drugs) == 0:
                continue

            # Check clinical notes
            if not discharge_text:
                continue
            if not radiology_text:
                continue

            # Check lab events
            if labs_data is None:
                continue

            # Check imaging data
            if not xray_negbio_features:
                continue
            if not image_path:
                continue

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "discharge": discharge_text,
                    "radiology": radiology_text,
                    "labs": labs_data,
                    "xrays_negbio": xray_negbio_features,
                    "image": image_path,
                    "mortality": mortality_label,
                }
            )

        return samples


class MortalityPredictionEICU(BaseTask):
    """Task for predicting mortality using eICU dataset.

    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.

    Features key-value pairs:
    - using diagnosis table (ICD9CM and ICD10CM) as condition codes
    - using physicalExam table as procedure codes
    - using medication table as drugs codes
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

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            visit = admissions[i]
            next_visit = admissions[i + 1]

            # Check discharge status for mortality label
            if next_visit.discharge_status not in ["Alive", "Expired"]:
                mortality_label = 0
            else:
                mortality_label = 0 if next_visit.discharge_status == "Alive" else 1

            # Get clinical codes
            diagnoses = patient.get_events(
                event_type="diagnosis", start=visit.timestamp, end=visit.discharge_time
            )
            physical_exams = patient.get_events(
                event_type="physicalExam",
                start=visit.timestamp,
                end=visit.discharge_time,
            )
            medications = patient.get_events(
                event_type="medication", start=visit.timestamp, end=visit.discharge_time
            )

            conditions = [event.code for event in diagnoses]
            procedures_list = [event.code for event in physical_exams]
            drugs = [event.code for event in medications]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            # TODO: Exclude visits with age < 18

            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": [conditions],
                    "procedures": [procedures_list],
                    "drugs": [drugs],
                    "mortality": mortality_label,
                }
            )

        return samples


class MortalityPredictionEICU2(BaseTask):
    """Task for predicting mortality using eICU dataset with alternative coding.

    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.

    Similar to MortalityPredictionEICU, but with different code mapping:
    - using admissionDx table and diagnosisString under diagnosis table as condition codes
    - using treatment table as procedure codes
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

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            visit = admissions[i]
            next_visit = admissions[i + 1]

            # Check discharge status for mortality label
            if next_visit.discharge_status not in ["Alive", "Expired"]:
                mortality_label = 0
            else:
                mortality_label = 0 if next_visit.discharge_status == "Alive" else 1

            # Get clinical codes
            admission_dx = patient.get_events(
                event_type="admissionDx",
                start=visit.timestamp,
                end=visit.discharge_time,
            )
            diagnosis_events = patient.get_events(
                event_type="diagnosis", start=visit.timestamp, end=visit.discharge_time
            )
            treatments = patient.get_events(
                event_type="treatment", start=visit.timestamp, end=visit.discharge_time
            )

            # Get diagnosis strings from diagnosis events
            diagnosis_strings = list(
                set(
                    [
                        getattr(event, "diagnosisString", "")
                        for event in diagnosis_events
                        if hasattr(event, "diagnosisString") and event.diagnosisString
                    ]
                )
            )

            admission_dx_codes = [event.code for event in admission_dx]
            treatment_codes = [event.code for event in treatments]

            # Combine admission diagnoses and diagnosis strings
            conditions = admission_dx_codes + diagnosis_strings

            # Exclude visits without sufficient codes
            if len(conditions) * len(treatment_codes) == 0:
                continue

            # TODO: Exclude visits with age < 18

            samples.append(
                {
                    "visit_id": visit.visit_id,
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
