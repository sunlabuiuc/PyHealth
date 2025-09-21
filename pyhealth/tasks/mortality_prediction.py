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
    """Task for predicting mortality using MIMIC-IV multimodal data including chest X-rays."""

    task_name: str = "MultimodalMortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
        "discharge": "text",
        "radiology": "text",
        "xrays_negbio": "sequence",
        "image_paths": "text",  # Added image paths to the schema
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

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
        """
        Constructs the relative path to a MIMIC-CXR image file based on the folder structure.

        Args:
            subject_id: The patient/subject ID (e.g., "10000032")
            study_id: The study ID (e.g., "50414267")
            dicom_id: The DICOM ID (e.g., "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014")

        Returns:
            The relative path to the image file
        """
        # Extract first two characters of the patient_id for the parent folder
        parent_folder = f"p{subject_id[0][:2]}"

        # Format the complete patient ID path component
        patient_folder = f"p{subject_id[0]}"

        # Construct the complete path
        return f"files/{parent_folder}/{patient_folder}/s{study_id}/{dicom_id}.jpg"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task."""
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

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
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

            # Get notes
            discharge_notes = patient.get_events(
                event_type="discharge",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            radiology_notes = patient.get_events(
                event_type="radiology",
                start=admission.timestamp,
                end=admission_dischtime,
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
                            # Convert the value to float first, then to int
                            # This handles both string and numeric representations
                            value = getattr(xray, f"{finding}", None)

                            # Convert to float first to handle string representations like '1.0'
                            if value is not None:
                                try:
                                    numeric_value = float(value)
                                    # Check if the numeric value is non-zero
                                    if numeric_value > 0:
                                        findings.append(finding)
                                except (ValueError, TypeError):
                                    # If conversion fails, skip this finding
                                    pass
                        except Exception as sub_e:
                            print(f"Error processing finding {finding}: {sub_e}")

                    # Extend the features list with findings for this X-ray
                    if findings:
                        xray_negbio_features.extend(findings)

                except Exception as e:
                    print(f"Error processing X-ray NegBio feature: {e}")

            # Generate image paths
            image_paths = []
            for xray in xrays_metadata:
                try:
                    study_id = getattr(xray, "study_id", None)
                    dicom_id = getattr(xray, "dicom_id", None)

                    if study_id and dicom_id:
                        image_path = self._construct_image_path(
                            f"p{patient.patient_id[0]}", study_id, dicom_id
                        )
                        image_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing X-ray image path: {e}")
            # Exclude visits without sufficient clinical data
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
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
                    "xrays_negbio": xray_negbio_features,
                    "image_paths": image_paths,
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
    """Task for predicting mortality using OMOP dataset.

    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.
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
            List[Dict[str, Any]]: A list of samples, each sample is a dict with
                patient_id, visit_id, conditions, procedures, drugs and mortality.
        """
        samples = []

        # Get visits
        visits = patient.get_events(event_type="visit_occurrence")
        if len(visits) <= 1:
            return []

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Check discharge status for mortality label
            mortality_label = int(next_visit.discharge_status)

            # Get clinical codes
            conditions = patient.get_events(
                event_type="condition_occurrence",
                start=visit.timestamp,
                end=visit.discharge_time,
            )
            procedures = patient.get_events(
                event_type="procedure_occurrence",
                start=visit.timestamp,
                end=visit.discharge_time,
            )
            drugs = patient.get_events(
                event_type="drug_exposure",
                start=visit.timestamp,
                end=visit.discharge_time,
            )

            condition_codes = [event.code for event in conditions]
            procedure_codes = [event.code for event in procedures]
            drug_codes = [event.code for event in drugs]

            # Exclude visits without condition, procedure, or drug code
            if len(condition_codes) * len(procedure_codes) * len(drug_codes) == 0:
                continue

            # TODO: Exclude visits with age < 18

            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": [condition_codes],
                    "procedures": [procedure_codes],
                    "drugs": [drug_codes],
                    "mortality": mortality_label,
                }
            )

        return samples
