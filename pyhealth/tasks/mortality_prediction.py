from datetime import datetime
from typing import Any, Dict, List, Optional
from .base_task import BaseTask


class MortalityPredictionMIMIC3(BaseTask):
    """Task for predicting mortality using MIMIC-III dataset with text data.
    
    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.
    """
    task_name: str = "MortalityPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence", 
        "procedures": "sequence", 
        "drugs": "sequence",
        "clinical_notes": "text"  # Added support for clinical notes
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task.

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a dict with
                patient_id, visit_id, conditions, procedures, drugs, notes and mortality.
        """
        samples = []

        # We will drop the last visit
        visits = patient.get_events(event_type="admissions")
        if len(visits) <= 1:
            return []

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Check discharge status for mortality label - more robust handling
            try:
                mortality_label = int(next_visit.discharge_status) if next_visit.discharge_status in [0, 1] else 0
            except (ValueError, AttributeError):
                mortality_label = 0

            # Get clinical codes
            diagnoses = patient.get_events(
                event_type="DIAGNOSES_ICD",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            procedures = patient.get_events(
                event_type="PROCEDURES_ICD",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            medications = patient.get_events(
                event_type="PRESCRIPTIONS",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            
            # Get clinical notes
            notes = patient.get_events(
                event_type="NOTEEVENTS",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            
            conditions = [event.code for event in diagnoses]
            procedures_list = [event.code for event in procedures]
            drugs = [event.code for event in medications]
            
            # Extract note text - concatenate if multiple exist
            note_text = " ".join([getattr(note, "code", "") for note in notes])
            if not note_text.strip():
                note_text = None

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue
            
            samples.append({
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures_list],
                "drugs": [drugs],
                "clinical_notes": note_text,
                "mortality": mortality_label,
            })
        
        return samples

class MortalityPredictionMIMIC4(BaseTask):
    """Task for predicting mortality using MIMIC-IV EHR data only."""
    task_name: str = "MortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence", 
        "procedures": "sequence", 
        "drugs": "sequence"
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
        cleaned = [str(item).strip() for item in sequence if item is not None and str(item).strip()]
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
            try:
                mortality_label = int(next_admission.hospital_expire_flag)
            except (ValueError, AttributeError):
                # Default to 0 if value can't be interpreted
                mortality_label = 0

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If date parsing fails, skip this admission
                continue

            # Get clinical codes
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime
            )
            
            # Extract relevant data
            conditions = self._clean_sequence([
                getattr(event, 'icd_code', None) for event in diagnoses_icd
            ])
            procedures_list = self._clean_sequence([
                getattr(event, 'icd_code', None) for event in procedures_icd
            ])
            drugs = self._clean_sequence([
                getattr(event, 'drug', None) for event in prescriptions
            ])

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append({
                "visit_id": admission.hadm_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures_list,
                "drugs": drugs,
                "mortality": mortality_label,
            })
        
        return samples


class MultimodalMortalityPredictionMIMIC4(BaseTask):
    """Task for predicting mortality using MIMIC-IV multimodal data."""
    task_name: str = "MultimodalMortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence", 
        "procedures": "sequence", 
        "drugs": "sequence",
        "discharge": "text",      
        "radiology": "text",      
        "xrays_negbio": "sequence"      
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
        cleaned = [str(item).strip() for item in sequence if item is not None and str(item).strip()]
        return cleaned

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """
        Clean text by:
        1. Stripping whitespace
        2. Returning None if empty
        """
        if text is None:
            return None
        
        cleaned_text = str(text).strip()
        return cleaned_text if cleaned_text else None

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
            try:
                mortality_label = int(next_admission.hospital_expire_flag)
            except (ValueError, AttributeError):
                # Default to 0 if value can't be interpreted
                mortality_label = 0

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If date parsing fails, skip this admission
                continue

            # Get clinical codes
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime
            )
            
            # Get discharge notes if available
            discharge_notes = patient.get_events(
                event_type="discharge",
                start=admission.timestamp,
                end=admission_dischtime
            )
            
            # Get radiology notes if available
            radiology_notes = patient.get_events(
                event_type="radiology",
                start=admission.timestamp,
                end=admission_dischtime
            )
            
            # Get X-ray labels if available
            xrays_negbio = patient.get_events(
                event_type="xrays_negbio"
            )
            
            # Extract relevant data
            conditions = self._clean_sequence([
                getattr(event, 'icd_code', None) for event in diagnoses_icd
            ])
            procedures_list = self._clean_sequence([
                getattr(event, 'icd_code', None) for event in procedures_icd
            ])
            drugs = self._clean_sequence([
                getattr(event, 'drug', None) for event in prescriptions
            ])
            
            # Extract text data - concatenate multiple notes if present
            discharge_text = self._clean_text(" ".join([
                getattr(note, "text", "") for note in discharge_notes
            ]))
            radiology_text = self._clean_text(" ".join([
                getattr(note, "text", "") for note in radiology_notes
            ]))
            
            # Process X-ray features 
            xray_negbio_features = []
            if xrays_negbio:
                for xray in xrays_negbio:
                    try:
                        # Collect all non-empty, non-zero findings
                        findings = []
                        for finding in [
                            "no finding", "enlarged cardiomediastinum", "cardiomegaly", 
                            "lung opacity", "lung lesion", "edema", "consolidation", 
                            "pneumonia", "atelectasis", "pneumothorax", 
                            "pleural effusion", "pleural other", "fracture", 
                            "support devices"
                        ]:
                            value = getattr(xray, finding, None)
                            # Only add non-zero, non-None findings
                            if value and value != 0 and value != '0':
                                findings.append(f"{finding}:{value}")
                        
                        # Only add non-empty feature lists
                        if findings:
                            xray_negbio_features.extend(findings)
                    except Exception as e:
                        print(f"Error processing X-ray NegBio feature: {e}")

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append({
                "visit_id": admission.hadm_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures_list,
                "drugs": drugs,
                "discharge": discharge_text,
                "radiology": radiology_text,
                "xrays_negbio": xray_negbio_features if xray_negbio_features else None,
                "mortality": mortality_label,
            })
        
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
        "drugs": "sequence"
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
                event_type="diagnosis",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            physical_exams = patient.get_events(
                event_type="physicalExam",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            medications = patient.get_events(
                event_type="medication",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            
            conditions = [event.code for event in diagnoses]
            procedures_list = [event.code for event in physical_exams]
            drugs = [event.code for event in medications]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue
            
            # TODO: Exclude visits with age < 18

            samples.append({
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures_list],
                "drugs": [drugs],
                "mortality": mortality_label,
            })
        
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
    input_schema: Dict[str, str] = {
        "conditions": "sequence", 
        "procedures": "sequence"
    }
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
                end=visit.discharge_time
            )
            diagnosis_events = patient.get_events(
                event_type="diagnosis",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            treatments = patient.get_events(
                event_type="treatment",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            
            # Get diagnosis strings from diagnosis events
            diagnosis_strings = list(set([
                getattr(event, "diagnosisString", "") 
                for event in diagnosis_events
                if hasattr(event, "diagnosisString") and event.diagnosisString
            ]))
            
            admission_dx_codes = [event.code for event in admission_dx]
            treatment_codes = [event.code for event in treatments]
            
            # Combine admission diagnoses and diagnosis strings
            conditions = admission_dx_codes + diagnosis_strings

            # Exclude visits without sufficient codes
            if len(conditions) * len(treatment_codes) == 0:
                continue
            
            # TODO: Exclude visits with age < 18

            samples.append({
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": treatment_codes,
                "mortality": mortality_label,
            })
        
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
        "drugs": "sequence"
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
                end=visit.discharge_time
            )
            procedures = patient.get_events(
                event_type="procedure_occurrence",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            drugs = patient.get_events(
                event_type="drug_exposure",
                start=visit.timestamp,
                end=visit.discharge_time
            )
            
            condition_codes = [event.code for event in conditions]
            procedure_codes = [event.code for event in procedures]
            drug_codes = [event.code for event in drugs]

            # Exclude visits without condition, procedure, or drug code
            if len(condition_codes) * len(procedure_codes) * len(drug_codes) == 0:
                continue
            
            # TODO: Exclude visits with age < 18

            samples.append({
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [condition_codes],
                "procedures": [procedure_codes],
                "drugs": [drug_codes],
                "mortality": mortality_label,
            })
        
        return samples
    

    