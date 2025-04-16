from datetime import datetime
from typing import Any, Dict, List

from .base_task import BaseTask


class MortalityPredictionMIMIC3(BaseTask):
    """Task for predicting mortality using MIMIC-III dataset.
    
    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.
    """
    task_name: str = "MortalityPredictionMIMIC3"
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

        # We will drop the last visit
        visits = patient.get_events(event_type="admissions")
        if len(visits) <= 1:
            return []

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Check discharge status for mortality label
            if next_visit.discharge_status not in [0, 1]:
                mortality_label = 0
            else:
                mortality_label = int(next_visit.discharge_status)

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
            
            conditions = [event.code for event in diagnoses]
            procedures_list = [event.code for event in procedures]
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


class MortalityPredictionMIMIC4(BaseTask):
    """Task for predicting mortality using MIMIC-IV dataset.
    
    This task aims to predict whether the patient will decease in the next hospital
    visit based on clinical information from the current visit.
    """
    task_name: str = "MortalityPredictionMIMIC4"
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

        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        
        demographics = demographics[0]
        anchor_age = getattr(demographics, "anchor_age", None)
        
        if anchor_age is not None and int(anchor_age) < 18:
            return []  # Skip patients under 18

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            admission = admissions[i]
            next_admission = admissions[i + 1]

            # Check discharge status for mortality label
            if next_admission.hospital_expire_flag not in [0, 1]:
                mortality_label = 0
            else:
                mortality_label = int(next_admission.hospital_expire_flag)

            # Parse admission timestamps
            admission_dischtime = datetime.strptime(
                admission.dischtime, "%Y-%m-%d %H:%M:%S"
            )

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
            
            conditions = [event.icd_code for event in diagnoses_icd]
            procedures_list = [event.icd_code for event in procedures_icd]
            drugs = [event.drug for event in prescriptions]

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append({
                "visit_id": admission.hadm_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures_list],
                "drugs": [drugs],
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
    

    