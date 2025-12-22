from datetime import datetime
from collections import defaultdict
import math
from pyhealth.tasks import BaseTask

class PatientLinkageMIMIC3Task(BaseTask):
    """
    Patient linkage task for MIMIC-III using the Patient/Visit/Event API.

    Produces the same sample keys as the original patient_linkage_mimic3 task
    so pyhealth.models.medlink.convert_to_ir_format works as usual

    Output sample schema:
      - patient_id: ground-truth entity id (equivalent to "master patient record id" in MIMIC)
      - visit_id: query admission id (hadm_id)
      - conditions, age, identifiers: query side
      - d_visit_id: doc admission id (hadm_id)
      - d_conditions, d_age, d_identifiers: doc side
    """

    task_name = "patient_linkage_mimic3"
    input_schema = {"conditions": "sequence", "d_conditions": "sequence"}
    output_schema = {}

    def __call__(self, patient):
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        admissions = sorted(admissions, key=lambda e: e.timestamp)
        q_visit = admissions[-1]
        d_visit = admissions[-2]

        patients_events = patient.get_events(event_type="patients")
        if not patients_events:
            return []
        demo = patients_events[0]

        gender = str(demo.attr_dict.get("gender") or "")

        dob_raw = demo.attr_dict.get("dob")
        birth_dt = None
        if isinstance(dob_raw, datetime):
            birth_dt = dob_raw
        elif dob_raw is not None:
            try:
                birth_dt = datetime.fromisoformat(str(dob_raw))
            except Exception:
                birth_dt = None

        def compute_age(ts):
            if birth_dt is None or ts is None:
                return None
            return int((ts - birth_dt).days // 365.25)

        q_age = compute_age(q_visit.timestamp)
        d_age = compute_age(d_visit.timestamp)
        if q_age is None or d_age is None or q_age < 18 or d_age < 18:
            return []

        diag_events = patient.get_events(event_type="diagnoses_icd")
        hadm_to_codes = defaultdict(list)
        for ev in diag_events:
            hadm = ev.attr_dict.get("hadm_id")
            code = ev.attr_dict.get("icd9_code")
            if hadm is None or code is None:
                continue
            hadm_to_codes[str(hadm)].append(str(code))

        q_hadm = str(q_visit.attr_dict.get("hadm_id"))
        d_hadm = str(d_visit.attr_dict.get("hadm_id"))

        q_conditions = hadm_to_codes.get(q_hadm, [])
        d_conditions = hadm_to_codes.get(d_hadm, [])
        if len(q_conditions) == 0 or len(d_conditions) == 0:
            return []

        def clean(x):
            if x is None:
                return ""
            if isinstance(x, float) and math.isnan(x):
                return ""
            return str(x)

        def build_identifiers(adm_event):
            insurance = clean(adm_event.attr_dict.get("insurance"))
            language = clean(adm_event.attr_dict.get("language"))
            religion = clean(adm_event.attr_dict.get("religion"))
            marital_status = clean(adm_event.attr_dict.get("marital_status"))
            ethnicity = clean(adm_event.attr_dict.get("ethnicity"))
            return "+".join([gender, insurance, language, religion, marital_status, ethnicity])

        sample = {
            "patient_id": patient.patient_id,

            "visit_id": q_hadm,
            "conditions": [""] + q_conditions,
            "age": q_age,
            "identifiers": build_identifiers(q_visit),

            "d_visit_id": d_hadm,
            "d_conditions": [""] + d_conditions,
            "d_age": d_age,
            "d_identifiers": build_identifiers(d_visit),
        }
        return [sample]
