from pyhealth.data import Patient


def patient_linkage_mimic3_fn(patient: Patient):
    """ Patient linkage task for the mimic3 dataset. """

    # exclude patients with less than two visits
    if len(patient) < 2:
        return []

    samples = []
    q_visit = patient.get_visit_by_index(len(patient) - 1)
    d_visit = patient.get_visit_by_index(len(patient) - 2)

    q_age = (q_visit.encounter_time - patient.birth_datetime).days // 365.25
    d_age = (d_visit.encounter_time - patient.birth_datetime).days // 365.25

    # exclude patients under 18
    if (q_age < 18) or (d_age < 18):
        return []

    q_conditions = q_visit.get_code_list(table="DIAGNOSES_ICD")
    d_conditions = d_visit.get_code_list(table="DIAGNOSES_ICD")

    # exclude patients without conditions
    if len(q_conditions) * len(d_conditions) == 0:
        return []

    # identifiers
    gender = patient.gender
    insurance = q_visit.attr_dict["insurance"]
    language = q_visit.attr_dict["language"]
    religion = q_visit.attr_dict["religion"]
    marital_status = q_visit.attr_dict["marital_status"]
    ethnicity = q_visit.attr_dict["ethnicity"]
    insurance = "" if insurance != insurance else insurance
    language = "" if language != language else language
    religion = "" if religion != religion else religion
    marital_status = "" if marital_status != marital_status else marital_status
    ethnicity = "" if ethnicity != ethnicity else ethnicity
    q_identifiers = "+".join(
        [gender, insurance, language, religion, marital_status, ethnicity]
    )

    insurance = d_visit.attr_dict["insurance"]
    language = d_visit.attr_dict["language"]
    religion = d_visit.attr_dict["religion"]
    marital_status = d_visit.attr_dict["marital_status"]
    ethnicity = d_visit.attr_dict["ethnicity"]
    insurance = "" if insurance != insurance else insurance
    language = "" if language != language else language
    religion = "" if religion != religion else religion
    marital_status = "" if marital_status != marital_status else marital_status
    ethnicity = "" if ethnicity != ethnicity else ethnicity
    d_identifiers = "+".join(
        [gender, insurance, language, religion, marital_status, ethnicity]
    )

    samples.append({
        "patient_id": patient.patient_id,
        "visit_id": q_visit.visit_id,
        "conditions": ["<cls>"] + q_conditions,
        "age": q_age,
        "identifiers": q_identifiers,
        "d_visit_id": d_visit.visit_id,
        "d_conditions": ["<cls>"] + d_conditions,
        "d_age": d_age,
        "d_identifiers": d_identifiers,
    })

    return samples
