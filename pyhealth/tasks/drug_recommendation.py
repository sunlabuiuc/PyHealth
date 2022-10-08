from pyhealth.data import Patient, Visit
from pyhealth.tasks.utils import get_code_from_list_of_event


def drug_recommendation_mimic3_fn(patient: Patient):
    """
    Drug recommendation aims at recommending a set of drugs given the patient health history  (e.g., conditions
    and procedures).

    Process a single patient for the drug recommendation task.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
         attributes as key

    Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
    to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
    from the task dataset by returning an empty list.
    """

    samples = []
    for visit in patient:
        visit: Visit
        conditions = get_code_from_list_of_event(visit.get_event_list(event_type="DIAGNOSES_ICD"))
        procedures = get_code_from_list_of_event(visit.get_event_list(event_type="PROCEDURES_ICD"))
        drugs = get_code_from_list_of_event(visit.get_event_list(event_type="PRESCRIPTIONS"))
        # exclude: visits without (condition or procedure) or drug code
        if (len(conditions) + len(procedures)) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append({"visit_id": visit.visit_id,
                        "patient_id": patient.patient_id,
                        "conditions": conditions,
                        "procedures": procedures,
                        "drugs": drugs})
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
    return samples


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    mimic3dataset = MIMIC3Dataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4",
                                  tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
                                  dev=True,
                                  code_mapping={"PRESCRIPTIONS": "ATC3"},
                                  refresh_cache=False)
    mimic3dataset.stat()
    mimic3dataset.set_task("drug_recommendation", drug_recommendation_mimic3_fn)
    mimic3dataset.stat()
