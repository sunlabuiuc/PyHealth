from pyhealth.data import Patient, Visit


def mortality_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import mortality_prediction_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(mortality_prediction_mimic3_fn)
        >>> mimic3_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 0}]
    """
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction_mimic4_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import mortality_prediction_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(mortality_prediction_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction_eicu_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Features key-value pairs:
    - using diagnosis table (ICD9CM and ICD10CM) as condition codes
    - using physicalExam table as procedure codes
    - using medication table as drugs codes

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "physicalExam"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import mortality_prediction_eicu_fn
        >>> eicu_sample = eicu_base.set_task(mortality_prediction_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 0}]
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in ["Alive", "Expired"]:
            mortality_label = 0
        else:
            mortality_label = 0 if next_visit.discharge_status == "Alive" else 1

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def mortality_prediction_eicu_fn2(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Similar to mortality_prediction_eicu_fn, but with different code mapping:
    - using admissionDx table and diagnosisString under diagnosis table as condition codes
    - using treatment table as procedure codes

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "admissionDx", "treatment"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import mortality_prediction_eicu_fn2
        >>> eicu_sample = eicu_base.set_task(mortality_prediction_eicu_fn2)
        >>> eicu_sample.samples[0]
        {'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 0}
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in ["Alive", "Expired"]:
            mortality_label = 0
        else:
            mortality_label = 0 if next_visit.discharge_status == "Alive" else 1

        admissionDx = visit.get_code_list(table="admissionDx")
        diagnosisString = list(
            set(
                [
                    dx.attr_dict["diagnosisString"]
                    for dx in visit.get_event_list("diagnosis")
                ]
            )
        )
        treatment = visit.get_code_list(table="treatment")

        # exclude: visits without treatment, admissionDx, diagnosisString
        if len(admissionDx) + len(diagnosisString) * len(treatment) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": admissionDx + diagnosisString,
                "procedures": treatment,
                "label": mortality_label,
            }
        )
    print(samples)
    # no cohort selection
    return samples


def mortality_prediction_omop_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import mortality_prediction_omop_fn
        >>> omop_sample = omop_base.set_task(mortality_prediction_eicu_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # labs = visit.get_code_list(table="measurement")

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset

    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"ICD9CM": "CCSCM", "NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_mimic3_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import eICUDataset

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_eicu_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "admissionDx", "treatment"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_eicu_fn2)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

    from pyhealth.datasets import OMOPDataset

    base_dataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_omop_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
