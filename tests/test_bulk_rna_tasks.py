''' Name: Szymon Szymura
    NetID: szymura2
    Paper title: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
    Paper link: https://www.biorxiv.org/content/10.1101/2024.06.18.599483v2.full
    Description: This is the code for test of cancer classification task and cancer survival task using custom dataset of bulk RNA-seq data from TCGA. For simplicity data is obtained from cBioPortal datasets website: 
    https://www.cbioportal.org/datasets instead of original TCGA portal. Classification task uses cancer type as target variable and RNA-seq expression of top x genes as features
    for multiclass classification. Survival task uses patient survival information as target variable and RNA-seq expression of top x genes 
    as features. Patient survival data is binned into bins and multiclass classification is performed.
    Files to review:
    - pyhealth/datasets/cbioportal_bulk_rna.py
    - pyhealth/tasks/bulk_rna_classification.py
    - pyhealth/tasks/bulk_rna_survival.py
    - tests/test_cbioportal_bulk_rna_dataset.py
    - tests/test_bulk_rna_tasks.py
    - examples/cbioportal_bulk_rna_showcase.ipynb
    - docs/api/datasets/pyhealth.datasets.cbioportal_bulk_rna.rst
    - docs/api/tasks/pyhealth.tasks.bulk_rna_cancer_classification.rst
    - docs/api/tasks/pyhealth.tasks.bulk_rna_survival_prediction.rst
'''



import json

import polars as pl

from pyhealth.tasks.bulk_rna_classification import BulkRNACancerClassification
from pyhealth.tasks.bulk_rna_survival import BulkRNASurvivalPrediction


class FakePatient:

    def __init__(self, patient_id: str, row: dict):
        self.patient_id = patient_id
        self.data_source = pl.DataFrame([row])


def test_cancer_type_classification_task() -> None:
    patient = FakePatient(
        patient_id="P1",
        row={
            "samples/expression_json": json.dumps([0.1, 0.2, 0.3]),
            "samples/cancer_type": "brca_tcga",
            "samples/subtype": "LumA",
        },
    )

    task = BulkRNACancerClassification(label_field="cancer_type")
    samples = task(patient)

    assert len(samples) == 1
    assert samples[0]["patient_id"] == "P1"
    assert samples[0]["y"] == "brca_tcga"
    assert len(samples[0]["x"]) == 3


def test_subtype_classification_task() -> None:
    patient = FakePatient(
        patient_id="P1",
        row={
            "samples/expression_json": json.dumps([1.0, 2.0]),
            "samples/cancer_type": "brca_tcga",
            "samples/subtype": "Basal",
        },
    )

    task = BulkRNACancerClassification(label_field="subtype")
    samples = task(patient)

    assert len(samples) == 1
    assert samples[0]["y"] == "Basal"


def test_survival_task_bins_correctly() -> None:
    task = BulkRNASurvivalPrediction()

    p_short = FakePatient(
        patient_id="P1",
        row={
            "samples/expression_json": json.dumps([0.1, 0.2]),
            "samples/os_months": 6.0,
            "samples/os_status": "1:DECEASED",
        },
    )
    p_mid = FakePatient(
        patient_id="P2",
        row={
            "samples/expression_json": json.dumps([0.1, 0.2]),
            "samples/os_months": 18.0,
            "samples/os_status": "0:LIVING",
        },
    )
    p_long = FakePatient(
        patient_id="P3",
        row={
            "samples/expression_json": json.dumps([0.1, 0.2]),
            "samples/os_months": 48.0,
            "samples/os_status": "0:LIVING",
        },
    )

    s1 = task(p_short)[0]
    s2 = task(p_mid)[0]
    s3 = task(p_long)[0]

    assert s1["y"] == 0
    assert s2["y"] == 1
    assert s3["y"] == 2

    assert s1["event"] == 1
    assert s2["event"] == 0


def test_survival_task_skips_missing_values() -> None:
    task = BulkRNASurvivalPrediction()

    patient = FakePatient(
        patient_id="P1",
        row={
            "samples/expression_json": json.dumps([0.1, 0.2]),
            "samples/os_months": None,
            "samples/os_status": "1:DECEASED",
        },
    )

    samples = task(patient)
    assert samples == []