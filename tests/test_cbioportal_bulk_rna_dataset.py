''' Name: Szymon Szymura
    NetID: szymura2
    Paper title: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
    Paper link: https://www.biorxiv.org/content/10.1101/2024.06.18.599483v2.full
    Description: This is the code for a test of custom dataset of bulk RNA-seq data from TCGA. For simplicity data is obtained from cBioPortal datasets website: 
    https://www.cbioportal.org/datasets instead of original TCGA portal. Data from cBioPortal is already pre-processed and organized into a folder for 
    each study, which contains multiple files, including RNA-seq expression as well as patient information and survival along with other data (methylation, CNA)
    While cBioPortal file of RNA-seq data is not a raw counts and rather, quantified expression, it should be compatible with downstream PyHealth models. 
    This code reads in RNA expression and clinical data files from each folder and combines them based on patient ID. Then it selects top x genes shared across cancers
    (folders), subsets each expression file and concatanates into filnal dataset.
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
from pathlib import Path

import pandas as pd

from pyhealth.datasets.cbioportal_bulk_rna import CBioPortalBulkRNADataset


def _write_expression(study_dir: Path, rows: dict) -> None:
    pd.DataFrame(rows).to_csv(
        study_dir / "data_mrna_seq_v2_rsem.txt",
        sep="\t",
        index=False,
    )


def _write_clinical(study_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(
        study_dir / "data_clinical_patient.txt",
        sep="\t",
        index=False,
    )


def test_prepare_metadata_creates_processed_samples(tmp_path: Path) -> None:

    brca = tmp_path / "brca_tcga"
    luad = tmp_path / "luad_tcga"
    brca.mkdir()
    luad.mkdir()

    # expression files are gene x sample
    _write_expression(
        brca,
        {
            "Hugo_Symbol": ["TP53", "EGFR", "MYC", "BRCA1"],
            "TCGA-AB-0001-01": [1.0, 2.0, 3.0, 4.0],
            "TCGA-AB-0002-01": [1.5, 2.5, 3.5, 4.5],
        },
    )
    _write_expression(
        luad,
        {
            "Hugo_Symbol": ["TP53", "EGFR", "MYC", "KRAS"],
            "TCGA-CD-0001-01": [10.0, 20.0, 30.0, 40.0],
            "TCGA-CD-0002-01": [11.0, 21.0, 31.0, 41.0],
        },
    )

    _write_clinical(
        brca,
        [
            {
                "PATIENT_ID": "TCGA-AB-0001",
                "OS_MONTHS": 10.0,
                "OS_STATUS": "1:DECEASED",
                "DFS_MONTHS": 5.0,
                "DFS_STATUS": "1:Recurred/Progressed",
                "SUBTYPE": "LumA",
            },
            {
                "PATIENT_ID": "TCGA-AB-0002",
                "OS_MONTHS": 40.0,
                "OS_STATUS": "0:LIVING",
                "DFS_MONTHS": 18.0,
                "DFS_STATUS": "0:DiseaseFree",
                "SUBTYPE": "Basal",
            },
        ],
    )
    _write_clinical(
        luad,
        [
            {
                "PATIENT_ID": "TCGA-CD-0001",
                "OS_MONTHS": 8.0,
                "OS_STATUS": "1:DECEASED",
                "DFS_MONTHS": 4.0,
                "DFS_STATUS": "1:Recurred/Progressed",
            },
            {
                "PATIENT_ID": "TCGA-CD-0002",
                "OS_MONTHS": 25.0,
                "OS_STATUS": "0:LIVING",
                "DFS_MONTHS": 10.0,
                "DFS_STATUS": "0:DiseaseFree",
            },
        ],
    )

    CBioPortalBulkRNADataset.prepare_metadata(
        root=str(tmp_path),
        study_dirs=["brca_tcga", "luad_tcga"],
        top_k_genes=2,
    )

    out_csv = tmp_path / "cbioportal_bulk_rna_samples-pyhealth.csv"
    genes_txt = tmp_path / "cbioportal_bulk_rna_selected_genes.txt"

    assert out_csv.exists()
    assert genes_txt.exists()

    df = pd.read_csv(out_csv)

    # two patients from each cohort
    assert len(df) == 4

    # expression should be serialized
    assert "expression_json" in df.columns
    first_vec = json.loads(df.loc[0, "expression_json"])
    assert len(first_vec) == 2

    # cancer labels are created from folder names
    assert set(df["cancer_type"]) == {"brca_tcga", "luad_tcga"}

    # subtype should exist as a column even if missing for some cohorts
    assert "subtype" in df.columns


def test_dataset_init_runs_preprocessing(tmp_path: Path) -> None:

    study = tmp_path / "brca_tcga"
    study.mkdir()

    _write_expression(
        study,
        {
            "Hugo_Symbol": ["TP53", "EGFR"],
            "TCGA-AB-0001-01": [1.0, 2.0],
        },
    )
    _write_clinical(
        study,
        [
            {
                "PATIENT_ID": "TCGA-AB-0001",
                "OS_MONTHS": 12.0,
                "OS_STATUS": "0:LIVING",
                "DFS_MONTHS": 6.0,
                "DFS_STATUS": "0:DiseaseFree",
            }
        ],
    )

    dataset = CBioPortalBulkRNADataset(
        root=str(tmp_path),
        study_dirs=["brca_tcga"],
        top_k_genes=2,
    )

    assert dataset is not None
    assert (tmp_path / "cbioportal_bulk_rna_samples-pyhealth.csv").exists()