from pyhealth.datasets.cbioportal_bulk_rna import CBioPortalBulkRNADataset
from pyhealth.tasks.bulk_rna_survival import BulkRNASurvivalPrediction

dataset = CBioPortalBulkRNADataset(
    root="/Users/szymonszymura/Desktop/cancer_datasets/",
    study_dirs=["brca_tcga", "luad_tcga"],
    top_k_genes=1000,
)

task = BulkRNASurvivalPrediction()

# get one patient
patient = dataset.get_patient(dataset.unique_patient_ids[0])

samples = task(patient)

print(samples)