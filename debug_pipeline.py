from pyhealth.datasets.cbioportal_bulk_rna import CBioPortalBulkRNADataset
from pyhealth.tasks.bulk_rna_survival import BulkRNASurvivalPrediction

dataset = CBioPortalBulkRNADataset(
    root="/Users/szymonszymura/Desktop/cancer_datasets/",
    study_dirs=["brca_tcga", "luad_tcga"],
    top_k_genes=1000,
)

task = BulkRNASurvivalPrediction()

sample_dataset = dataset.set_task(task)

print("Number of samples:", len(sample_dataset))

# print one sample
print(sample_dataset[0])