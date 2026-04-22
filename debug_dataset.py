from pyhealth.datasets.cbioportal_bulk_rna import CBioPortalBulkRNADataset

dataset = CBioPortalBulkRNADataset(
    root="/Users/szymonszymura/Desktop/cancer_datasets/",   # <-- IMPORTANT
    study_dirs=["brca_tcga", "luad_tcga"],
    top_k_genes=1000,
)

print("Dataset created!")
dataset.stats()