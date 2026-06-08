pyhealth.datasets.cbioportal_bulk_rna
=====================================

Overview
--------

``CBioPortalBulkRNADataset`` provides access to bulk RNA-seq data collected
from cBioPortal cancer studies. The dataset expects one subfolder per study
under the root directory, where each study contains an RNA expression file
(``data_mrna_seq_v2_rsem.txt``) and a clinical patient file
(``data_clinical_patient.txt``).

During preprocessing, the dataset loads expression and clinical tables from
multiple studies, derives patient identifiers from sample barcodes, finds
genes shared across studies, standardizes expression values, keeps the top
variable genes, merges expression with clinical metadata, and writes the
result into a PyHealth-ready samples CSV. It then loads the processed data
through ``BaseDataset`` using the associated YAML configuration.

By default, this dataset uses ``BulkRNACancerClassification`` as its default
task.

API Reference
-------------

.. autoclass:: pyhealth.datasets.CBioPortalBulkRNADataset
    :members:
    :undoc-members:
    :show-inheritance: