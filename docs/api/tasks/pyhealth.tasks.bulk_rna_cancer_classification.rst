pyhealth.tasks.bulk_rna_cancer_classification
=============================================

Overview
--------

``BulkRNACancerClassification`` defines a supervised learning task for
bulk RNA-seq data derived from cBioPortal cancer studies.

The task operates on samples produced by ``CBioPortalBulkRNADataset``,
where each patient is represented by a vector of standardized gene
expression values (top variable genes selected during preprocessing).

By default, the task performs multiclass classification using the
``cancer_type`` field as the target. It can also be configured for
within-cancer subtype prediction by setting ``label_field="subtype"``.

For each patient, the task generates one sample:
- ``x``: gene expression vector (parsed from serialized JSON)
- ``y``: categorical label (cancer type or subtype)

Samples with missing or empty labels are skipped.

API Reference
-------------

.. autoclass:: pyhealth.tasks.BulkRNACancerClassification
    :members:
    :undoc-members:
    :show-inheritance: