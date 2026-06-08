pyhealth.tasks.bulk_rna_survival_prediction
===========================================

Overview
--------

``BulkRNASurvivalPrediction`` defines a survival prediction task for bulk
RNA-seq data derived from cBioPortal cancer studies.

The task operates on samples produced by ``CBioPortalBulkRNADataset``,
where each patient is represented by a vector of standardized gene
expression values. It uses overall survival information as the target,
specifically survival time in months and survival status.

This task converts survival prediction into a multiclass classification
problem by binning overall survival time into discrete intervals. By
default, the survival bins are:
Survival is converted into three bins:
    0: survival time less than first bin edge
    1: first bin edge less or equal to survival time less than second bin edge
    2: survival time longer or equal to second bin edge

For each patient, the task generates one sample containing:
- ``x``: gene expression vector
- ``y``: discretized survival class
- ``survival_months``: raw survival time in months
- ``event``: event indicator derived from survival status

Samples with missing expression values, missing survival time, or missing
survival status are skipped.

This task is intended as a simplified classification-style survival
benchmark and does not implement Cox proportional hazards or other
time-to-event survival modeling approaches.

API Reference
-------------

.. autoclass:: pyhealth.tasks.BulkRNASurvivalPrediction
    :members:
    :undoc-members:
    :show-inheritance: