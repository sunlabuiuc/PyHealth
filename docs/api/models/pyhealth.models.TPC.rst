pyhealth.models.TPC
===================================

Overview
--------

Contributor: Hasham Ul Haq (huhaq2)

Paper: `Temporal Pointwise Convolutional Networks for Length of Stay
Prediction in the Intensive Care Unit <https://arxiv.org/abs/2007.09483>`_

This contribution adds a PyHealth-adapted implementation of Temporal
Pointwise Convolution (TPC) for sequential EHR modeling. The model follows the
standard ``BaseModel`` interface, supports PyHealth sequence and StageNet-style
temporal processors, and is intended to be evaluated on existing PyHealth LOS
tasks such as the MIMIC-IV temporal length-of-stay pipeline.

Task Modes
----------

TPC uses the task ``output_schema`` from the provided ``SampleDataset`` to
select the output size, loss function, and prediction format through
``BaseModel``. This means the same model implementation can be used with
PyHealth binary, multiclass, multilabel, and regression tasks.

The included MIMIC-IV length-of-stay example uses the existing
``LengthOfStayStageNetMIMIC4`` task, which defines LOS as a 10-class
classification problem. This follows the current PyHealth LOS task interface.
The original TPC paper reports LOS mainly with regression and ordinal metrics;
a regression LOS task can use the same TPC model by defining an output schema
such as ``{"remaining_los": "regression"}``.

Required Data
-------------

The accompanying example uses the existing
``LengthOfStayStageNetMIMIC4`` task with ``MIMIC4Dataset``. For that temporal
LOS pipeline, the MIMIC-IV root passed through ``EHR_ROOT`` should contain:

- ``hosp/patients.csv.gz``
- ``hosp/admissions.csv.gz``
- ``hosp/diagnoses_icd.csv.gz``
- ``hosp/procedures_icd.csv.gz``
- ``hosp/labevents.csv.gz``
- ``hosp/d_labitems.csv.gz``
- ``icu/icustays.csv.gz``

Although the example explicitly requests diagnoses, procedures, and
``labevents``, the PyHealth MIMIC-IV EHR loader also includes core tables such
as ``patients``, ``admissions``, and ``icustays``.

API Reference
-------------

.. autoclass:: pyhealth.models.TPCBlock
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TPCLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TPC
    :members:
    :undoc-members:
    :show-inheritance:
