pyhealth.tasks.cbis_ddsm_classification
=======================================

Image-level classification tasks for the CBIS-DDSM mammography dataset:
malignant vs benign pathology, and the BI-RADS assessment (1-5). Both read the
full-field mammogram lazily with the ``dicom`` processor.

.. autoclass:: pyhealth.tasks.CBISDDSMPathologyClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.CBISDDSMBIRADSClassification
    :members:
    :undoc-members:
    :show-inheritance:
