pyhealth.tasks.TCGACRCkMSIClassification
========================================

Slide-level MSI classification task for TCGA-CRCk.

This task groups all tile events from the same slide into a single bag for binary MSI classification. It is designed for the TCGA-CRCk dataset and uses PyHealth's 'time_image' input type so each sample can be represented as a bag of image paths with simple monotonic timestamps.

.. autoclass:: pyhealth.tasks.TCGACRCkMSIClassification
    :members:
    :undoc-members:
    :show-inheritance: