pyhealth.tasks.yaib_mimic4
==========================

The MIMIC-IV tasks of YAIB (Yet Another ICU Benchmark, van de Water et al.,
ICLR 2024) on :class:`~pyhealth.datasets.MIMIC4EHRDataset`: ICU mortality, acute kidney
injury, sepsis, kidney function and remaining length of stay. All tasks share
the YAIB cohort and feature set (4 static and 48 hourly dynamic variables
from ICU admission), ported from ``ricu`` and the YAIB-cohorts scripts.

.. autoclass:: pyhealth.tasks.YAIBMortalityMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBAKIMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBSepsisMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBKidneyFunctionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBLengthOfStayMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:
