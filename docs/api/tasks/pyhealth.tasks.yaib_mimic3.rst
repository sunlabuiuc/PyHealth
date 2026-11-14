pyhealth.tasks.yaib_mimic3
==========================

The MIMIC-III tasks of YAIB (Yet Another ICU Benchmark, van de Water et al.,
ICLR 2024) on :class:`~pyhealth.datasets.MIMIC3Dataset`: ICU mortality, acute kidney
injury, sepsis, kidney function and remaining length of stay. All tasks share
the YAIB cohort and feature set (4 static and 48 hourly dynamic variables
from ICU admission), ported from ``ricu`` and the YAIB-cohorts scripts.

.. autoclass:: pyhealth.tasks.YAIBMortalityMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBAKIMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBSepsisMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBKidneyFunctionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBLengthOfStayMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:
