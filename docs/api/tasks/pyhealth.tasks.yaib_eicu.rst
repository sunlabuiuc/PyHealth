pyhealth.tasks.yaib_eicu
========================

The eICU-CRD tasks of YAIB (Yet Another ICU Benchmark, van de Water et al.,
ICLR 2024) on :class:`~pyhealth.datasets.eICUDataset`: ICU mortality, acute kidney
injury, sepsis, kidney function and remaining length of stay. All tasks share
the YAIB cohort and feature set (4 static and 48 hourly dynamic variables
from ICU admission), ported from ``ricu`` and the YAIB-cohorts scripts.

.. autoclass:: pyhealth.tasks.YAIBMortalityEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBAKIEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBSepsisEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBKidneyFunctionEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.YAIBLengthOfStayEICU
    :members:
    :undoc-members:
    :show-inheritance:
