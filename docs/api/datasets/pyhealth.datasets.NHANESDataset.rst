pyhealth.datasets.NHANESDataset
===============================

Data are available from the CDC: accelerometer and exam files at
https://wwwn.cdc.gov/nchs/nhanes/ and the linked mortality files at
https://www.cdc.gov/nchs/data-linkage/mortality-public.htm.

In the NHANES 2003-2004 and 2005-2006 cycles, participants wore a hip
ActiGraph accelerometer for 7 days, which recorded one activity count per
minute. The dataset joins these recordings with demographics, body measures,
blood pressure and the NCHS 2019 linked mortality files by ``SEQN``. It keeps
adults (18 and older) who are eligible for mortality linkage and have
accelerometer data, and writes one file of 5-minute epochs per participant.

Non-wear is detected from runs of zero counts. ``nonwear="authors"`` (the
default) uses the MotionAge authors' version of the rule, and
``nonwear="choi"`` uses the published Choi et al. (2011) algorithm.

Use it with :class:`~pyhealth.tasks.NHANESMortalityPrediction`. Without the
real data, ``generate_synthetic_nhanes`` writes small fake versions of the
same tables.

.. autoclass:: pyhealth.datasets.NHANESDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.nhanes_synthetic.generate_synthetic_nhanes
