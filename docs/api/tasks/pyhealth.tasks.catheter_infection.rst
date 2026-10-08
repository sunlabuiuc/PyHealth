pyhealth.tasks.catheter_infection
=================================

Catheter-associated urinary tract infection (CAUTI) prediction on MIMIC-IV.

The ICD-based tasks label an admission from discharge diagnosis codes. The
temporal tasks follow the NHSN SUTI 1a timing rules. Each sample is an
admission with a timed indwelling catheter (CPT 51702/51703 from
``hcpcsevents``, or ICU Foley charting from ``procedureevents`` and
``outputevents``) that was in place for more than 2 consecutive days, with the
eligible day on hospital day 3 or later. The label is the union of a
catheter-specific ICD code, a general urinary infection code, and a urine
culture (``microbiologyevents``) collected inside the eligible catheter window.

.. autoclass:: pyhealth.tasks.catheter_infection.CatheterAssociatedInfectionPredictionMIMIC4Temporal
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.catheter_infection.CatheterAssociatedInfectionPredictionStageNetMIMIC4Temporal
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.catheter_infection.CatheterAssociatedInfectionPredictionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.catheter_infection.CatheterAssociatedInfectionPredictionStageNetMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.catheter_infection.CatheterAssociatedInfectionPredictionMIMIC4DualContext
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.catheter_infection.CatheterAssociatedInfectionPredictionStageNetMIMIC4DualContext
    :members:
    :undoc-members:
    :show-inheritance:
