pyhealth.tasks.generate_ehr
===========================================

Task that turns a longitudinal EHR dataset into per-patient, per-visit code
sequences for training unconditional synthetic-EHR generators (HALO, GPT2,
PromptEHR, MedGAN, CorGAN), plus helpers to flatten generated output into the
long-form dataframe consumed by :mod:`pyhealth.metrics.generative`.

Task Classes
------------

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGeneration
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGenerationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGenerationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

Helper Functions
----------------

.. autofunction:: pyhealth.tasks.generate_ehr.decode_dataset

.. autofunction:: pyhealth.tasks.generate_ehr.to_evaluation_dataframe
