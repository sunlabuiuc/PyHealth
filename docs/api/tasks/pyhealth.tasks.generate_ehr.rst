pyhealth.tasks.generate_ehr
===========================================

Tasks that turn a longitudinal EHR dataset into training samples for
unconditional synthetic-EHR generators, plus helpers to flatten generated
output into the long-form dataframe consumed by
:mod:`pyhealth.metrics.generative`.

The classes are flat and independent -- one per (model family, dataset). The
extraction is MIMIC-shaped, assuming an ``admissions`` event type and a
``hadm_id`` linking codes to an admission, so the dataset is named in the class
and a task for eICU/OMOP/MEDS belongs alongside these rather than below them.

Match the task to the model: each generator family reads its codes in a
different shape, and handing a model the wrong shape fails silently rather than
loudly.

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Task
     - Encoding
     - Models
   * - ``EHRGenerationMIMIC3`` / ``MIMIC4``
     - one multi-hot row per visit
     - HALO
   * - ``EHRSequenceGenerationMIMIC3`` / ``MIMIC4``
     - per-visit code indices
     - GPT2, PromptEHR
   * - ``EHRCodeSetGenerationMIMIC3`` / ``MIMIC4``
     - one code set per patient
     - MedGAN, CorGAN

Task Classes
------------

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGenerationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGenerationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRSequenceGenerationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRSequenceGenerationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRCodeSetGenerationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRCodeSetGenerationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

Helper Functions
----------------

.. autofunction:: pyhealth.tasks.generate_ehr.decode_dataset

.. autofunction:: pyhealth.tasks.generate_ehr.to_evaluation_dataframe
