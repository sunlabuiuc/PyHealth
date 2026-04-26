pyhealth.tasks.EHREvidenceRetrievalTask
=========================================

Binary task that pairs a patient's concatenated clinical notes with a
free-text query diagnosis.  The label indicates whether the patient has been
assigned any of the specified ICD-9 codes, providing a computable proxy for
expert ground-truth labels.

This task is designed to be used with
:class:`~pyhealth.datasets.MIMIC3NoteDataset` and
:class:`~pyhealth.models.ZeroShotEvidenceLLM` to reproduce the zero-shot EHR
evidence retrieval pipeline from `Ahsan et al. (2024)
<https://arxiv.org/abs/2309.04550>`_.

.. autoclass:: pyhealth.tasks.EHREvidenceRetrievalTask
    :members:
    :undoc-members:
    :show-inheritance:
