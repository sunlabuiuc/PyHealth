pyhealth.datasets.MIMIC3NoteDataset
=====================================

The open Medical Information Mart for Intensive Care III (MIMIC-III) Clinical Notes dataset specialized for NLP and evidence retrieval tasks. This class extends the general 
:class:`~pyhealth.datasets.MIMIC3Dataset` by always loading the ``noteevents`` and ``diagnoses_icd`` tables and exposing the ``iserror`` flag so that erroneous notes can be filtered downstream.  It is designed to pair with 
:class:`~pyhealth.tasks.EHREvidenceRetrievalTask` and
:class:`~pyhealth.models.ZeroShotEvidenceLLM` to reproduce the zero-shot EHR evidence retrieval pipeline of `Ahsan et al. (2024) <https://arxiv.org/abs/2309.04550>`_.

Refer to the `MIMIC-III documentation <https://mimic.mit.edu/docs/iii/>`_ for data access instructions.

.. autoclass:: pyhealth.datasets.MIMIC3NoteDataset
    :members:
    :undoc-members:
    :show-inheritance:
