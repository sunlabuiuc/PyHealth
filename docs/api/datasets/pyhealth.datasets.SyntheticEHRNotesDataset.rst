pyhealth.datasets.SyntheticEHRNotesDataset
==========================================

Overview
--------

A fully synthetic, dependency-free corpus of EHR-style clinical note
snippets, labelled with a diagnosis condition and a weak ground-truth
"is positive" flag. The dataset is the recommended entry point for
experimenting with the :class:`~pyhealth.tasks.EvidenceRetrievalMIMIC3`
task when MIMIC-III access is not available.

The corpus ships with the repository and is materialized to
``<root>/synthetic_notes.csv`` on first use.

API Reference
-------------

.. autoclass:: pyhealth.datasets.SyntheticEHRNotesDataset
    :members:
    :undoc-members:
    :show-inheritance:
