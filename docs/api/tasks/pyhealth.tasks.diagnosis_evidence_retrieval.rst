
pyhealth.tasks.diagnosis_evidence_retrieval
==========================================

Usage
-----

This task prepares diagnosis-specific historical clinical notes
for downstream evidence retrieval models. It requires reviewed
diagnosis queries associated with index radiology reports.

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import DiagnosisEvidenceRetrievalMIMIC3

    dataset = MIMIC3Dataset(
        root="/path/to/mimiciii",
        tables=["noteevents"],
    )

    task = DiagnosisEvidenceRetrievalMIMIC3(
        reviewed_queries=[
            {
                "patient_id": "1",
                "index_note_id": "105",
                "diagnosis": "stroke",
            }
        ],
        min_prior_notes=8,
        max_notes=40,
        strict_availability=True,
        deduplication="exact",
    )

    samples = dataset.set_task(task, num_workers=1)

The patient and report identifiers above are illustrative and
must be replaced with valid reviewed records.

The task returns historical notes, source note identifiers,
diagnosis queries, temporal metadata, and deduplication audit
information. It does not generate ground-truth relevance labels.

To retain historical notes without admission identifiers,
use a dataset configuration with a left join for note events.

Rebuild any outdated MIMIC-III dataset caches that do not
contain the required note metadata and chart-time provenance.

Runnable Example
----------------

A runnable synthetic example with an evaluated deduplication
ablation is available at:

``examples/mimic3_diagnosis_evidence_retrieval_transformers.py``.

Reference
---------

`Retrieving Evidence from EHRs with LLMs: Possibilities and Challenges
<https://arxiv.org/abs/2309.04550>`_

API Reference
-------------

.. autoclass:: pyhealth.tasks.diagnosis_evidence_retrieval.DiagnosisEvidenceRetrievalMIMIC3
    :members: task_name, input_schema, output_schema
    :special-members: __call__
    :show-inheritance:
