pyhealth.tasks.PriorExamPreferenceTask
=======================================

.. autoclass:: pyhealth.tasks.PriorExamPreferenceTask
    :members:
    :undoc-members:
    :show-inheritance:

Data requirements
-----------------

This task is applied to a :class:`~pyhealth.datasets.MIMIC4CXRDataset` loaded
with its ``metadata`` table.  The caller must separately provide local
ReXPref-Prior CSV files.  Both sources are credentialed and the task does not
download either one.

Each annotation CSV must contain ``study_id``, ``section``, ``sentence_id``,
``orig_sentence`` and ``new_sentence``.  A blank ``new_sentence`` represents a
deletion.  The mapping key records the published split and must be ``train``,
``val`` or ``test``.

The model input is ``text`` and the binary output is ``needs_edit``.  A value
of one means that ReXPref-Prior rewrote or deleted the sentence; it is not an
independent hallucination label.  The ``context_mode`` option changes only the
model text while preserving labels and examples:

* ``sentence_only`` includes the target sentence.
* ``neighbor_context`` also includes adjacent sentences from its section.
* ``section_context`` includes its full findings or impression section.

Example
-------

.. code-block:: python

    from pyhealth.datasets import MIMIC4CXRDataset
    from pyhealth.tasks import PriorExamPreferenceTask

    dataset = MIMIC4CXRDataset(
        root="/path/to/mimic-cxr-jpg/2.1.0",
        tables=["metadata"],
    )
    task = PriorExamPreferenceTask(
        annotation_files={
            "train": "/path/to/train_rexprefprior.csv",
            "val": "/path/to/val_rexprefprior.csv",
            "test": "/path/to/test_rexprefprior.csv",
        },
        context_mode="sentence_only",
    )
    samples = dataset.set_task(task)
