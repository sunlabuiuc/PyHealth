pyhealth.tasks.ChapmanShaoxingSuperclassClassification
======================================================

Multi-label ECG classification tasks for the Chapman-Shaoxing 12-lead ECG
dataset: the seven CLIMB superclasses (with RBBB in conduction disturbance)
and, optionally, the raw SNOMED-CT diagnosis codes.

.. autoclass:: pyhealth.tasks.ChapmanShaoxingSuperclassClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.ChapmanShaoxingSNOMEDClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.chapman_shaoxing_classification.map_dx_to_superclasses

.. autofunction:: pyhealth.tasks.chapman_shaoxing_classification.count_unmapped_codes

.. autofunction:: pyhealth.tasks.chapman_shaoxing_classification.load_chapman_record
