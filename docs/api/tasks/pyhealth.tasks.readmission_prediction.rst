pyhealth.tasks.readmission_prediction
=======================================

.. autoclass:: pyhealth.tasks.readmission_prediction.ReadmissionPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.readmission_prediction.ReadmissionPredictionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

Minimum admission gap
---------------------

``ReadmissionPredictionMIMIC4`` supports an optional ``min_gap`` parameter.
An admission that starts at most ``min_gap`` after the previous discharge,
such as an internal transfer, is merged into the same stay. The stay gives one
sample, with the codes of all its admissions, and is labelled from its last
discharge.

For example:

.. code-block:: python

    from datetime import timedelta
    from pyhealth.tasks import ReadmissionPredictionMIMIC4

    task = ReadmissionPredictionMIMIC4(
        window=timedelta(days=30),
        min_gap=timedelta(hours=3),
    )

To keep existing results unchanged, ``min_gap`` defaults to ``None``, which
merges nothing, so some "readmissions" are in fact internal transfers.

.. autoclass:: pyhealth.tasks.readmission_prediction.ReadmissionPredictionEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.readmission_prediction.ReadmissionPredictionOMOP
    :members:
    :undoc-members:
    :show-inheritance:
