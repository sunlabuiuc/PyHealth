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

``ReadmissionPredictionMIMIC4`` supports an optional ``min_gap`` parameter to
exclude very short intervals between discharge and the next admission from
being labeled as readmissions. This can be useful when short gaps may reflect
internal transfers rather than a new hospital admission.

For example:

.. code-block:: python

    from datetime import timedelta
    from pyhealth.tasks import ReadmissionPredictionMIMIC4

    task = ReadmissionPredictionMIMIC4(
        window=timedelta(days=30),
        min_gap=timedelta(hours=3),
    )

By default, ``min_gap`` is ``None``, which preserves the existing behavior.

.. autoclass:: pyhealth.tasks.readmission_prediction.ReadmissionPredictionEICU
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.readmission_prediction.ReadmissionPredictionOMOP
    :members:
    :undoc-members:
    :show-inheritance:
