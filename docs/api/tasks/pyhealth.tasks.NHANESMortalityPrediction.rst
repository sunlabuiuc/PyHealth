pyhealth.tasks.NHANESMortalityPrediction
=======================================

Predicts mortality within a configurable follow-up horizon using
wearable physical activity from the NHANES dataset.

The default prediction horizon is 60 months. Participants with
insufficient mortality follow-up are excluded when their outcome
cannot be determined.

Activity is divided into non-overlapping windows of 1,008 five-minute
epochs. Windows with insufficient valid wear coverage are excluded.

Supported Dataset
-----------------

- :doc:`../datasets/pyhealth.datasets.NHANESDataset`

Feature Representations
-----------------------

The task supports two feature formats:

- ``activity``: a combined activity tensor for general sequence models.
- ``motionage``: separate intensity, hour, relative day, and wear-mask
  tensors for the MotionAge model.

The MotionAge format supports two wear representations:

- ``binary``: valid epochs receive weight one and invalid epochs zero.
- ``fraction``: valid epochs receive their continuous wear fraction,
  while invalid epochs receive zero.

Optional static covariates include age, gender, body mass index,
systolic blood pressure, and diastolic blood pressure.


Temporal Features
-----------------

Hour and day indices are derived from the sequential activity
epochs. They represent positions within the monitoring period
rather than independently verified clock times and calendar
days of the week.

Wear weights affect temporal pooling, but do not prevent
non-wear epochs from influencing recurrent hidden states.

Example
-------

.. code-block:: python

    from pyhealth.tasks import NHANESMortalityPrediction

    task = NHANESMortalityPrediction(
        root="/path/to/nhanes",
        feature_format="motionage",
        wear_representation="fraction",
        include_static=False,
        pos_weight="balanced",
    )

When using balanced class weights or static feature normalization,
fit processors on the training split only.

.. autoclass:: pyhealth.tasks.NHANESMortalityPrediction
    :members:
    :undoc-members:
    :show-inheritance:
