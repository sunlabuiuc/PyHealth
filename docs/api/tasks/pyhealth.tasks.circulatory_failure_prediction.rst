pyhealth.tasks.circulatory_failure_prediction
=============================================

Overview
--------

CirculatoryFailurePredictionTask defines a time-series prediction task for early
detection of circulatory failure.

The task predicts whether a patient will experience circulatory failure within
the next 12 hours based on physiological measurements.

Label definition:

- label = 1 if circulatory failure occurs within the next 12 hours
- label = 0 otherwise

API Reference
-------------

.. autoclass:: pyhealth.tasks.CirculatoryFailurePredictionTask
   :members:
   :undoc-members:
   :show-inheritance: