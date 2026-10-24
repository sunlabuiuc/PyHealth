pyhealth.tasks.sleep_staging
=======================================

.. autofunction:: pyhealth.tasks.sleep_staging.sleep_staging_isruc_fn
.. autofunction:: pyhealth.tasks.sleep_staging.sleep_staging_sleepedf_fn
.. autofunction:: pyhealth.tasks.sleep_staging.sleep_staging_shhs_fn

.. note::
   ``sleep_staging_shhs_fn`` only emits epochs that have a corresponding
   annotation: when a recording contains more complete signal epochs than
   parsed ``SleepStage`` labels, the trailing unlabeled epochs are skipped.
