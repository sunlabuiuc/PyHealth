pyhealth.tasks.cardiology_detect
=======================================

.. autofunction:: pyhealth.tasks.cardiology_detect.cardiology_isAR_fn
.. autofunction:: pyhealth.tasks.cardiology_detect.cardiology_isBBBFB_fn
.. autofunction:: pyhealth.tasks.cardiology_detect.cardiology_isAD_fn
.. autofunction:: pyhealth.tasks.cardiology_detect.cardiology_isCD_fn
.. autofunction:: pyhealth.tasks.cardiology_detect.cardiology_isWA_fn

Epoch-file handling
-------------------

The cardiology detection functions close each output epoch file before adding
its path to the returned samples. Output handles are also closed if serialization
raises an exception. A failed write can leave a partial file; it is not returned
as a completed sample.
