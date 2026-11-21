pyhealth.tasks.cardiology_detect
=======================================

``CardiologyDetection`` is the task for the 2.0
:class:`~pyhealth.datasets.CardiologyDataset`. The ``cardiology_is*_fn``
functions are the PyHealth 1.x tasks; they take 1.x record dicts and are kept
for existing code.

.. autoclass:: pyhealth.tasks.CardiologyDetection
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: pyhealth.tasks.cardiology_detect.CARDIOLOGY_CATEGORIES

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
