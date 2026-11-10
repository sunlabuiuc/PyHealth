pyhealth.datasets.Support2Dataset
==================================

Overview
--------

The SUPPORT2 (Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments) dataset contains data on seriously ill hospitalized adults. It includes patient demographics, diagnoses, clinical measurements, and outcomes such as survival and hospital mortality.

The dataset is commonly used for mortality prediction, length of stay prediction, and other clinical outcome prediction tasks.

.. autoclass:: pyhealth.datasets.Support2Dataset
    :members:
    :undoc-members:
    :show-inheritance:

The complete-case time-to-event task is
:class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`. It is separate from the
existing fixed-horizon probability regression task. The released-split helpers
map local reference arrays to source patient numbers without downloading data.

.. autofunction:: pyhealth.datasets.support2_paper_time_transform
.. autofunction:: pyhealth.datasets.support2_released_split_ids

