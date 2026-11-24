pyhealth.tasks.DSAActivityClassification
========================================

Activity classification on the DSA dataset (19 classes). Each sample holds the
target sensor stream (``signal``) and, for transfer learning, one synchronized
``source_<name>`` stream per source sensor. Optionally rescales every channel to
[-1, 1], as in Zhang et al. (2024).

.. autoclass:: pyhealth.tasks.dsa_activity_classification.DSAActivityClassification
    :members:
    :undoc-members:
    :show-inheritance:
