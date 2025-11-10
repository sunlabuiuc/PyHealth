pyhealth.metrics.interpretability
===================================

Interpretability metrics evaluate the faithfulness of feature attribution methods
by measuring how model predictions change when important features are removed or retained.

Evaluator
---------

.. currentmodule:: pyhealth.metrics.interpretability

.. autoclass:: Evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Functional API
--------------

.. autofunction:: evaluate_attribution

Removal-Based Metrics
---------------------

Base Class
^^^^^^^^^^

.. autoclass:: RemovalBasedMetric
   :members:
   :undoc-members:
   :show-inheritance:

Comprehensiveness
^^^^^^^^^^^^^^^^^

.. autoclass:: ComprehensivenessMetric
   :members:
   :undoc-members:
   :show-inheritance:

Sufficiency
^^^^^^^^^^^

.. autoclass:: SufficiencyMetric
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. currentmodule:: pyhealth.metrics.interpretability

.. autofunction:: get_model_predictions

.. autofunction:: create_validity_mask
