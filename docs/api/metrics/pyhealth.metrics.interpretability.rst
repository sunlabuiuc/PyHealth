pyhealth.metrics.interpretability
===================================

Interpretability metrics evaluate the faithfulness of feature attribution methods
by measuring how model predictions change when important features are removed or retained.

Evaluator
---------

.. currentmodule:: pyhealth.metrics.interpretability.evaluator

.. autoclass:: Evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Functional API
--------------

.. currentmodule:: pyhealth.metrics.interpretability.evaluator

.. autofunction:: evaluate_attribution

Removal-Based Metrics
---------------------

For binary classifiers, a sample filter can mark class-0 predictions as
``SampleClass.NEGATIVE``. Removal-based metrics then score probability changes
from the class-0 perspective. Each percentage is evaluated independently, so a
percentage's score does not depend on the other requested percentages.

Base Class
^^^^^^^^^^

.. currentmodule:: pyhealth.metrics.interpretability.base

.. autoclass:: RemovalBasedMetric
   :members:
   :undoc-members:
   :show-inheritance:

Comprehensiveness
^^^^^^^^^^^^^^^^^

.. currentmodule:: pyhealth.metrics.interpretability.comprehensiveness

.. autoclass:: ComprehensivenessMetric
   :members:
   :undoc-members:
   :show-inheritance:

Sufficiency
^^^^^^^^^^^

.. currentmodule:: pyhealth.metrics.interpretability.sufficiency

.. autoclass:: SufficiencyMetric
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. currentmodule:: pyhealth.metrics.interpretability.utils

.. autofunction:: get_model_predictions

.. autofunction:: create_validity_mask
