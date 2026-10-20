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

Integer inputs, such as the code ids of ``sequence`` and ``nested_sequence``
fields, are always ablated by setting the removed codes to the padding id 0,
whatever the ``ablation_strategy``; averaging or adding noise to code ids has
no meaning. A sample whose codes would all be removed keeps its first code, so
every sequence has at least one non-padding position. Float inputs use the
chosen strategy (``"zero"``, ``"mean"`` or ``"noise"``). See
``examples/interpretability/comprehensiveness_code_sequences.py``.

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
