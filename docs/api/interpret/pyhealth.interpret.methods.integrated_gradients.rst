pyhealth.interpret.methods.integrated_gradients
===============================================

Overview
--------

The Integrated Gradients method computes feature attributions for PyHealth models by integrating
gradients along a path from a baseline to the actual input. This helps identify which features 
(e.g., diagnosis codes, lab values) most influenced a model's prediction.

For a complete working example, see:
``examples/integrated_gradients_mortality_mimic4_stagenet.py``

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.IntegratedGradients
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
