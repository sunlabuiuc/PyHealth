pyhealth.interpret.methods.baseline
===================================

Overview
--------

:class:`pyhealth.interpret.methods.RandomBaseline` assigns random importance
scores to input features. It works with any PyHealth model and needs no
gradients, and it is intended as a sanity-check baseline: a meaningful
attribution method should be clearly more faithful to the model than random
scores. See ``examples/interpretability/interpret_playground.py`` for usage
alongside other methods.

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.RandomBaseline
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
