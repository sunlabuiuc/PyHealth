pyhealth.interpret.methods.tree_shap
====================================

Exact TreeSHAP attributions for gradient-boosted tree models such as
:class:`~pyhealth.models.XGBoostModel`, computed by the tree library itself
(no sampling). ``attribute`` returns one tensor per input field in the field's
shape. The column-level values, including codes that are absent from a sample,
come from ``model.explain(**batch)``; ``model.mean_abs_shap(loader)`` gives the
global ranking.

.. autoclass:: pyhealth.interpret.methods.TreeSHAP
    :members:
    :show-inheritance:
