pyhealth.interpret.methods.ig_gim
=================================

Overview
--------

:class:`pyhealth.interpret.methods.IntegratedGradientGIM` runs the standard
Integrated Gradients path integral with the Gradient Interaction Modifications
(GIM) backward rules applied at every step. This removes gradient artefacts
caused by softmax redistribution and LayerNorm re-centring. The model must
implement ``forward_from_embedding()`` and ``get_embedding_model()``. See also
:doc:`pyhealth.interpret.methods.gim` and
:doc:`pyhealth.interpret.methods.integrated_gradients`.

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.IntegratedGradientGIM
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
