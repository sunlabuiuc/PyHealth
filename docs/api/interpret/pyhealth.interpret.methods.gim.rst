pyhealth.interpret.methods.gim
================================

Overview
--------

The Gradient Interaction Modifications (GIM) interpreter adapts the StageNet
attribution method described by Edin et al. (2025). It recomputes softmax
gradients with a higher temperature so that token-level interactions remain
visible when cumulative softmax layers are present.

Use this interpreter with StageNet-style models that expose
``forward_from_embedding`` and ``embedding_model``.

For a complete working example, see:
``examples/gim_stagenet_mimic4.py``

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.GIM
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
