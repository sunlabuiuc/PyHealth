pyhealth.interpret.methods.raw_attention
========================================

Overview
--------

Raw Attention uses the attention weights of a single Transformer layer, by
default the last one, averaged over heads, as each input token's relevance
score. It is the simplest attention-based explanation and the baseline used by
Chefer et al. (2021), "Transformer Interpretability Beyond Attention
Visualization" (https://arxiv.org/abs/2012.09838).

Raw attention ignores how information mixed in the layers below the one it
reads. :class:`~pyhealth.interpret.methods.AttentionRollout` composes every
layer, and :class:`~pyhealth.interpret.methods.CheferRelevance` also weights
attention by gradients for a target class. Running all three on the same model
shows how much each of those additions changes the explanation.

Key Features
------------

- **Gradient-free and class-agnostic**: one forward pass, no backward pass; ``target_class_idx`` is accepted but ignored
- **Layer choice**: ``layer`` picks the attention layer as a Python list index (``-1``, the last layer, by default)
- **Head fusion**: ``"mean"`` (default, as in the paper) or ``"max"``
- **Distribution over tokens**: with mean fusion, per-token relevance sums to 1 (before the input-shape expansion)
- **Model-agnostic by duck-typing**: works with any model exposing ``set_attention_hooks``, ``get_attention_layers`` and ``get_relevance_tensor`` (currently :class:`~pyhealth.models.Transformer` and :class:`~pyhealth.models.StageAttentionNet`)

Usage Notes
-----------

1. **Do not wrap in** ``torch.no_grad()``: the shared attention-readout plumbing registers a gradient hook during the forward pass, so calling ``attribute`` inside ``torch.no_grad()`` raises a ``RuntimeError``. No backward pass is performed.
2. **Layer range**: ``StageAttentionNet`` has a single attention layer, so only ``layer=0`` or ``-1`` is valid there. An out-of-range ``layer`` raises ``ValueError``.
3. **Comparison**: the ``examples/interpretability/*_mimic4_interpret.py`` scripts accept ``--methods raw_attention,rollout,chefer`` to score the three side by side with comprehensiveness and sufficiency.

Quick Start
-----------

.. code-block:: python

    from pyhealth.interpret.methods import RawAttention
    from pyhealth.datasets import get_dataloader

    # Assume a trained Transformer `model` and a `test_dataset`
    raw = RawAttention(model)  # last layer, heads averaged

    batch = next(iter(get_dataloader(test_dataset, batch_size=1, shuffle=False)))
    scores = raw.attribute(**batch)

    for feature_key, attribution in scores.items():
        print(feature_key, attribution[0].topk(5).indices)

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.RawAttention
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
