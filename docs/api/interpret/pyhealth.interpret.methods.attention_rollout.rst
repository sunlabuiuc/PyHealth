pyhealth.interpret.methods.attention_rollout
=============================================

Overview
--------

Attention Rollout provides token-level relevance scores for Transformer models
in PyHealth. It quantifies how attention propagates information across layers by
composing the per-layer attention matrices (with a residual-connection
correction), yielding a single importance score per input token (e.g. diagnosis
codes, procedure codes, medications) for a given patient sample.

Unlike :class:`~pyhealth.interpret.methods.CheferRelevance`, which is
gradient-weighted and **class-specific**, attention rollout is **forward-pass
only**, **gradient-free**, and **class-agnostic**: it explains how information
flows through the attention mechanism independent of any target class. It serves
as the standard baseline that gradient-based attention methods are compared
against, and complements Chefer rather than replacing it.

This method is particularly useful for:

- **Clinical decision support**: Understanding which medical codes drove a particular prediction
- **Model debugging**: Identifying whether the model attends to clinically meaningful features
- **Feature importance**: Ranking tokens by how much attention flows to them
- **Trust and transparency**: Providing interpretable, class-agnostic explanations for model predictions

The implementation follows the paper by Abnar & Zuidema (2020): "Quantifying
Attention Flow in Transformers" (https://arxiv.org/abs/2005.00928).

Key Features
------------

- **Multi-modal support**: Works with multiple feature types (conditions, procedures, drugs, labs, etc.)
- **Gradient-free**: Computed from a single forward pass; no backward pass is used in the attribution math
- **Class-agnostic**: Independent of the predicted/target class (``target_class_idx`` is accepted but ignored)
- **Layer-wise composition**: Composes per-layer attention as ``rollout = Â_L @ ... @ Â_1`` with the residual correction ``Â = 0.5 * (A + I)``
- **Distribution over tokens**: Because each ``Â`` is row-stochastic, so is their product; per-token relevance sums to 1 (before the input-shape expansion)
- **Model-agnostic by duck-typing**: Works with any model exposing the attention-readout methods ``set_attention_hooks``, ``get_attention_layers`` and ``get_relevance_tensor`` (currently :class:`~pyhealth.models.Transformer` and :class:`~pyhealth.models.StageAttentionNet`), not just one named model

Usage Notes
-----------

1. **Batch size**: For interpretability, use ``batch_size=1`` to get per-sample explanations.
2. **Do not wrap in** ``torch.no_grad()``: Although rollout is gradient-free in its math, the shared attention-readout plumbing registers a gradient hook on the attention tensors during the forward pass, so calling ``attribute(**batch)`` inside ``torch.no_grad()`` raises a ``RuntimeError``. Call it under the default (grad-enabled) context; no backward pass is performed.
3. **Model compatibility**: Works with any model that exposes ``set_attention_hooks``, ``get_attention_layers`` and ``get_relevance_tensor`` — not restricted to the Transformer. Incompatible models raise ``TypeError`` at construction.
4. **Class specification**: ``target_class_idx`` is accepted for API compatibility but ignored, since rollout is class-agnostic.

Quick Start
-----------

.. code-block:: python

    from pyhealth.models import Transformer
    from pyhealth.interpret.methods import AttentionRollout
    from pyhealth.datasets import get_dataloader

    # Assume you have a trained transformer model and dataset
    model = Transformer(dataset=sample_dataset, ...)
    # ... train the model ...

    # Create interpretability object
    rollout = AttentionRollout(model)

    # Get a test sample (batch_size=1)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    batch = next(iter(test_loader))

    # Compute attributions (target_class_idx is accepted but ignored)
    scores = rollout.attribute(**batch)

    # Analyze results
    for feature_key, attribution in scores.items():
        print(f"{feature_key}: {attribution.shape}")
        top_tokens = attribution[0].topk(5).indices
        print(f"  Top 5 most relevant tokens: {top_tokens}")

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.AttentionRollout
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
