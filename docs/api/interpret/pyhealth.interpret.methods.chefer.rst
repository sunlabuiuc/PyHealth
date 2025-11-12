pyhealth.interpret.methods.chefer
===================================

Overview
--------

The Chefer interpretability method provides token-level relevance scores for Transformer models
in PyHealth. This approach is based on attention-based gradient propagation, which helps identify
which input tokens (e.g., diagnosis codes, procedure codes, medications) most influenced the 
model's prediction for a given patient sample.

This method is particularly useful for:

- **Clinical decision support**: Understanding which medical codes drove a particular prediction
- **Model debugging**: Identifying if the model is focusing on clinically meaningful features
- **Feature importance**: Ranking tokens by their contribution to the prediction
- **Trust and transparency**: Providing interpretable explanations for model predictions

The implementation follows the paper by Chefer et al. (2021): "Generic Attention-model 
Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers" 
(https://arxiv.org/abs/2103.15679).

Key Features
------------

- **Multi-modal support**: Works with multiple feature types (conditions, procedures, drugs, labs, etc.)
- **Gradient-based**: Uses attention gradients to compute relevance scores
- **Layer-wise propagation**: Aggregates relevance across transformer layers
- **Non-negative scores**: Returns clamped scores where higher values indicate greater relevance

Usage Notes
-----------

1. **Batch size**: For interpretability, use batch_size=1 to get per-sample explanations
2. **Gradients required**: Do not use within ``torch.no_grad()`` context
3. **Model compatibility**: Only works with PyHealth's Transformer model
4. **Class specification**: You can specify a target class or use the predicted class

Quick Start
-----------

.. code-block:: python

    from pyhealth.models import Transformer
    from pyhealth.interpret.methods import CheferRelevance
    from pyhealth.datasets import get_dataloader

    # Assume you have a trained transformer model and dataset
    model = Transformer(dataset=sample_dataset, ...)
    # ... train the model ...

    # Create interpretability object
    relevance = CheferRelevance(model)

    # Get a test sample (batch_size=1)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    batch = next(iter(test_loader))

    # Compute relevance scores
    scores = relevance.get_relevance_matrix(**batch)

    # Analyze results
    for feature_key, relevance_tensor in scores.items():
        print(f"{feature_key}: {relevance_tensor.shape}")
        top_tokens = relevance_tensor[0].topk(5).indices
        print(f"  Top 5 most relevant tokens: {top_tokens}")

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.CheferRelevance
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Helper Functions
----------------

The module also includes internal helper functions for relevance computation:

.. autofunction:: pyhealth.interpret.methods.chefer.apply_self_attention_rules

.. autofunction:: pyhealth.interpret.methods.chefer.avg_heads