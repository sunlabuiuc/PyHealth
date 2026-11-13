pyhealth.interpret.methods.lrp
===============================

Overview
--------

Layer-wise Relevance Propagation (LRP) explains a prediction by redistributing
the model's output score backwards through the network, layer by layer, until
every input feature carries a share of the relevance. Unlike gradient-based
saliency, which measures local sensitivity, LRP conserves a *quantity* of
relevance as it flows, so attributions are directly comparable across features
of the same sample.

The implementation follows Bach et al. (2015), "On Pixel-Wise Explanations for
Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation"
(https://doi.org/10.1371/journal.pone.0130140), with the local-renormalization
handling of Binder et al. (2016) (https://arxiv.org/abs/1604.00825).

This method is particularly useful for:

- **Multi-modal EHR models**: Attributing a prediction across discrete codes and continuous labs in one pass
- **Medical imaging**: Producing pixel-level relevance heatmaps for CNN classifiers
- **Model debugging**: Checking that relevance concentrates on clinically meaningful inputs
- **Method comparison**: Serving as a conservation-based counterpart to gradient and perturbation methods

Propagation Rules
-----------------

- ``rule="epsilon"`` — the :math:`\varepsilon`-rule stabilises the denominator
  with :math:`z_j + \varepsilon \cdot \mathrm{sign}(z_j)`. Larger ``epsilon``
  absorbs more relevance and yields sparser, less noisy attributions.
- ``rule="alphabeta"`` — the :math:`\alpha\beta`-rule separates positive and
  negative contributions and recombines them as
  :math:`\alpha R^{+} - \beta R^{-}`, with ``alpha - beta == 1`` by convention.
  The split is taken over the *contributions* :math:`z_{ij} = x_i w_{ij}`, not
  over the weights alone -- for :math:`x_i < 0` a positive weight yields a
  negative contribution, so both the input and the weight are split. With
  ``beta=0`` the negative path is disabled and relevance keeps the sign of the
  output relevance.

Layer Handlers
--------------

Per-layer propagation lives in :mod:`pyhealth.interpret.methods.lrp_base`, which
defines an abstract :class:`~pyhealth.interpret.methods.lrp_base.LRPLayerHandler`
and an :class:`~pyhealth.interpret.methods.lrp_base.LRPHandlerRegistry` that maps
a layer type to its handler. ``LayerwiseRelevancePropagation`` builds a registry
at construction and dispatches each layer through it, so support for a custom
layer can be added without subclassing the interpreter::

    lrp._registry.register(MyLayerHandler())

Handlers ship for ``Linear``, ``Conv2d``, ``ReLU``, ``MaxPool2d``, ``AvgPool2d``,
``AdaptiveAvgPool2d``, ``Flatten``, ``BatchNorm2d``, ``Dropout`` and the
recurrent ``LSTM``/``GRU`` layers. Layer types with no registered handler pass
relevance through unchanged.

Usage Notes
-----------

1. **Batch size**: Use ``batch_size=1`` for per-sample explanations.
2. **Evaluation mode**: Call ``model.eval()`` first so dropout and batch-norm
   behave deterministically.
3. **Embeddings**: With ``use_embeddings=True`` the model must implement the
   :class:`~pyhealth.interpret.api.Interpretable` interface (or expose both
   ``forward_from_embedding`` and ``get_embedding_model``), and relevance is
   propagated from the embedding output rather than the raw token indices.
4. **Conservation is approximate**: Per-layer conservation is exact only for a
   bias-free linear layer under the :math:`\varepsilon`-rule. Biases, the
   stabiliser itself, and pooling all absorb relevance, so
   ``sum(attributions)`` tracks the logit rather than matching it exactly. The
   gap is largest for models whose logit is small relative to their biases.
5. **Recurrent models**: ``StageNet`` applies the same ``nn.Linear`` repeatedly
   inside a Python loop, so a forward hook retains only the final time step.
   Relevance for recurrent layers is therefore distributed uniformly across
   time steps, and per-time-step attributions from these models should be read
   as a feature-level summary rather than a precise temporal attribution.

Quick Start
-----------

.. code-block:: python

    from pyhealth.datasets import get_dataloader
    from pyhealth.interpret.methods import LayerwiseRelevancePropagation
    from pyhealth.models import StageNet

    # Assume you have a trained model and dataset
    model = StageNet(dataset=sample_dataset)
    # ... train the model ...
    model.eval()

    # Per-sample explanations
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    batch = next(iter(test_loader))

    lrp = LayerwiseRelevancePropagation(
        model, rule="epsilon", epsilon=0.01, use_embeddings=True
    )
    attributions = lrp.attribute(**batch)

    for feature_key, relevance in attributions.items():
        print(f"{feature_key}: {relevance.shape}, total={relevance.sum():.4f}")

For complete working examples, see:

- ``examples/interpretability/lrp_stagenet_synthetic.py`` — self-contained run on synthetic data, no external dataset required
- ``examples/interpretability/lrp_stagenet_mimic4.py`` — mortality prediction on MIMIC-IV
- ``examples/ChestXray-Classification-ResNet-with-Saliency.ipynb`` — LRP heatmaps on chest X-rays, compared against gradient saliency

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.LayerwiseRelevancePropagation
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. automodule:: pyhealth.interpret.methods.lrp_base
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
