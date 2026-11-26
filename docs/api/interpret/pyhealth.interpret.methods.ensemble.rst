pyhealth.interpret.methods.ensemble
===================================

Overview
--------

Ensemble interpreters combine the attributions of several expert interpreters
(at least three) into one consensus attribution. Each expert's attributions are
rank-normalized to a common [0, 1] scale, aggregated by a strategy-specific
rule, and reshaped back to the original tensor shapes. Choose a strategy:

- :class:`~pyhealth.interpret.methods.AvgEnsemble`: uniform average (AGGMean / Borda).
- :class:`~pyhealth.interpret.methods.VarEnsemble`: mean divided by standard
  deviation (AGGVar), which suppresses features the experts disagree on.
- :class:`~pyhealth.interpret.methods.CrhEnsemble`: iteratively reweights experts
  by their agreement with the consensus (CRH truth discovery).

To add a new strategy, subclass
:class:`~pyhealth.interpret.methods.base_ensemble.BaseInterpreterEnsemble` and
implement ``_ensemble``.

Writing a custom strategy
-------------------------

A new strategy only needs to override ``_ensemble``. It receives the
rank-normalized attributions of all experts as a tensor of shape
``(B, I, M)`` (batch, experts, flattened features) and returns a consensus
tensor of shape ``(B, M)``, both with values in [0, 1].

.. code-block:: python

    import torch
    from pyhealth.interpret.methods import IntegratedGradients, DeepLift, LimeExplainer
    from pyhealth.interpret.methods.base_ensemble import BaseInterpreterEnsemble

    class MaxEnsemble(BaseInterpreterEnsemble):
        """Keep, for each feature, the highest score among the experts."""

        def _ensemble(self, attributions: torch.Tensor) -> torch.Tensor:
            return attributions.max(dim=1).values

    # model is a trained PyHealth model that implements the Interpretable
    # interface (e.g. MLP), which all three experts below require, and
    # batch is a batch from a dataloader over its SampleDataset.
    experts = [IntegratedGradients(model), DeepLift(model), LimeExplainer(model)]
    ensemble = MaxEnsemble(model, experts)
    attrs = ensemble.attribute(**batch)

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.base_ensemble.BaseInterpreterEnsemble
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyhealth.interpret.methods.AvgEnsemble
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyhealth.interpret.methods.VarEnsemble
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyhealth.interpret.methods.CrhEnsemble
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
