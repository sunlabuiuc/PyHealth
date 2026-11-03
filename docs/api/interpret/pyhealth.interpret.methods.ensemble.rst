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
