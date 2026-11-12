pyhealth.models.SHy
===================================

SHy, the self-explaining hypergraph neural network for next-visit diagnosis
prediction (Yu et al., "Self-Explaining Hypergraph Neural Networks for
Diagnosis Prediction", CHIL 2025, https://arxiv.org/abs/2502.10689). Each
patient is a hypergraph of diagnosis codes (nodes) and visits (hyperedges);
the model recovers likely-missing diagnoses and predicts from K temporal
phenotypes, which together with their weights are the explanation
(:meth:`SHy.explain`). Pair it with
:class:`~pyhealth.tasks.NextVisitDiagnosisPredictionMIMIC3` and the
``recall@k`` / ``ndcg@k`` metrics of :func:`~pyhealth.metrics.multilabel_metrics_fn`.

.. autoclass:: pyhealth.models.SHy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.shy.HierarchicalCodeEmbedding
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.shy.UniGINConv
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.shy.PhenotypeExtractor
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.shy.PhenotypeEncoder
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.shy.FidelityDecoder
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.shy.Incidences
    :members:

.. autofunction:: pyhealth.models.shy.build_code_levels

.. autofunction:: pyhealth.models.shy.icd9_ancestor_fn
