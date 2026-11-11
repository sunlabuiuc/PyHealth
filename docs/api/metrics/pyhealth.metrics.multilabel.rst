pyhealth.metrics.multilabel
===================================

.. currentmodule:: pyhealth.metrics.multilabel

.. autofunction:: multilabel_metrics_fn

The ``precision@K``, ``recall@K`` and ``ndcg@K`` metrics are computed by
:func:`pyhealth.metrics.ranking.multilabel_ranking_at_k`.


Metric order
------------

``multilabel_metrics_fn`` walks the ``metrics`` list one entry at a time, and
no entry changes the arrays the others read, so the order of the list does not
change the numbers. ``metrics=["ddi", "jaccard_samples"]`` and
``metrics=["jaccard_samples", "ddi"]`` return the same values.

DDI matrix
----------

``"ddi"`` needs the drug-drug interaction matrix of the label vocabulary. Pass
it as ``ddi_adj``. ARMR, GAMENet, MICRON, MoleRec and SafeDrug store theirs in
``model.ddi_adj_matrix``, and ``Trainer.evaluate`` passes that matrix for you
when the evaluated model has one:

.. code-block:: python

    scores = multilabel_metrics_fn(
        y_true, y_prob, metrics=["ddi"], ddi_adj=model.ddi_adj_matrix
    )

Without ``ddi_adj``, ``"ddi"`` loads ``ddi_adj.npy`` from the PyHealth cache
directory, and raises ``FileNotFoundError`` if it is missing. The five models
write that file when they are built (atomically, so a reader never sees a
partial file), but it is shared: it holds the matrix of whichever model was
built last, in any process. If another run or a model with a different drug
vocabulary was built in between, the file belongs to that model, so pass
``ddi_adj`` whenever you call the function yourself.
