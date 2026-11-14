pyhealth.metrics.multiclass
===================================

.. currentmodule:: pyhealth.metrics.multiclass

.. autofunction:: multiclass_metrics_fn

Ranking metrics
---------------

``multiclass_metrics_fn`` also reports how highly the true class is ranked among the predicted probabilities:

- ``"hits@K"`` (for any positive integer ``K``, e.g. ``"hits@3"``) returns
    ``HITS@K``, the fraction of samples whose true class is in the top ``K``.
- ``"hits@n"`` returns ``HITS@1``, ``HITS@5`` and ``HITS@10`` together.
- ``"mean_rank"`` returns ``mean_rank`` and ``mean_reciprocal_rank``.

.. code-block:: python

    from pyhealth.metrics import multiclass_metrics_fn

    multiclass_metrics_fn(y_true, y_prob, metrics=["hits@3", "mean_rank"])
