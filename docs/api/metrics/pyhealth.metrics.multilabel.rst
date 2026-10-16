pyhealth.metrics.multilabel
===================================

.. currentmodule:: pyhealth.metrics.multilabel

.. autofunction:: multilabel_metrics_fn


Metric order
------------

``multilabel_metrics_fn`` walks the ``metrics`` list one entry at a time, and
no entry changes the arrays the others read, so the order of the list does not
change the numbers. ``metrics=["ddi", "jaccard_samples"]`` and
``metrics=["jaccard_samples", "ddi"]`` return the same values.

``"ddi"`` loads ``ddi_adj.npy`` from the PyHealth cache directory. GAMENet,
MICRON, MoleRec and SafeDrug write that file when the model is built, so
``"ddi"`` is available once one of those models has been created.
