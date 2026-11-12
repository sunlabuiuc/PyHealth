Competing-risk preprocessing
============================

.. automodule:: pyhealth.tasks.competing_risks
    :members:

Fit on inner-training patients only. This helper imputes first and then
standardizes the imputed matrix.
That ordering differs from ``TensorProcessor(normalization="zscore")``.
Persist ``state`` with the checkpoint and restore with ``from_state``.
