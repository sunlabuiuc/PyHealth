Competing-risk preprocessing
============================

.. automodule:: pyhealth.tasks.competing_risks
    :members:

Fit on inner-training patients only. This helper imputes first and then
standardizes the imputed matrix.
``TensorProcessor(normalization="zscore")`` and
``pyhealth.processors.utils.fit_normalization`` fit scales on observed values,
ignoring NaNs. They do not compute the standard deviation of the mean-imputed
matrix when given the original incomplete samples. That difference changes
feature scales when values are missing. This split-scoped helper keeps the
impute-then-standardize order, native observation masks, and time scaling
together without changing shared processor behavior.
Persist ``state`` with the checkpoint and restore with ``from_state``.
