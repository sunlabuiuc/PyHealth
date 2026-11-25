pyhealth.metrics.significance
===================================

Statistical comparison of benchmark settings across repeated runs. This is
the protocol of Section 4.1 of Nonaka & Seita (2021), *In-depth Benchmarking
of Deep Neural Network Architectures for ECG Diagnosis*
(https://proceedings.mlr.press/v149/nonaka21a.html): train every setting once
per data split, report the mean and standard deviation of the per-split
scores, and compare each setting against the baseline with a two-sided
Welch's t-test at ``alpha = 0.05``, correcting the family of p-values with
Benjamini-Hochberg. Settings that do not improve on the baseline are
reported but excluded from testing, because the question is only whether a
setting is significantly *better* than the baseline.

Use this when the repeats are independent training runs (one per split),
which is what a benchmark sweep produces. For confidence intervals on a
single run's test set, use :mod:`pyhealth.metrics.bootstrap` instead.

.. code-block:: python

    from pyhealth.metrics import welch_relative_improvement_test

    scores = {
        # setting -> ResNet depth -> one macro ROC-AUC per trial split
        "all/roc_auc_macro": {18: [0.914, 0.912, 0.915], 50: [0.917, 0.916, 0.918]},
    }
    for row in welch_relative_improvement_test(scores, baseline=18):
        print(row["mean"], row["std"], row["relative_improvement"], row["q_value"])

The paper reports no significant improvement over ResNet-18 for any of the
architectures it tested. See ``examples/ptbxl_ecg_benchmark_resnet1d.py``
for a sweep that fills this in from real runs.

.. currentmodule:: pyhealth.metrics.significance

.. autofunction:: welch_relative_improvement_test

.. autofunction:: benjamini_hochberg
