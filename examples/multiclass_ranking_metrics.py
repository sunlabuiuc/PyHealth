"""Ranking metrics with ``multiclass_metrics_fn``.

Shows how to compute HITS@K for any cutoff K, the HITS@1/5/10 shorthand
``"hits@n"``, and mean (reciprocal) rank on a small synthetic example.

Run:
    python examples/multiclass_ranking_metrics.py
"""

import numpy as np

from pyhealth.metrics import multiclass_metrics_fn

rng = np.random.default_rng(0)
n_samples, n_classes = 200, 30

y_true = rng.integers(0, n_classes, size=n_samples)
logits = rng.normal(size=(n_samples, n_classes))
logits[np.arange(n_samples), y_true] += 2.0  # make the true class more likely
y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

print(multiclass_metrics_fn(y_true, y_prob, metrics=["hits@3", "hits@20"]))
print(multiclass_metrics_fn(y_true, y_prob, metrics=["hits@n"]))
print(multiclass_metrics_fn(y_true, y_prob, metrics=["mean_rank"]))
