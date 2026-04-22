pyhealth.tasks.fairness\_utils
==============================

Framework-agnostic bootstrap audit utilities implementing the FAMEWS §3.1
methodology: for each (grouping, category), draw ``n_bootstrap`` resamples of
the test set, compute cohort vs complement metrics, run one-sided Mann-Whitney
U tests, and return cohort-level disparity deltas with Bonferroni-corrected
significance flags.

Works with any per-sample predictions dict, not just outputs from
:func:`~pyhealth.tasks.mortality_prediction_mimic3_with_fairness_fn` — pass
predictions + a list of sample dicts carrying cohort attributes.

.. automodule:: pyhealth.tasks.fairness_utils
    :members:
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from pyhealth.tasks.fairness_utils import audit_predictions

    # After training and generating probabilities
    audit = audit_predictions(
        samples=test_samples,       # each has `sex`, `age_group`, ... cohort keys
        probs=y_prob,               # shape (n_test,)
        labels=y_true,              # shape (n_test,)
        n_bootstrap=100,
        significance_level=0.001,
    )
    print(audit[audit["significantly_worse"]])

References
----------

Hoche et al. (2024). FAMEWS: A Fairness Auditing Tool for Medical
Early-Warning Systems. PMLR 248:297-311.
