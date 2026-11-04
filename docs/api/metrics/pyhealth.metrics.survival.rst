pyhealth.metrics.survival
=========================

Metrics for right-censored survival predictions: Harrell's concordance index,
the IPCW Brier score (Graf et al., 1999) and the cumulative/dynamic AUC
(Uno et al., 2007). Censoring weights come from the Kaplan-Meier estimate of
the censoring distribution on the evaluated data.

.. code-block:: python

    from pyhealth.metrics import survival_metrics_fn

    scores = survival_metrics_fn(
        duration, event, risk, survival=surv_curves, time_grid=grid, times=[7, 30]
    )
    # {'c_index': ..., 'auc@7': ..., 'brier@7': ..., 'auc@30': ..., 'brier@30': ...}

.. currentmodule:: pyhealth.metrics.survival

.. autofunction:: concordance_index

.. autofunction:: ipcw_brier_score

.. autofunction:: cumulative_dynamic_auc

.. autofunction:: survival_metrics_fn
