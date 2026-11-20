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

Time-dependent concordance (longitudinal predictions)
------------------------------------------------------

For models that predict at repeated visits (landmark or counting-process
samples), the time-dependent concordance index C(t, delta_t) scores each
patient's risk from their latest visit at or before ``t`` against events in
``(t, t + delta_t]``, as in the MAS paper (Gao et al., MLHC 2024).

.. code-block:: python

    from pyhealth.metrics import (
        mean_tdci_bootstrap_ci, outcomes_from_labels, time_dependent_concordance_index,
    )

    event_time, event = outcomes_from_labels(y_true, landmark_time, patient_id)
    out = time_dependent_concordance_index(event_time, event, risk, landmark_time, patient_id)
    # {'mean_tdci': ..., 'tdci': [4, 4] grid, 'n_at_risk': ..., 'n_events': ...}
    ci = mean_tdci_bootstrap_ci(event_time, event, risk, landmark_time, patient_id)

.. autofunction:: time_dependent_concordance_index

.. autofunction:: outcomes_from_labels

.. autofunction:: mean_tdci_bootstrap_ci

.. autofunction:: mean_tdci_paired_bootstrap_diff
