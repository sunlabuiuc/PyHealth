pyhealth.metrics.clinical
===================================

Clinical summaries of a binary risk score, beyond AUROC:

- **Acting on the highest risks** (``alert_rate_metrics``): what flagging the
  top 5, 10 or 20% of samples by risk would catch (sensitivity), how many
  flags are true events (PPV), and the flags per event caught (number needed
  to evaluate).
- **Net benefit** (``net_benefit``): the decision curve of Vickers and Elkin
  (2006) at risk thresholds where one might act. Acting on no one scores 0.
- **Calibration in bins** (``calibration_bins``): mean predicted risk against
  the observed rate.
- **Subgroups** (``subgroup_metrics``): discrimination and observed /
  expected events within each group, and the largest gap between groups.

``clinical_summary`` computes all of them for one model on one test set.
Every function withholds results that could describe fewer than 11
patients, the HIPAA / CMS small-cell convention (``min_count=11`` by
default):

- an alert rate or net-benefit value whose true or false positives number
  fewer than 11;
- calibration bins until each has 11 events and 11 non-events;
- a group's (or the whole test set's) ``n`` and event count together, when
  it has fewer than 11 patients with or without the event, so one cannot be
  recovered from the other; and if only one subgroup of an attribute is
  withheld, the next smallest too, so its counts cannot be found by
  subtracting the others from the total.

Pass ``groups=patient_ids`` so cells count distinct patients rather than
samples. ``min_count=0`` turns suppression off, e.g. for
``alert_rate_metrics`` or ``net_benefit`` used as plain metrics on data
that is not published.

.. code-block:: python

    from pyhealth.metrics import clinical_summary

    summary = clinical_summary(y_true, y_prob, groups=patient_ids,
                               subgroups={"sex": sex, "age band": age_band})
    summary["alerts"]["top 10%"]    # {'sensitivity': 0.48, 'ppv': 0.34, ...}

.. currentmodule:: pyhealth.metrics.clinical

.. autofunction:: clinical_summary

.. autofunction:: alert_rate_metrics

.. autofunction:: net_benefit

.. autofunction:: calibration_bins

.. autofunction:: subgroup_metrics
