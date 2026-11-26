pyhealth.metrics.event_alarm
============================

Event-level metrics of an alarm policy for early event prediction (Hyland et
al., 2020; Section 4.1 of Yeche et al., 2024): a threshold on the risk score
with silencing, alarm precision, event recall at a given alarm precision, the
distance of the first alarm to the event, and the area under the alarm
precision / event recall curve. They complement the time-step ROC-AUC and
PR-AUC, which do not tell how early or how often a system raises alarms.

.. code-block:: python

    from pyhealth.metrics import event_alarm_metrics

    scores = event_alarm_metrics(
        scores=risk_per_stay,          # list of [steps] arrays
        times=hours_per_stay,
        time_to_event=tte_per_stay,    # from DecompensationMIMIC3(label_mode="survival")
        event=event_per_stay,
        horizon=24,
        silence=1.0,
    )
    # {'alarm_event_auprc': ..., 'event_recall@60': ..., 'first_alarm_distance@60': ...}

.. currentmodule:: pyhealth.metrics.event_alarm

.. autofunction:: alarm_policy

.. autofunction:: alarm_curve

.. autofunction:: alarm_event_auprc

.. autofunction:: event_alarm_metrics
