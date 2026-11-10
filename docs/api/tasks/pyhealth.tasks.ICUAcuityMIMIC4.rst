pyhealth.tasks.ICUAcuityMIMIC4
==============================

Real-time ICU acuity on MIMIC-IV, the setting of APRICOT-M (Contreras et al.,
Nature Communications 2025): every 4 hours, the events of the past 4 hours
predict the next 4 hours' acuity state (discharge, stable, unstable,
deceased), stable/unstable transitions and the stop or onset of mechanical
ventilation, vasopressors and CRRT. Pairs with
:class:`~pyhealth.models.APRICOTM`.

.. autoclass:: pyhealth.tasks.ICUAcuityMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:
