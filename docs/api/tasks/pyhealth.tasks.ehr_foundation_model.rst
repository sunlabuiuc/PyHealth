pyhealth.tasks.ehr_foundation_model
===================================

Wraps any task so each sample carries its raw, time-ordered event timeline
(``(timestamp, code, value, unit)`` tuples) and the patient's birth time,
both with the ``"raw"`` processor. Pretrained EHR foundation models such as
:class:`~pyhealth.models.HFEHRFoundationModel` map these to their own
vocabulary in ``forward()``, so no PyHealth vocabulary is built.

.. autoclass:: pyhealth.tasks.ehr_foundation_model.EHRFoundationModelTask
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.tasks.ehr_foundation_model.EventSource

.. autofunction:: pyhealth.tasks.ehr_foundation_model.clean_value

.. autodata:: pyhealth.tasks.ehr_foundation_model.BIRTH_CODES
