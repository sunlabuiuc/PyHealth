pyhealth.tasks.ed_disposition_mimic4
====================================

Emergency department tasks on MIMIC-IV-ED from MEME (Lee et al., npj Digital
Medicine 2025): each ED stay becomes one templated pseudo-note per domain
(arrival, triage, medication reconciliation, vital signs, Pyxis and,
optionally, ED diagnosis codes). :class:`~pyhealth.tasks.EDDispositionMIMIC4`
predicts admission versus discharge home;
:class:`~pyhealth.tasks.EDOutcomesMIMIC4` predicts the hospital outcomes of
admitted patients. Pairs with :class:`~pyhealth.models.MEME`.

ED diagnosis codes are assigned at the end of the stay and can leak the
disposition, so they are excluded unless ``include_codes=True``.

.. autoclass:: pyhealth.tasks.EDDispositionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.EDOutcomesMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ed_disposition_mimic4.arrival_note
.. autofunction:: pyhealth.tasks.ed_disposition_mimic4.triage_note
.. autofunction:: pyhealth.tasks.ed_disposition_mimic4.medrecon_note
.. autofunction:: pyhealth.tasks.ed_disposition_mimic4.vitals_note
.. autofunction:: pyhealth.tasks.ed_disposition_mimic4.pyxis_note
.. autofunction:: pyhealth.tasks.ed_disposition_mimic4.codes_note
