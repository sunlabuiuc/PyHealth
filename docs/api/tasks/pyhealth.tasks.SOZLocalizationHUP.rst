pyhealth.tasks.SOZLocalizationHUP
=================================

Seizure onset zone localization on the HUP iEEG dataset.

For each interictal recording the task keeps the channels marked good in
``channels.tsv``, re-references, filters and resamples them, and cuts every
channel into windows. Each window of one contact is a sample, labelled 1
when the contact is in the clinically determined seizure onset zone (or in
the resected tissue, with ``target="resect"``) and 0 otherwise.

All the windows of a contact carry the same label, and a contact belongs to
one patient, so windows of one patient must stay on one side of a split: use
``split_by_patient`` or ``PatientSplit``, never a split by sample. In
version 1.1.3, 538 of the 6119 contacts (8.8%) are in the seizure onset
zone.

Each sample also carries the name of its contact and its position ``x``,
``y``, ``z`` in millimetres, as stored in ``electrodes.tsv``, NaN for the 2%
of contacts without a position. They are not model inputs by default.

Task parameters:

- target: ``"soz"`` (default) or ``"resect"``
- window_seconds: float, default=30
- stride_seconds: float, default=30
- resample_rate: float, default=200
- bandpass: ``(low, high)`` in Hz or None, default=(0.5, 80)
- notch: float or None, default=60
- reference: ``"average"`` (default) or ``"stored"``
- outcomes: tuple of outcome codes or None, default=None (every patient);
  ``("S",)`` keeps the patients who became seizure free

.. autoclass:: pyhealth.tasks.SOZLocalizationHUP
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.soz_localization_hup.read_hup_channels

.. autofunction:: pyhealth.tasks.soz_localization_hup.read_hup_electrodes
