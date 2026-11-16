pyhealth.tasks.DementiaClassificationAHEPA
==========================================

Dementia classification on the AHEPA resting-state EEG dataset.

For each subject the task reads the 19 channels of the artifact-cleaned (or
unprocessed) recording in microvolts and cuts overlapping windows, each
labelled with the diagnosis of the subject. The label is ``binary`` for the
two-class problems and ``multiclass`` for ``"three_class"``.

Every window of a subject carries the same label, so windows of one subject
must stay on one side of a split: use ``split_by_patient`` or
``PatientSplit``, never a split by sample.

Task parameters:

- problem: ``"ad_vs_cn"`` (default), ``"ftd_vs_cn"``, ``"ad_vs_ftd"``,
  ``"dementia_vs_cn"`` or ``"three_class"``
- window_seconds: float, default=30
- stride_seconds: float, default=15
- variant: ``"derivatives"`` (default) or ``"raw"``
- resample_rate: float or None, default=None (500 Hz)

.. autoclass:: pyhealth.tasks.DementiaClassificationAHEPA
    :members:
    :undoc-members:
    :show-inheritance:
