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
- window_seconds: float, default=4
- stride_seconds: float, default=2
- variant: ``"derivatives"`` (default) or ``"raw"``
- resample_rate: float or None, default=None (500 Hz)
- reference: ``"stored"`` (default) or ``"average"``

Reference electrodes
--------------------

The artifact-cleaned recordings are referenced to the average of the mastoid
electrodes A1 and A2 (data descriptor, section 3.2), so whatever these two
electrodes pick up is added to all 19 channels. In the 88 cleaned recordings
of version 1.0.9, the mean over channels carries a median of 94% of the
signal variance (90% to 96% for the middle half of the subjects), mostly
below 4 Hz.

- ``reference="stored"`` (default) keeps this signal, as distributed.
- ``reference="average"`` subtracts the mean over the 19 channels at each
  time. `Bigdely-Shamlo et al. (2015) <https://doi.org/10.3389/fninf.2015.00016>`_
  describe a mastoid reference as a single point of failure whose noise is
  common to all channels, and report that re-referencing such data to the
  average gives essentially the same result as an average reference from
  the start. An average over 19 scalp channels is not a neutral reference
  either: `Lei and Liao (2017) <https://doi.org/10.3389/fnins.2017.00205>`_
  note that it is valid only with sufficient electrode density and coverage
  of the head.

For the unprocessed recordings (``variant="raw"``), the BIDS sidecar files
give A1 and A2 as reference, while the data descriptor mentions a referential
montage that uses Cz for common mode rejection. They are not re-referenced
unless ``reference="average"``.

.. autoclass:: pyhealth.tasks.DementiaClassificationAHEPA
    :members:
    :undoc-members:
    :show-inheritance:
