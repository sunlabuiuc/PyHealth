pyhealth.tasks.ECGDelineationLUDB
=================================

ECG delineation on :class:`~pyhealth.datasets.LUDBDataset`: label every
point of a lead as P wave, QRS complex, T wave or none.

The task follows Park et al., "Benchmarking ECG Delineation using Deep
Neural Network-based Semantic Segmentation Models", CHIL 2025
(`PMLR 287:63-88 <https://proceedings.mlr.press/v287/park25a.html>`_). As in
the paper, each lead of each record is one sample, and the signal is
filtered with a Butterworth 0.05 Hz high-pass and 150 Hz low-pass. Models
are scored by the IoU of each wave and their mean (mIoU).

Each sample holds:

- ``signal``: float32, shape ``(1, 1, n_samples)``, in mV. The extra
  dimension makes it a one-row image, so PyHealth's ``UNet2D`` can use it.
- ``mask``: int64, shape ``(1, n_samples)``. Values are 0 background,
  1 P wave, 2 QRS complex, 3 T wave and 255 for points the cardiologists
  did not annotate (before the first and after the last wave). Pass
  ``ignore_index=255`` to the model.
- ``patient_id`` (the record number) and ``lead``.

The paper predicts three sigmoid channels with binary cross-entropy; one
class per point holds the same information because LUDB's waves never
overlap, and works with PyHealth's segmentation models and metrics.
Splitting with ``split_by_patient`` keeps all leads of a record together,
as the paper does.

Task parameters:

- leads: lead names (any case), default all 12
- apply_filter: bool, apply the band-pass filter, default True
- noise_snr_db: float or None, add white Gaussian noise at this
  signal-to-noise ratio before filtering, default None
- noise_seed: int, default 0

.. code-block:: python

    from pyhealth.datasets import LUDBDataset, split_by_patient
    from pyhealth.models import UNet2D
    from pyhealth.tasks import ECGDelineationLUDB

    dataset = LUDBDataset(root="/path/to/ludb/1.0.1")
    samples = dataset.set_task(ECGDelineationLUDB())
    train, val, test = split_by_patient(samples, [0.8, 0.1, 0.1], seed=0)
    model = UNet2D(train, in_channels=1, num_classes=4, ignore_index=255)

.. autoclass:: pyhealth.tasks.ECGDelineationLUDB
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ecg_delineation_ludb.annotations_to_mask
