pyhealth.tasks.ecg_delineation_ludb
=================================

Four classes: background=0, P=1, QRS=2, T=3. Native ``BaseTask`` consumes
manifest window events and reads each patient's WFDB record and frozen masks.
Install the optional ``wfdb`` reader in the chosen environment before real
data use; importing this task does not require WFDB or download data.

The task uses existing PyHealth processors, without changing core classes:

* ``signal``: float32 tensor (512,), no loader-level normalization.
* ``ignore_mask``: bool tensor (512,), retained as a mask, never background.
* ``patient_context``: raw string, preserving every supplied metadata field.
* ``label``: ``timeseries_multiclass`` output with fixed labels [0,1,2,3].
  Only non-ignored positions appear in the label records.
* ``dense_label``: long tensor (512,) output for independent auditing; not
  included as a prediction input feature.

The native label collator produces ``(values, time, offsets)`` for each batch.
All-ignored windows remain in the dataset, with empty values/times. The model
checks that sparse times agree exactly with the full frozen ignore mask.
Window boundaries, lead order and supervision flags are checked explicitly.
No patient-level train/test split is inferred inside the task.

.. autoclass:: pyhealth.tasks.ecg_delineation_ludb.ECGDelineationLUDB
    :members:
    :show-inheritance:
