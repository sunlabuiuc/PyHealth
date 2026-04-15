pyhealth.tasks.TUSZHelper
=========================================

The TUSZHelper module contains the actual implementation of the substeps of TUSZTask.

The substeps are:
- skip_file: skip certain files
- is_seizure_patient: get patient status
- process_label: get labels
- resample: resample signals to the intended sample rate
- transform_labels_with_resampled_signals: transform labels with resampled signals
- segment_signals: segment signals according to 
        min_binary_slicelength, min_binary_edge_seiz, feature_sample_rate and sample_rate
- convert_labels: convert labels according to different requirements
- create_bipolar_signals: create bipolar signals

Tasks Parameters (defaults taken from TUSZTask):
- sample_rate: int, default=200
- feature_sample_rate: int, default=200
- label_type: str, default='csv'
- eeg_type: str, default='bipolar' # bipolar, uni_bipolar
- min_binary_slicelength: int, default=30
- min_binary_edge_seiz: int, default=3

.. autoclass:: pyhealth.tasks.TUSZHelper
    :members:
    :undoc-members:
    :show-inheritance:
