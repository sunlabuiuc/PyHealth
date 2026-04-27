pyhealth.tasks.TUSZTask
=========================================

The TUSZTask module contains the tasks for the Temple University EEG Seizure Corpus.
It defines the workflow of preprocessing and utilizes TUSZHelper class to fulfill the details.

The workflow is:
- Read edf: read .edf files from the TUSZ dataset.
- Skip certain files
- Get labels
- Get patient status
- Resample signals
- Transform labels with resampled signals
- Segment signals
- Convert labels to binary targets and bytes
- Create bipolar signals

Tasks Parameters:
- sample_rate: int, default=200
- feature_sample_rate: int, default=200
- label_type: str, default='csv'
- eeg_type: str, default='bipolar' # bipolar, uni_bipolar
- min_binary_slicelength: int, default=30
- min_binary_edge_seiz: int, default=3

.. autoclass:: pyhealth.tasks.TUSZTask
    :members:
    :undoc-members:
    :show-inheritance:
