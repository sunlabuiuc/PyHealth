pyhealth.tasks.temple_university_EEG_tasks
=========================================

The temple_university_EEG_tasks module contains the tasks for the Temple University EEG Corpus.

The tasks are:
- EEGEventsTUEV: EEG event classification task for the TUEV dataset.
- EEGAbnormalTUAB: Binary classification task for the TUAB dataset (abnormal vs normal).

Tasks Parameters:
- resample_rate: int, default=200 # Resample rate
- bandpass_filter: tuple, default=(0.1, 75.0) # Bandpass filter
- notch_filter: float, default=50.0 # Notch filter

.. autoclass:: pyhealth.tasks.temple_university_EEG_tasks.EEGEventsTUEV
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.temple_university_EEG_tasks.EEGAbnormalTUAB
    :members:
    :undoc-members:
    :show-inheritance: