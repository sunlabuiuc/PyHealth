pyhealth.datasets.DREAMTDataset
===================================

The Dataset for Real-time sleep stage EstimAtion using Multisensor wearable
Technology (DREAMT) includes wrist-based wearable and polysomnography (PSG)
sleep data from 100 participants recruited from the Duke University Health
System (DUHS) Sleep Disorder Lab.

This includes wearable signals, PSG signals, sleep labels, and clinical data
related to sleep health and disorders.

``DREAMTDataset`` supports both official DREAMT release layouts and partial
local subsets. It builds metadata linking each patient to locally available
signal files and exposes data for both the simplified
``SleepStagingDREAMT`` window-classification task and the more sequence-style
``SleepStagingDREAMTSeq`` task.

Refer to the `doc <https://physionet.org/content/dreamt/>`_ for more
information about the dataset.

.. autoclass:: pyhealth.datasets.DREAMTDataset
    :members:
    :undoc-members:
    :show-inheritance:
