pyhealth.datasets.DREAMTDataset
===================================

The Dataset for Real-time sleep stage EstimAtion using Multisensor wearable
Technology (DREAMT) includes wrist-based wearable and polysomnography (PSG)
sleep data from 100 participants recruited from the Duke University Health
System (DUHS) Sleep Disorder Lab.

This includes wearable signals, PSG signals, sleep labels, and clinical data
related to sleep health and disorders.

``DREAMTDataset`` supports both the official PhysioNet release layout and
processed per-subject files named with DREAMT subject identifiers. It builds a
metadata table with signal references so tasks such as
``SleepStagingDREAMT`` can create fixed-size wearable windows for model
training. In partial local downloads, subjects without detected local signal
files are skipped automatically.

Refer to the `doc <https://physionet.org/content/dreamt/>`_ for more
information about the dataset.

.. autoclass:: pyhealth.datasets.DREAMTDataset
    :members:
    :undoc-members:
    :show-inheritance:

   

   
   
   
