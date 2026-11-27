pyhealth.datasets.ISRUCDataset
===================================

Dataset is available at https://sleeptight.isr.uc.pt/

ISRUC-Sleep is a polysomnography (PSG) dataset in three subgroups: 100
subjects with sleep disorders (subgroup I), 8 subjects recorded twice
(subgroup II) and 10 healthy subjects (subgroup III). Each recording is an
EDF file with a ``.rec`` extension, sampled at 200 Hz, and two experts
scored every 30 second epoch with AASM stages.

Point ``root`` at one subgroup folder. The dataset writes one metadata row
per recording and gives each ``.rec`` file an ``.edf`` link, since MNE only
opens EDF files with that extension. The default task,
``SleepStagingISRUC``, cuts each recording into epochs on the six EEG
channels F3, F4, C3, C4, O1 and O2.

.. autoclass:: pyhealth.datasets.ISRUCDataset
    :members:
    :undoc-members:
    :show-inheritance:
