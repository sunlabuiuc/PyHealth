pyhealth.tasks.SleepStagingISRUC
====================================

Sleep staging on ISRUC-Sleep.

For each recording the task reads the scoring of one expert, keeps the
selected channels (the six EEG channels F3, F4, C3, C4, O1 and O2 by
default, matched without their A1/A2 or M1/M2 reference) in microvolts, and
cuts the signal into epochs. ISRUC scores every 30 seconds, so with
``epoch_seconds=10`` each score labels three epochs.

Labels are 0 W, 1 N1, 2 N2, 3 N3 and 4 REM. Recordings missing a channel
are skipped with a warning.

Task parameters:

- channels: channel names without the reference, default the six EEG channels
- epoch_seconds: int, default=30, has to divide 30
- annotator: 1 or 2, default=1

.. autoclass:: pyhealth.tasks.SleepStagingISRUC
    :members:
    :undoc-members:
    :show-inheritance:
