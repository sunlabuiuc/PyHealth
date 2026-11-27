pyhealth.tasks.SleepStagingHMC
====================================

Sleep staging on the Haaglanden Medisch Centrum (HMC) database.

For each recording the task reads the hypnogram and the lights markers from
``SNxxx_sleepscoring.txt``, keeps the selected channels (the four EEG
derivations by default) in microvolts, and cuts the signal into epochs. HMC
scores every 30 seconds, so with ``epoch_seconds=10`` each score labels three
epochs. By default only the scored epochs that lie entirely between the
lights off and lights on markers are kept.

Labels are 0 W, 1 N1, 2 N2, 3 N3 and 4 REM. Recordings missing a channel
are skipped with a warning.

Task parameters:

- channels: EDF channel names, default ``HMC_EEG_CHANNELS`` (four EEG
  derivations); ``HMC_PSG_CHANNELS`` adds the two EOG and the chin EMG
- epoch_seconds: int, default=30, has to divide 30
- lights_period_only: bool, default=True

.. autoclass:: pyhealth.tasks.SleepStagingHMC
    :members:
    :undoc-members:
    :show-inheritance:
