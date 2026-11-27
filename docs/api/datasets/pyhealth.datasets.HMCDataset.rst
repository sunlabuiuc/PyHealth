pyhealth.datasets.HMCDataset
===================================

Dataset is available at https://physionet.org/content/hmc-sleep-staging/1.1/

The Haaglanden Medisch Centrum (HMC) sleep staging database holds 151
whole-night polysomnography (PSG) recordings collected in 2018 at a clinical
sleep center in The Netherlands, from patients referred for various sleep
disorders. Sleep technicians scored every 30 second epoch with the AASM
stages (W, N1, N2, N3, R). Signals are sampled at 256 Hz: four EEG
derivations (F4-M1, C4-M1, O2-M1, C3-M2), two EOG, chin EMG and ECG. The
data are open under the CC BY 4.0 license.

Point ``root`` at the downloaded ``recordings`` folder or its parent. The
dataset writes one metadata row per recording that has a
``_sleepscoring.txt`` file, in ``hmc-pyhealth.csv``. This file is written
once: after completing a partial download, delete it so the new recordings
are included. The default task, ``SleepStagingHMC``, cuts each
recording into 30 second epochs on the four EEG derivations, between the
lights off and lights on markers.

.. autoclass:: pyhealth.datasets.HMCDataset
    :members:
    :undoc-members:
    :show-inheritance:
