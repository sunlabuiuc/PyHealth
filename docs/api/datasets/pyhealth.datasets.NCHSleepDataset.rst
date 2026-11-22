pyhealth.datasets.NCHSleepDataset
=================================

Overview
--------

The NCH Sleep DataBank holds 3,984 clinical sleep studies (polysomnography)
of 3,673 patients, mostly children, recorded at Nationwide Children's
Hospital. Each study is an EDF recording of EEG, EOG, EMG, ECG and breathing
channels (nasal pressure, airflow, chest and abdomen effort, SpO2 and
end-tidal CO2), with scored sleep stages and respiratory events such as
obstructive apneas and hypopneas. Electronic health record tables come with
the recordings. Access is through NSRR and needs an approved data request.

Point ``root`` at the downloaded ``nchsdb`` folder, the one holding
``sleep_data/`` and ``health_data/``. The dataset writes one metadata row
per study, with the EDF and annotation paths, the sampling rate, duration
and channel labels of the recording, the number of apneas and hypopneas,
an apnea-hypopnea index counted from the annotations, the age at the study
and the patient's demographics. A patient can have several studies, so
split by patient.

Source: `NSRR nchsdb <https://sleepdata.org/datasets/nchsdb>`_, described in
`Lee et al., Scientific Data 9:421, 2022
<https://doi.org/10.1038/s41597-022-01545-6>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.NCHSleepDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.nch_sleep.read_nch_annotations

.. autofunction:: pyhealth.datasets.nch_sleep.summarize_nch_annotations
