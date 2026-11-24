pyhealth.datasets.HelsinkiNeonatalDataset
=========================================

Dataset is available at https://zenodo.org/records/4940267

The Helsinki neonatal EEG dataset holds multi-channel EEG of 79 term
neonates admitted to the neonatal intensive care unit of the Helsinki
University Hospital, with a median recording duration of 74 minutes. Each
recording has 19 referential EEG channels of the 10-20 system plus ECG and
respiration, sampled at 256 Hz. Three experts annotated seizures
independently, second by second. The data are open under the CC BY 4.0
license.

Point ``root`` at the folder holding the ``eeg<N>.edf`` files, the three
``annotations_2017_*.csv`` files and ``clinical_information.csv``. The
dataset writes one metadata row per recording. The default task,
``SeizureDetectionHelsinki``, builds an 18-channel bipolar montage and cuts
16 second windows labelled by the consensus of the three experts.

.. autoclass:: pyhealth.datasets.HelsinkiNeonatalDataset
    :members:
    :undoc-members:
    :show-inheritance:
