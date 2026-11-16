pyhealth.datasets.AHEPADementiaDataset
======================================

Dataset is available at https://openneuro.org/datasets/ds004504

The AHEPA dataset holds eyes-closed resting-state EEG of 88 subjects recorded
at the AHEPA General Hospital of Thessaloniki: 36 with Alzheimer's disease,
23 with frontotemporal dementia and 29 healthy controls. Each subject has one
recording of 19 scalp channels at 500 Hz, in an unprocessed and an
artifact-cleaned version. The data follow BIDS and are open under the CC0
license.

Point ``root`` at the downloaded folder, the one holding
``participants.tsv``. The dataset writes one metadata row per subject, with
both recordings, the group, the gender, the age and the MMSE score. The
default task, ``DementiaClassificationAHEPA``, cuts the artifact-cleaned
recordings into 30 second windows labelled with the diagnosis of the subject.
Split the windows by patient.

.. autoclass:: pyhealth.datasets.AHEPADementiaDataset
    :members:
    :undoc-members:
    :show-inheritance:
