pyhealth.datasets.AHEPADementiaDataset
======================================

Overview
--------

The AHEPA dataset holds eyes-closed resting-state EEG of 88 subjects recorded
at the AHEPA General Hospital of Thessaloniki: 36 with Alzheimer's disease,
23 with frontotemporal dementia and 29 healthy controls. Each subject has one
recording of 19 scalp channels at 500 Hz, in an unprocessed and an
artifact-cleaned version. The data follow BIDS and are open under the CC0
license, with no registration.

Point ``root`` at the downloaded folder, the one holding
``participants.tsv``, the ``sub-0XX`` folders and ``derivatives``. The
dataset writes one metadata row per subject, with both recordings, the
group, the gender, the age and the MMSE score. The default task,
``DementiaClassificationAHEPA``, cuts the artifact-cleaned recordings into
4 second windows labelled with the diagnosis of the subject. Split the
windows by patient.

Source: `OpenNeuro ds004504 <https://openneuro.org/datasets/ds004504>`_,
described in `Miltiadous et al., Data 8(6):95, 2023
<https://doi.org/10.3390/data8060095>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.AHEPADementiaDataset
    :members:
    :undoc-members:
    :show-inheritance:
