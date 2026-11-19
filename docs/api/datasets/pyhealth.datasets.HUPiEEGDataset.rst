pyhealth.datasets.HUPiEEGDataset
================================

Overview
--------

The HUP iEEG dataset holds intracranial EEG of 57 patients with
drug-resistant epilepsy, recorded at the Hospital of the University of
Pennsylvania before resection or laser ablation, with ECoG or SEEG
electrodes. It holds interictal recordings of 5 minutes and recordings of
seizures, 319 EDF files in total, at 256 to 1024 Hz and with
52 to 232 channels. For every recording the dataset marks the contacts of
the clinically determined seizure onset zone and those that overlap with the
resected or ablated tissue, and it gives the surgical outcome of each
patient. The data follow BIDS and are open under the CC0 license, with no
registration.

Point ``root`` at the downloaded folder, the one holding
``participants.tsv`` and the ``sub-HUP*`` folders. The dataset writes one
metadata row per recording, with its ``channels.tsv``, the
``electrodes.tsv`` of its patient and implant (positions of the contacts),
its kind
(interictal or ictal), the seizure onset and offset of ictal recordings, and the
clinical columns of the patient. The default task, ``SOZLocalizationHUP``,
cuts the interictal recordings into single-channel windows labelled with the
seizure onset zone. Split the windows by patient.

Source: `OpenNeuro ds004100 <https://openneuro.org/datasets/ds004100>`_,
described in `Bernabei, Li et al., Brain 146(6):2248-2258, 2023
<https://doi.org/10.1093/brain/awad007>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.HUPiEEGDataset
    :members:
    :undoc-members:
    :show-inheritance:
