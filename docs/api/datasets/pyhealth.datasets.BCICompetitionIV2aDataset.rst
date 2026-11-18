pyhealth.datasets.BCICompetitionIV2aDataset
===========================================

Overview
--------

BCI Competition IV dataset 2a (BNCI Horizon 001-2014): four-class motor
imagery EEG (left hand, right hand, feet, tongue) from 9 subjects, 22 EEG
channels at 250 Hz, two sessions of 288 trials each. Open access under
CC BY-ND 4.0 (do not redistribute modified or preprocessed copies). ``root``
is the folder holding ``A0{1..9}{T,E}.mat``; ``download=True`` fetches them.
Session T is the official training set and session E the test set.

Source: `BNCI Horizon 2020, 001-2014 <https://bnci-horizon-2020.eu/database/data-sets>`_
(`description <https://www.bbci.de/competition/iv/desc_2a.pdf>`_)

API Reference
-------------

.. autoclass:: pyhealth.datasets.BCICompetitionIV2aDataset
    :members:
    :undoc-members:
    :show-inheritance:
