"""MedMod (Elsharief et al., 2025) in-hospital mortality: the parts behind
``MedModMIMIC4Dataset`` and ``MedModInHospitalMortality``.

* ``extraction``: cohort, labels, value cleaning, time series and CXR pairing,
  one subject at a time, following MedMod's code line by line.
* ``discretizer``: MedMod's Discretizer and Normalizer.
* ``downloads``: MedMod's subject lists and item map, downloaded from MedMod's
  GitHub (pinned commit, SHA-256 checked) on first use.
* ``files/``: MedMod's discretizer config and normalizer (Apache-2.0, see
  ``files/NOTICE``).
"""
