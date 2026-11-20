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

Getting the data
----------------

- **Source page:** `BNCI Horizon 2020 data sets <https://bnci-horizon-2020.eu/database/data-sets>`_,
  entry "1. Four class motor imagery (001-2014)" (originally data set 2a of
  BCI Competition IV). Paradigm description:
  `desc_2a.pdf <https://www.bbci.de/competition/iv/desc_2a.pdf>`_.
- **Paper:** M. Tangermann, K.-R. Muller, A. Aertsen, N. Birbaumer, C. Braun,
  C. Brunner, et al. "Review of the BCI Competition IV." *Frontiers in
  Neuroscience* 6, 55 (2012). https://doi.org/10.3389/fnins.2012.00055
- **Access:** open download. No account, registration or data-use agreement.
- **Files:** 18 MATLAB files ``A01T.mat``, ``A01E.mat`` ... ``A09T.mat``,
  ``A09E.mat``, 37-46 MB each (779,873,919 bytes, about 780 MB, in all).
  ``T`` is each subject's training session and ``E`` the evaluation session.
- **Download:** pass ``download=True`` to fetch the requested files into
  ``root``, or run:

  .. code-block:: bash

      mkdir -p bnci/001-2014 && cd bnci/001-2014
      for s in 1 2 3 4 5 6 7 8 9; do for x in T E; do
        wget https://bnci-horizon-2020.eu/database/data-sets/001-2014/A0${s}${x}.mat
      done; done

  The links redirect to ``https://lampx.tugraz.at/~bci/database/001-2014/``.
- **Unpack:** nothing to unpack; the ``.mat`` files are read directly.
- **Layout:** only the files for the requested ``subjects`` and ``sessions``
  need to be present.

  .. code-block:: text

      <root>/
          A01T.mat
          A01E.mat
          ...
          A09T.mat
          A09E.mat

  The derived metadata goes to ``metadata_cache_dir``, never into ``root``.
  The BNCI ``E`` files include the evaluation labels (the original
  competition GDF release does not), so the GDF files are not supported.
- **Licence:** `CC BY-ND 4.0 <https://creativecommons.org/licenses/by-nd/4.0/>`_,
  licensor Institute for Knowledge Discovery, Graz University of Technology.
  You may use and share the original files with attribution, but you must
  not distribute modified or preprocessed copies (including the cached
  samples PyHealth produces). There is no non-commercial (NC) term.

API Reference
-------------

.. autoclass:: pyhealth.datasets.BCICompetitionIV2aDataset
    :members:
    :undoc-members:
    :show-inheritance:
