pyhealth.datasets.CardiologyDataset
===================================

Dataset is available at https://physionet.org/content/challenge-2020/1.0.2/
(open access, CC BY 4.0).

The training set of the PhysioNet/CinC Challenge 2020 pools 43,101 12-lead
ECGs from six sources: ``cpsc_2018``, ``cpsc_2018_extra``, ``georgia``,
``ptb``, ``ptb-xl`` and ``st_petersburg_incart``. Each source is loaded as
its own table, and every recording is its own patient, since the Challenge
gives no patient identifiers. Pick sources with ``sources=[...]`` (or the
1.x ``chosen_dataset`` flags). The default task, ``CardiologyDetection``,
cuts 10-second windows at 500 Hz and labels one diagnosis category.

.. autoclass:: pyhealth.datasets.CardiologyDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.cardiology.parse_header
