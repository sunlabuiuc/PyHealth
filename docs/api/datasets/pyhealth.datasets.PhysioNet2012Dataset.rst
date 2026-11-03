pyhealth.datasets.PhysioNet2012Dataset
======================================

Overview
--------

The PhysioNet/Computing in Cardiology Challenge 2012 dataset holds 12,000 ICU
stays of at least 48 hours (sets a, b and c). Each record has up to 37
irregularly sampled vital-sign and lab time series from the first 48 hours,
six descriptors recorded on admission, and outcomes (SAPS-I, SOFA, length of
stay, survival and in-hospital death). It is open access, so
``PhysioNet2012Dataset(root, download=True)`` fetches it (about 20 MB).

For more information see `PhysioNet Challenge 2012
<https://physionet.org/content/challenge-2012/1.0.0/>`_ and Silva et al.,
Computing in Cardiology 2012.

The dataset builds two tables from the raw record files and writes them into
``root``: ``events`` (one row per observation, with ``parameter``, ``value``
and ``minutes`` since ICU admission) and ``patients`` (descriptors and
outcomes). Use it with :class:`~pyhealth.tasks.PhysioNet2012Forecasting` for
irregular time-series forecasting.

API Reference
-------------

.. autoclass:: pyhealth.datasets.PhysioNet2012Dataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.physionet2012.download_physionet2012

.. autofunction:: pyhealth.datasets.physionet2012.prepare_physionet2012

.. autofunction:: pyhealth.datasets.physionet2012.read_record
