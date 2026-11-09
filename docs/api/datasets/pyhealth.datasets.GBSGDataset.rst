pyhealth.datasets.GBSGDataset
=============================

Overview
--------

Load the pooled Rotterdam/GBSG breast-cancer survival benchmark from a local
CSV. This is the 2,232-row, seven-feature benchmark, not the separate GBSG2
dataset. Source group/row identifiers are surrogates, not original patient IDs.
Equal recorded values are retained; duplicate identifiers are rejected.

Prepare a separately obtained DeepSurv HDF5 release with ``prepare_gbsg_csv``.
That optional preparation step requires ``h5py``. The loader does not download
data. Its default task keeps source months and emits separate duration/event
tensors for :class:`pyhealth.models.SurvivalMDN`.

The explicitly named paper-time and released-split helpers reproduce Han et al.'s
convention. Their GBSG time divisor uses the full cohort and must not be confused
with a transform fitted only on training patients.

API reference
-------------

.. autoclass:: pyhealth.datasets.GBSGDataset
    :members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.prepare_gbsg_csv
.. autofunction:: pyhealth.datasets.gbsg_paper_time_transform
.. autofunction:: pyhealth.datasets.gbsg_released_split_ids

See :doc:`../../survival_pipeline` for the native pipeline and manual checks.
