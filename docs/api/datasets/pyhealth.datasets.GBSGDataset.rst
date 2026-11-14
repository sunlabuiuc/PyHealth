pyhealth.datasets.GBSGDataset
=============================

Overview
--------

Load the pooled Rotterdam/GBSG breast-cancer survival benchmark from a local
CSV. This is the 2,232-row, seven-feature benchmark, not the separate GBSG2
dataset. Source group/row identifiers are surrogates, not original patient IDs.
Equal recorded values are retained; duplicate identifiers are rejected.

The benchmark comes from `Katzman et al. (2018), DeepSurv`_. Open the
`gbsg_cancer_train_test.h5`_ file in the DeepSurv repository and select
**Download raw file**. Pass the local file to ``prepare_gbsg_csv``. That
preparation step requires ``h5py``. The loader's default task keeps source
months and emits separate duration/event tensors for
:class:`pyhealth.models.SurvivalMDN`.

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

.. _Katzman et al. (2018), DeepSurv: https://doi.org/10.1186/s12874-018-0482-1
.. _gbsg_cancer_train_test.h5: https://github.com/jaredleekatzman/DeepSurv/blob/master/experiments/data/gbsg/gbsg_cancer_train_test.h5
