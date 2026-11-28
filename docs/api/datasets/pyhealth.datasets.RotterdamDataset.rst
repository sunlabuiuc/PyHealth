pyhealth.datasets.RotterdamDataset
==================================

Overview
--------

Load the Rotterdam tumour-bank breast-cancer cohort from a local CSV, keeping
recurrence and death as two separate outcomes. The file has one row per
patient: 2,982 patients with primary breast cancer (surgery 1978-1993), ten
baseline covariates, and for each outcome its own follow-up time in days and
event indicator (``rtime``/``recur`` and ``dtime``/``death``). 1,518 patients
recurred and 1,272 died; 1,077 had both events and 1,269 neither.

This is the full multi-event cohort. It is not the 2,232-row pooled
Rotterdam/GBSG benchmark loaded by :doc:`pyhealth.datasets.GBSGDataset`, which
has seven features and a single recurrence-free survival endpoint in months.

Patient times are kept exactly as recorded. 43 patients died after their
recurrence follow-up had ended without a recorded recurrence (``recur=0``,
``death=1``, ``rtime < dtime``); their recurrence stays censored at ``rtime``
and is not extended to the death time. Tied recurrence and death times are
kept, and tumour ``size`` stays one of its three source categories
(``"<=20"``, ``"20-50"``, ``">50"`` mm); numeric coding belongs to the task.
The whole CSV is validated when the dataset is created, before anything is
cached, and errors name the column and ``pid``. The default task is
:class:`~pyhealth.tasks.MultiEventSurvivalRotterdam`.

Data source
-----------

The cohort is distributed with the R ``survival`` package as
`survival::rotterdam`_, from Royston and Altman, `External validation of a Cox
prognostic model: principles and methods`_ (BMC Medical Research Methodology,
2013). It is open; no account or agreement is needed. The MENSA authors ship
the same table as `data/rotterdam.csv`_ in their repository: download that
file (**Download raw file**) and put it in the folder you pass as ``root``.
Alternatively, export it from R with
``write.csv(survival::rotterdam, "rotterdam.csv", row.names = FALSE)``; the
loader validates whichever file you use. The loader never downloads anything.

The CSV needs these 15 columns: ``pid``, ``year``, ``age``, ``meno``,
``size``, ``grade``, ``nodes``, ``pgr``, ``er``, ``hormon``, ``chemo``,
``rtime``, ``recur``, ``dtime`` and ``death``.

Usage
-----

.. code-block:: python

    from pyhealth.datasets import RotterdamDataset, split_by_patient

    dataset = RotterdamDataset(root="/path/to/folder")  # holds rotterdam.csv
    samples = dataset.set_task()  # MultiEventSurvivalRotterdam by default
    train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=0)

See :doc:`../tasks/pyhealth.tasks.multi_event_survival_rotterdam` for the
sample format and :doc:`../models/pyhealth.models.MENSA` for a model that
uses it.

API reference
-------------

.. autoclass:: pyhealth.datasets.RotterdamDataset
    :members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.rotterdam.parse_rotterdam_record

.. _survival::rotterdam: https://stat.ethz.ch/R-manual/R-devel/library/survival/html/rotterdam.html
.. _External validation of a Cox prognostic model\: principles and methods: https://doi.org/10.1186/1471-2288-13-33
.. _data/rotterdam.csv: https://github.com/thecml/mensa/blob/main/data/rotterdam.csv
