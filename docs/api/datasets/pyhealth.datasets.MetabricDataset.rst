pyhealth.datasets.MetabricDataset
=================================

Overview
--------

METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) in the
nine-feature survival version released by DeepSurv and used by SODEN and
Survival Mixture Density Networks (Han et al., MLHC 2022): 1,904 breast cancer
patients with four gene-expression values (MKI67, EGFR, PGR, ERBB2), age at
diagnosis, hormone treatment, radiotherapy, chemotherapy and ER status, the
follow-up time in months and a death indicator. About 42% are censored.

The dataset is built from the ten published train/validation/test splits in
the `Survival MDN repository <https://github.com/XintianHan/Survival-MDN>`_
(``data/metabric/{train,valid,test}_{1..10}.npz``). :func:`prepare_metabric`
turns them into one ``metabric.csv`` with raw features and the split
membership of every patient, and :meth:`MetabricDataset.paper_split` returns
one of those splits as a :class:`~pyhealth.datasets.SubjectSplit` for
``set_task(task, split=...)``, so that preprocessing is fitted on the training
patients only, as in the papers.

.. code-block:: python

    from pyhealth.datasets import MetabricDataset
    from pyhealth.tasks import MetabricSurvivalPrediction

    dataset = MetabricDataset(root="/data/metabric", download=True)
    train, valid, test = dataset.set_task(
        MetabricSurvivalPrediction(), split=dataset.paper_split(1)
    )

Use with :class:`~pyhealth.tasks.MetabricSurvivalPrediction` and
:class:`~pyhealth.models.SurvivalMDN`.

.. autoclass:: pyhealth.datasets.MetabricDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.metabric.prepare_metabric
