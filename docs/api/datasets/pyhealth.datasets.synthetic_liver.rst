pyhealth.datasets.synthetic_liver
=================================

Seeded, tie-free synthetic cohorts in the ``pbcseq`` layout, with outcomes drawn
from a time-varying Cox model whose true effects are known. Use them to test
survival models and to run the MAS example without real patient data.

.. code-block:: python

    from pyhealth.datasets import create_synthetic_liver_dataset, get_dataloader

    dataset = create_synthetic_liver_dataset(n_patients=20, seed=0)
    batch = next(iter(get_dataloader(dataset, batch_size=len(dataset))))
    # batch["features"]: (B, 4); batch["survival_interval"]: (B, 3)

.. currentmodule:: pyhealth.datasets.synthetic_liver

.. autofunction:: generate_synthetic_pbcseq

.. autofunction:: create_synthetic_liver_dataset

.. autofunction:: write_synthetic_pbcseq_csv

.. autodata:: MAS4_FEATURES

.. autodata:: MAS6_FEATURES
