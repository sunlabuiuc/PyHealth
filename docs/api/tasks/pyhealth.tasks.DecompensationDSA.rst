pyhealth.tasks.DecompensationDSA
=================================

ICU decompensation prediction task for Dynamic Survival Analysis.

:class:`DecompensationDSA` is a :class:`~pyhealth.tasks.BaseTask` subclass
that extracts per-admission ICU time series and binary decompensation labels
from a PyHealth patient object. For synthetic experimentation (no external
dataset required), use :func:`~pyhealth.tasks.decompensation_dsa.make_synthetic_dsa_samples`
directly.

**Reference**: Yèche H. et al., *Dynamic Survival Analysis for Early Event
Prediction*, Proceedings of Machine Learning for Health (CHIL), 2024.
https://proceedings.mlr.press/v248/yeche24a.html

Quick Start — synthetic data
-----------------------------

.. code-block:: python

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.tasks import DecompensationDSA
    from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples

    # Build samples without any external dataset
    samples = make_synthetic_dsa_samples(
        n_patients=200,
        n_features=8,
        horizon=24,
        max_seq_len=100,
        event_rate=0.3,
        seed=42,
    )
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=DecompensationDSA.input_schema,
        output_schema=DecompensationDSA.output_schema,
        dataset_name="dsa_synthetic",
    )
    loader = get_dataloader(dataset, batch_size=16, shuffle=True)

Schemas
-------

**input_schema**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Processor
     - Description
   * - ``timeseries``
     - ``"tensor"``
     - Pre-padded float matrix of shape ``(max_seq_len, n_features)``

**output_schema**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Processor
     - Description
   * - ``label``
     - ``"binary"``
     - 1 if the patient decompensated within the prediction horizon, 0 otherwise

API Reference
-------------

.. autoclass:: pyhealth.tasks.DecompensationDSA
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.decompensation_dsa.make_synthetic_dsa_samples
