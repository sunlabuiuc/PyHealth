pyhealth.models.DynamicSurvivalModel
=====================================

GRU/LSTM-based Dynamic Survival Analysis model for ICU early-event prediction.

The model implements the DSA pipeline from Yèche et al. (CHIL 2024):
a linear embedding layer with L1 regularisation, a stacked recurrent encoder
(GRU or LSTM), and a hazard head that outputs per-horizon failure probabilities
λ̂(k | X_t) for k = 1 … horizon. At inference the cumulative failure
probability F(h | X_t) at the last observed timestep is used as the alarm
score.

**Reference**: Yèche H. et al., *Dynamic Survival Analysis for Early Event
Prediction*, Proceedings of Machine Learning for Health (CHIL), 2024.
https://proceedings.mlr.press/v248/yeche24a.html

Quick Start
-----------

.. code-block:: python

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import DynamicSurvivalModel
    from pyhealth.tasks import DecompensationDSA
    from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples

    # 1. Build dataset from synthetic data
    samples = make_synthetic_dsa_samples(n_patients=200, n_features=8, horizon=24)
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=DecompensationDSA.input_schema,
        output_schema=DecompensationDSA.output_schema,
        dataset_name="dsa_synthetic",
    )

    # 2. Instantiate model
    model = DynamicSurvivalModel(
        dataset=dataset,
        input_dim=8,
        hidden_dim=256,
        horizon=24,
    )

    # 3. Forward pass
    loader = get_dataloader(dataset, batch_size=16, shuffle=True)
    out = model(**next(iter(loader)))
    # out: {"loss": ..., "y_prob": (B,1), "y_true": (B,1), "logit": (B,1)}

API Reference
-------------

.. autoclass:: pyhealth.models.DynamicSurvivalModel
    :members:
    :undoc-members:
    :show-inheritance:
