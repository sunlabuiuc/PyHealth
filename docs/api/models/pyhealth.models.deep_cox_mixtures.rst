Deep Cox Mixtures
=================

Deep Cox Mixtures (DCM) is a survival regression model that
represents heterogeneous patient populations using a mixture
of component-specific Cox proportional hazards models.

The implementation includes:

* A shared neural network feature encoder.
* A probabilistic mixture gating network.
* Component-specific log hazard ratios.
* Breslow baseline cumulative hazard estimation.
* Shape-preserving cubic interpolation.
* Approximate expectation-maximization training.
* Patient-specific survival prediction.
* Checkpoint persistence for fitted baseline functions.

Training
--------

DCM requires a specialized training procedure through
``fit_em()`` rather than the standard PyHealth ``Trainer``.

The EM procedure alternates between estimating posterior
subgroup membership probabilities, optimizing the neural
network parameters, and updating component baseline hazards.

Example
-------

.. code-block:: python

    import torch
    from pyhealth.models import DeepCoxMixtures

    torch.manual_seed(42)

    features = torch.randn(100, 4)
    duration = torch.randint(1, 365, (100,)).float()
    event = torch.randint(0, 2, (100,)).float()

    model = DeepCoxMixtures(
        dataset=None,
        input_dim=4,
        num_components=3,
    )

    model.fit_em(
        features,
        duration,
        event,
        epochs=30,
    )

    survival = model.predict_survival(
        features[:5],
        horizons=[30, 90, 180],
    )

    print(survival.shape)
    # torch.Size([5, 3])

Model Outputs
-------------

The ``forward()`` method returns component log hazard ratios
and mixture membership probabilities.

The mixture probabilities returned under ``y_prob`` are
latent subgroup membership probabilities, not survival
probabilities.

Use ``predict_survival()`` to estimate patient-specific
survival probabilities at specified time horizons.

Limitations
-----------

* Training uses a dedicated approximate EM procedure.
* The standard PyHealth Trainer is not currently supported.
* Full-batch optimization preserves Cox risk sets but may
  limit scalability to very large datasets.
* PCHIP interpolation differs from the original paper's
  spline smoothing procedure.
* Survival predictions remain constant beyond the final
  observed component event time.

Reference
---------

Nagpal et al. (2021).
Deep Cox Mixtures for Survival Regression.
Proceedings of Machine Learning Research, 149.

https://proceedings.mlr.press/v149/nagpal21a.html

API Reference
-------------

.. automodule:: pyhealth.models.deep_cox_mixtures
   :members:
   :undoc-members:
   :show-inheritance: