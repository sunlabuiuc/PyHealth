pyhealth.models.survival_dcm
============================

Overview
--------

Deep Cox Mixtures models right-censored survival with a mixture of Cox models.
A shared multilayer perceptron encodes each patient. A gate gives the
probability of belonging to each latent subgroup, and an expert gives one log
hazard ratio per subgroup. Each subgroup has its own non-parametric baseline
survival curve, so proportional hazards only has to hold within a subgroup and
the predicted survival curves of two patients can cross. The model follows
Nagpal, Yadlowsky, Rostamzadeh and Heller, `Deep Cox Mixtures for Survival
Regression <https://proceedings.mlr.press/v149/nagpal21a.html>`_ (MLHC 2021).
With ``num_components=1`` it is a single deep Cox model.

Inputs and outputs
------------------

Create a PyHealth sample dataset with ``features: tensor`` as input schema and
``duration: tensor, event: tensor`` as output schema, for example with
:class:`~pyhealth.tasks.SurvivalDCMSupport2`. Each sample has a dense feature
vector, one positive duration and one event indicator (1 for an observed event,
0 for censoring). The features must be finite, so standardize them and impute
missing values first.

``forward`` returns ``log_gate`` and ``log_hazard_ratio``, each of shape
``[batch, num_components]``, and the training ``loss`` when the labels are
given. In evaluation mode it also returns ``y_true``, with columns duration and
event, and ``y_prob``, the survival curve at every time in ``baseline_grid``.
Use ``predict_survival(features, times)`` for survival probabilities at chosen
times.

Training
--------

Training is the approximate EM procedure of the paper. Call ``fit_baselines``
once before training and again after every epoch, from your own training loop::

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.fit_baselines(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        model.fit_baselines(train_loader)

PyHealth's ``Trainer.train`` does not refit the baseline curves between epochs,
so do not use it on its own to train this model. ``Trainer.inference`` and
``Trainer.evaluate`` work on a fitted model.

Differences from the authors' implementation
--------------------------------------------

The baseline curves are refitted once per epoch, not every few batches. They
are smoothed with a cubic spline, as the paper describes, where the authors'
released PyTorch code uses a linear one. The optional variational auto-encoder
term is not included. The log hazard ratio is bounded with a scaled tanh, and
the default bound of 1 matches the authors' default.

API
---

.. autoclass:: pyhealth.models.SurvivalDCM
   :members:
   :show-inheritance:

.. autofunction:: pyhealth.models.survival_dcm.cox_partial_log_likelihood

.. autofunction:: pyhealth.models.survival_dcm.breslow_baseline_survival

.. autofunction:: pyhealth.models.survival_dcm.smooth_baseline_survival
