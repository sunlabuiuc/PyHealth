pyhealth.models.MENSA
=====================

Overview
--------

MENSA (Multi-Event Network for Survival Analysis) predicts a survival curve
for each of several events from one shared network. Each event, plus an
event-free state used only in training, gets a mixture of Weibull
distributions and its own residual adapter on a shared trunk. Training
minimizes ``(1 - lambda) * L_surv + lambda * L_traj``: a weighted censored
likelihood over all states, plus a trajectory term that penalizes predicting
the later event (death) before the earlier one (recurrence) for patients who
had both. ``trajectory_variant="paper"`` follows the paper's Eq. 1;
``"reference"`` follows the authors' code.

Paper
-----

Lillelund, Gharari Foomani, Sun, Qi and Greiner, `MENSA: A Multi-Event Network
for Survival Analysis with Trajectory-based Likelihood Estimation`_, Machine
Learning for Health (ML4H) 2025, PMLR 297 (`arXiv:2409.06525`_). This
implementation is adapted from the authors' `reference code`_ (MIT License,
commit 6b219cd).

Inputs and outputs
------------------

MENSA works with the ``"tensor"`` outputs of
:doc:`../tasks/pyhealth.tasks.multi_event_survival_rotterdam`: ``features
[F]``, ``event_times [K]`` and ``event_indicators [K]``, in source time units.
It has no label processor, so ``mode`` is ``None`` and PyHealth's Trainer
tracks only the loss; compute survival metrics from survival curves.

Call ``fit_preprocessing`` with the **training split only** before training.
It fits the feature standardization, the per-state loss weights and the time
grid used for ``y_prob``, and stores them as buffers, so they are saved with
the model. ``forward`` returns ``loss``, ``y_prob`` (survival probabilities,
``[batch, K, T]`` on that grid) and ``y_true`` (times and indicators).
``predict_survival(features, times)`` returns ``[batch, K, len(times)]``
curves on any time grid; it runs in evaluation mode without gradients and
restores each module's previous mode.

Defaults are one hidden layer of 32, dropout 0.25, three Weibull components
and ``lambda = 0.25``, trained with batch size 32, learning rate 1e-3, weight
decay 1e-3 and gradient clipping at norm 1.0. The training loader drops its
last incomplete batch (``drop_last=True``), because BatchNorm cannot train on
a batch of one. The reference code's Rotterdam config uses
``hidden_dims=(16,)``, ``dropout=0.5`` and weight decay 1e-4 instead.

Usage
-----

.. code-block:: python

    import torch
    from pyhealth.datasets import RotterdamDataset, get_dataloader, split_by_patient
    from pyhealth.models import MENSA
    from pyhealth.trainer import Trainer

    samples = RotterdamDataset(root="/path/to/folder").set_task()
    train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=0)

    model = MENSA(samples)
    model.fit_preprocessing(train)  # training split only
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=get_dataloader(
            train, batch_size=32, shuffle=True, drop_last=True
        ),
        val_dataloader=get_dataloader(val, batch_size=1024),
        epochs=1000,
        optimizer_params={"lr": 1e-3},
        weight_decay=1e-3,
        max_grad_norm=1.0,
        monitor="loss",
        monitor_criterion="min",
        patience=20,
    )

    batch = next(iter(get_dataloader(test, batch_size=len(test))))
    times = torch.linspace(0, 7000, 50)  # days
    curves = model.predict_survival(batch["features"], times)  # [N, 2, 50]

API reference
-------------

.. autoclass:: pyhealth.models.MENSA
    :members:
    :show-inheritance:

Loss and distribution functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stateless tensor functions used by :class:`~pyhealth.models.MENSA`.

.. autofunction:: pyhealth.models.mensa.mensa_loss
.. autofunction:: pyhealth.models.mensa.survival_nll
.. autofunction:: pyhealth.models.mensa.trajectory_nll
.. autofunction:: pyhealth.models.mensa.extend_labels
.. autofunction:: pyhealth.models.mensa.compute_state_weights
.. autofunction:: pyhealth.models.mensa.mixture_log_weights
.. autofunction:: pyhealth.models.mensa.mixture_log_survival
.. autofunction:: pyhealth.models.mensa.mixture_log_density
.. autofunction:: pyhealth.models.mensa.weibull_log_survival
.. autofunction:: pyhealth.models.mensa.weibull_log_density

.. _MENSA\: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation: https://proceedings.mlr.press/v297/lillelund26a.html
.. _arXiv:2409.06525: https://arxiv.org/abs/2409.06525
.. _reference code: https://github.com/thecml/mensa
