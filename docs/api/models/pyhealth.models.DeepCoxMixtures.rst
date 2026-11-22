pyhealth.models.DeepCoxMixtures
===============================

Overview
--------

Deep Cox Mixtures (Nagpal, Yadlowsky, Rostamzadeh and Heller, `Deep Cox
Mixtures for Survival Regression <https://proceedings.mlr.press/v149/nagpal21a.html>`_,
MLHC 2021) models right-censored survival as a mixture of K Cox models. A
shared ReLU6 encoder feeds a gating head (group probabilities) and an expert
head (one log hazard ratio per group); each group has its own smoothed Breslow
baseline survival curve.

Inputs and outputs
------------------

Use a sample dataset with ``features: tensor`` as input schema and
``time: tensor, event: tensor`` as output schema (dense, preprocessed
covariates; positive follow-up; 1 = death, 0 = censored).
:class:`~pyhealth.tasks.Support2SurvivalTask` produces raw SUPPORT2 covariates
and these labels; fit imputation and encoding on the training split.

After fitting, ``forward`` returns ``y_prob`` (survival at ``horizons``),
``logit`` (log mixture weights and log hazard ratios), ``y_true`` and a
``loss`` equal to the batch-mean negative M-step objective with the fitted
splines held fixed. ``mode`` is ``None``.

Training
--------

Fit with ``fit_survival`` (arrays) or ``fit`` (dataloaders); they run the
authors' EM routine, which refits the baseline splines between gradient steps.
``Trainer.train`` raises ``TypeError``; ``Trainer.evaluate``, ``inference``,
``save_ckpt`` and ``load_ckpt`` work after fitting. Training needs
``pip install 'pyhealth[dcm]'`` and the authors' ``auton-survival`` package
installed with ``--no-deps`` (see ``examples/support2_dcm/README.md``).
Construction, ``forward`` and ``predict_survival`` need only core
dependencies. CPU only.

Example and ablation: ``python examples/support2_survival_dcm.py --synthetic`` (generated
data; K x width x missing-value indicators). SUPPORT2 results:
``examples/support2_dcm/RESULTS.md``.

API
---

.. autoclass:: pyhealth.models.DeepCoxMixtures
    :members:
    :show-inheritance:
