GRAFT survival analysis
=======================

GRAFT separates survival ranking from post-training probability estimation.
The implementation accepts raw numerical features with separate duration
and event outcomes. It retains censored subjects during fitting/evaluation.

Fitting order
-------------

GRAFT consumes one dense numerical feature vector and separate ``duration``
and ``event`` numerical tensors. Event 1 means an observed event; 0 means
right-censoring. Raw scores increase with predicted survival time.

Use ``model.fit(train_samples, validation_samples)``, followed by
``model.fit_calibration(train_samples)``. The same ordered training cohort is
required for both calls. ``fit_outside_trainer=True`` means that the generic
Trainer's training loop is not the fitting interface. Its classification
metrics also do not apply. ``forward`` returns ``score`` and ``risk_score``;
loss requires explicit ``imputed_log_times`` of shape (batch, draws). Held-out
duration/event fields do not trigger imputation or loss.

Only training rows fit medians, scales, local KM and Cox calibration.
Validation outcomes select the earliest accepted best C-index checkpoint.
The ranking fit uses separate initialization/dropout, gate, permutation and
imputation RNG streams. Self is excluded from neighborhoods. Empty KM tails
use a reported 1.1-times-censoring fallback, or raise in strict mode.
Finite event masses are renormalized; unidentified residual conditional mass
is reported rather than modeled. Too few local events always raise.

``predict_survival(features, times)`` uses the frozen centered Cox coefficient
and Breslow cumulative hazard, with linear interpolation and no extrapolation
beyond training follow-up. Cox fitting introduces proportional hazards and
does not guarantee calibration. ``evaluate_survival`` computes raw-score
Harrell C-index and IPCW IBS on a supplied training-supported time grid.
IBS measures prediction error, not calibration alone. The censoring convention
is scikit-survival's reverse KM, evaluated right-continuously at ties.

``save``/``load`` and ordinary ``state_dict`` preserve preprocessing,
calibration and censor-survival state. Reload requires no training dataset,
outcome labels, torchsort, lifelines or local-KM caches for prediction. CPU
fitting is currently supported; CUDA fitting raises rather than claiming
unverified reproducibility. Fitting needs optional ``torchsort``, ``lifelines``
and ``scikit-survival`` dependencies. Imports of unrelated PyHealth models do
not import these dependencies through this module.

Method: Mohammad Ashhad, Robert Hoehndorf and Ricardo Henao, *GRAFT:
Decoupling Ranking and Calibration for Survival Analysis*, 2026 preprint,
https://arxiv.org/abs/2602.07884v2. Training-only preprocessing, target midranks,
explicit L2 and validation checkpoint selection differ from the original
experiment script. See ``examples/support2_survival_graft.py`` for a complete
synthetic SUPPORT2 loader/task/model example. Its demonstration metrics are
not benchmark reproduction results.

Usage and verification
----------------------

The optional package extra is ``pyhealth[graft]``. Torchsort must be compatible
with the installed PyTorch build; a matching wheel or a source build may be
needed. CPU fitting was verified with PyTorch 2.10 and torchsort 0.1.10.

Run the complete example from the repository root:

.. code-block:: console

   python examples/support2_survival_graft.py --output /tmp/graft-demo
   python -m unittest tests.core.test_graft tests.core.test_survival_time_support2 -v

The example generates fabricated SUPPORT2 rows by default and downloads
nothing. Its result JSON, checkpoint and reload inputs go in the explicit
output directory. ``--encoding benchmark14`` selects the separate 14-column
numeric-coded route; ``onehot22`` is default. ``--root`` explicitly selects
existing local data. Neither example split is the paper's evaluation protocol.

Study configurations and results
--------------------------------

The example accepts ``--variant full``, ``no_stg`` or ``linear`` and
``--noise-multiplier 3 --rho 0``, ``0.5`` or ``0.9``. Nuisance arrays share
seeded Gaussian base draws across correlation levels. All learned transforms
and local neighborhoods remain confined to training subjects.

``--benchmark-file /path/to/support.feather --fold 0 --seed 42`` explicitly
selects the verified processed SUPPORT cohort and one fixed CV study cell.
It checks the cohort/order checksum and uses the study's inner split, batch
size and stopping rule. This mode defaults to a 1,000-epoch cap and never
launches the full matrix. The synthetic/raw route defaults to eight epochs.

Installation, study commands, scalar results and limitations are included in
``examples/support2_survival_graft_results.md`` and its CSV. The study completed
102 physical fits. Clean full GRAFT had mean C-index 0.5895 versus DeepSurv
0.6097. Positive mean full/no-STG contrasts under correlated nuisance were
strongly seed dependent. Splitting, stopping and IBS intervals differ from
the author executable, so these are controlled partial-reproduction results.

Typical fitting order, after creating disjoint task sample subsets:

.. code-block:: python

   from pyhealth.models import GRAFT

   model = GRAFT(dataset=train_samples)
   model.fit(train_samples, validation_samples, seed=42)
   model.fit_calibration(train_samples)
   scores = model.predict_score(raw_features)
   curves = model.predict_survival(raw_features, training_supported_grid)
   model.save("graft.pt")
   restored = GRAFT.load("graft.pt")

Pass raw features to every call; the model applies its fitted training
transforms. Do not apply those transforms a second time. Fit the calibrator
on the identical ordered training rows. Prediction needs only features and
query times. For metrics, select support/grid from training data before
examining evaluation outcomes. When IBS support fails, preserve separately
valid raw-score C-index rather than changing the grid to improve results.

The synthetic verification checks execution and numerical conventions.
A short 14-column fit can have reversed raw-score ordering and a positive
Cox coefficient; the implementation emits a warning and retains raw-score
discrimination. It does not flip scores based on evaluation performance.
The ranking objective is separately tested against known orderings and
nonzero gradients. These checks do not establish benchmark accuracy.

Public API
----------

.. currentmodule:: pyhealth.models.graft

.. autoclass:: GRAFT
   :members: fit, fit_calibration, forward, predict_score, predict_survival, evaluate_survival, deterministic_gates, save, load
   :show-inheritance:

.. autoclass:: LocalKMImputer
   :members: fit, sample
   :show-inheritance:
