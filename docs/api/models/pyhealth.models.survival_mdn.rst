pyhealth.models.SurvivalMDN
==========================

Overview
--------

Survival MDN predicts a covariate-dependent Gaussian mixture in latent time and
maps it to positive event time. The right-censored likelihood combines observed
event densities with survival probabilities for censored records. Event terms
include the inverse-transform Jacobian. The default softplus mapping follows
Han, Goldstein and Ranganath, `Survival Mixture Density Networks
<https://proceedings.mlr.press/v182/han22a.html>`_ (MLHC 2022).
The exponential mapping supports a controlled ablation.

Inputs and outputs
------------------

Create a PyHealth sample dataset with ``features: tensor`` as input schema and
``duration: tensor, event: tensor`` as output schema. Each sample contains a
dense feature vector, one strictly positive duration and one event indicator
(one for observed event, zero for censoring). Durations and evaluation grids
must use the same units; the SUPPORT example uses years plus 0.001 years.

``SurvivalMDN(dataset, hidden_dim=32, num_components=10)`` returns mean censored
NLL in ``loss``, labels in ``y_true`` with columns duration and event, and survival
curves in ``y_prob`` with shape ``[batch, len(time_grid)]``. ``logit`` concatenates
raw weight logits, means and raw scale logits; it is not a classification score.
Mixture parameters are also returned. Supply a time grid appropriate to your
dataset rather than assuming the default SUPPORT-oriented grid is universal.

Use ``model.eval()`` and ``torch.no_grad()`` for ``predict_survival``. The method
preserves the caller's training/evaluation mode. BatchNorm requires at least two
samples per training minibatch. ``mode=None`` is intentional: use the custom
survival loop and metrics, not PyHealth's classification metric dispatcher.

Example and ablation
--------------------

From a PyHealth checkout with this contribution installed::

    python examples/support_survival_survival_mdn.py --epochs 2 --vary-hidden

This uses synthetic data. For the existing dataset route, pass
``--source support2 --data /path/support2.csv``. An explicit ``sno`` or
``patient_id`` column is required. The benchmark route uses the verified encoded
table and exact membership CSV. Both routes fit scaling on training data only.

The example compares softplus and exponential mappings under paired settings;
``--vary-hidden`` additionally compares widths 32 and 64. Setup, historical
results, limitations and complete commands are in
``examples/survival_mdn_results/README.md``. The historical experiments used the
standalone core; companion validation documents native PyHealth checks.

API
---

.. autoclass:: pyhealth.models.SurvivalMDN
   :members:
   :show-inheritance:
