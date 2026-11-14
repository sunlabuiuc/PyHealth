pyhealth.models.CET
===================

Overview
--------

CET implements the conditional event-time framework of `Engelhard et al.
(MLHC 2020) <https://proceedings.mlr.press/v126/engelhard20a.html>`_. It
learns a probability of eventual occurrence and a log-normal event-time model
conditioned on sampled occurrence indicators. Its censored objective follows
the reference TensorFlow MIMIC implementation.

Inputs and labels
-----------------

Create a PyHealth ``SampleDataset`` containing a dense numeric ``features``
vector and ``duration`` and ``event`` tensor outputs. Each sample's duration
and event fields have **one entry per target event**, and every duration must
be strictly positive. Event ``1`` means the event occurred before censoring;
``0`` means the record is right-censored, not a definitive negative label.

``CET(dataset, hidden_dim=750, n_samples=100, temperature=0.3)`` returns the
training ``loss``, raw occurrence ``logit``, occurrence ``y_prob``, observed
indicator ``y_true``, and event-time ``time_pred``. For a one-target task,
provide lists of length one. The model infers input and target dimensions from
the processed dataset. ``mode=None`` is intentional: generic classification
metrics cannot evaluate eventual occurrence using censored labels.

Prediction and limitations
--------------------------

``predict_event_time(features)`` returns the median across sampled log-normal
conditional medians. ``posterior_occurrence(features, duration, event)``
returns an approximate per-event posterior for soft-label experiments. Use
separate hidden occurrence labels if artificial censoring supplies ground
truth for occurrence AUC. Exact reproduction of MIMIC results additionally
requires the authors' unavailable feature/item selection and cohort.

The optional ``logvar_bounds`` safeguard is a deliberate numerical-stability
difference from the reference implementation; set it to ``None`` to disable.

Example and ablation
--------------------

From the repository root::

   python examples/synthetic_event_time_cet.py --n 2000 --epochs 60 --seed 7 --output cet_ablation_seed7.json

Three paired settings compare 8 vs. 32 Monte Carlo occurrence samples and
Gumbel temperatures 0.3 vs. 1.0. Synthetic data is generated locally. To run
on the existing PyHealth SUPPORT2 data loader, supply ``--source support2``
and the path to the full, local SUPPORT2 CSV (see
``examples/cet_results/README.md``). The example generates temporary row IDs
when the public CSV lacks a patient ID; it does not modify the source file.
The SUPPORT2 route evaluates censored likelihood, not eventual-occurrence
AUC, because that outcome is not observed in real clinical data.

API
---

.. autoclass:: pyhealth.models.CET
   :members:
   :show-inheritance:
