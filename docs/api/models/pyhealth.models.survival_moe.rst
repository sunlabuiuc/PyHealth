Survival MoE
============

The discrete-time Survival Mixture-of-Experts model predicts an event-time
probability mass function while accounting for right-censored observations.
It supports personalized experts, fixed experts, and an MTLR baseline head.

The PyHealth model interface expects one dense tensor feature and two labels:

* ``duration_bin``: integer index on a time grid fitted using training data.
* ``event``: 1 when the event is observed, 0 when right-censored.

Pass the training-fitted grid using ``time_grid=...`` so the model retains the
bin values. The forward output includes ``y_prob`` (survival probabilities by
bin), ``routing``, and, when labels are supplied, ``y_true`` with the duration
bin and event indicator in its two columns.

Example
-------

.. code-block:: python

    from pyhealth.datasets import create_sample_dataset
    from pyhealth.models.survival_moe import SurvivalMoEModel

    model = SurvivalMoEModel(
        dataset=sample_dataset,
        input_dim=feature_dim,
        num_time_bins=100,
        hidden_dim=128,
        num_experts=8,
        expert_type="personalized",
        time_grid=train_time_grid,
    )

Training uses the standard PyHealth ``Trainer``. Since survival output is not
a classification mode, ``model.mode`` is ``None`` and the trainer optimizes
the loss returned by the model. Survival-specific C-index, IPCW Brier score,
and calibration metrics should be computed from the returned ``survival``
curves using held-out data.

Relationship to PersonalizedSurvivalMoE
---------------------------------------

PyHealth also provides
:class:`~pyhealth.models.personalized_survival_moe.PersonalizedSurvivalMoE`,
which implements the paper's personalized expert architecture and supports
categorical embeddings and dropout. This module is a complementary variant
for controlled comparisons: it exposes personalized, fixed-distribution, and
MTLR heads through one interface, plus optional top-k routing. It does not
replace ``PersonalizedSurvivalMoE`` or claim to outperform it. As planned in
the proposal, it does not implement the paper's Adjustable MoE head.

For the SUPPORT2 proposal ablations, see ``examples/support2_survival_moe.py``.
It implements A1 (load-balancing weights), A2 (time-grid resolutions with a
fixed expert chunk size), and A3 (dense, top-1, and top-2 routing), and reports
per-seed metrics, routing entropy/utilization, and pairwise routing ARI. The
script also supports generated synthetic data for a fast smoke test. From the
project root, run:

.. code-block:: console

    python -m examples.support2_survival_moe --csv data/support2/support2.csv

For a short synthetic smoke test, run:

.. code-block:: console

    python -m examples.support2_survival_moe --synthetic --seeds 11 --max-epochs 1

The project pins the compatible scikit-survival/OSQP versions in
``requirements-survival.txt``. Install them in the PyHealth environment with:

.. code-block:: console

    python -m pip install -r requirements-survival.txt

The full run writes ``ablation_by_seed.csv``, ``ablation_summary.csv``,
``ablation_assignments.csv``, and ``ablation_routing_stability.csv`` under
``outputs/support2_ablations``.

Local SUPPORT2 baseline comparison
----------------------------------

The five-seed baseline run uses the same 80/10/10 stratified splits and
training-only preprocessing. It ran on CPU with PyTorch 2.7.1+cpu; the machine
has an RTX 4090, but that CPU-only run did not use it. Values are mean
(standard deviation); C-index is on the 0--100 scale.

.. list-table::
    :header-rows: 1

    * - Model
      - C-index
      - ECE
      - Brier@Q50
    * - Personalized MoE
      - 72.72 (1.14)
      - 0.0365 (0.0028)
      - 0.1867 (0.0039)
    * - Fixed MoE
      - 70.36 (1.77)
      - 0.0553 (0.0111)
      - 0.1993 (0.0096)
    * - MTLR
      - 71.83 (1.43)
      - 0.0422 (0.0091)
      - 0.1891 (0.0063)
    * - CoxPH
      - 72.94 (1.05)
      - 0.0449 (0.0067)
      - 0.1877 (0.0046)
    * - Random survival forest
      - 71.14 (0.80)
      - 0.0553 (0.0062)
      - 0.1929 (0.0027)

Compared with MTLR, the personalized mode improved mean C-index by 0.88
points, ECE by 0.0057, and Brier@Q50 by 0.0024; the five-seed routing ARI was
0.395 (0.111) across the ten seed pairs. These directions are consistent with
the proposal's H1/H4 hypotheses. However, the proposal reports paper values of
80.84 versus 79.91 for C-index, 0.048 versus 0.057 for ECE, and 0.142 versus
0.149 for median-time Brier (Personalized MoE versus MTLR). The local run does
not match those absolute C-index and Brier values, so it should be described as
a partial reproduction rather than an exact numerical reproduction until the
evaluation setup is reconciled.

One known protocol difference is the feature set: the local helper excludes
post-outcome and post-entry columns to reduce leakage, whereas the paper
reference example uses its reported 42-feature setup and retains ``dnr`` and
``dnrday``. This is intentional for the local leakage-safe experiment, but it
means the scores are not an apples-to-apples numerical reproduction.

Local SUPPORT2 ablation results
-------------------------------

The following exploratory results summarize a completed five-seed run
(``11, 22, 33, 44, 55``) on an NVIDIA RTX 4090 with PyTorch 2.14.1+cu130,
CUDA 13.0, and ``device=cuda``. Data splitting is stratified by event status
into 80% training, 10% validation, and 10% test sets; preprocessing and the
time grid are fitted on the training split only. All A1--A3 runs use a
one-layer, 128-wide backbone, eight personalized experts, and an Adam learning
rate of 5e-4; A2 keeps the expert chunk size fixed at 16. Training monitors
validation negative log-likelihood with patience 10. The run used
``ABLATION_SMOKE_TEST=False``, a 200-epoch cap, and batch size 64. Each value
is mean (standard deviation) across seeds. C-index is reported on the 0--100
scale; ECE and Brier scores are on the 0--1 scale, and lower is better for ECE
and Brier scores. ``Brier@Q50`` is evaluated at the median training-duration
quartile.

* **A1 -- load balancing:** For load-balance weights 0, 0.001, 0.01, and 0.1,
  C-index was 72.64 (1.18), 72.65 (1.17), 72.72 (1.14), and 72.76 (1.14);
  ECE was 0.0367 (0.0036), 0.0365 (0.0038), 0.0365 (0.0028), and
  0.0364 (0.0031); Brier@Q50 was 0.1870 (0.0039), 0.1870 (0.0039),
  0.1867 (0.0039), and 0.1863 (0.0039). From weight 0 to 0.1, mean routing
  entropy increased from 1.819 to 2.075 (the maximum for eight experts is
  ``ln(8)``), and mean nonempty experts increased from 6.8 to 8.0.
* **A2 -- time-grid size:** For 25, 50, 100, and 200 bins, C-index was
  72.69 (1.02), 72.64 (1.12), 72.72 (1.14), and 72.72 (1.19); ECE was
  0.0357 (0.0034), 0.0363 (0.0013), 0.0365 (0.0028), and 0.0381 (0.0045);
  Brier@Q50 was 0.1862 (0.0045), 0.1871 (0.0050), 0.1867 (0.0039), and
  0.1890 (0.0042). More bins did not improve these metrics consistently.
* **A3 -- routing sparsity:** Dense, top-1, and top-2 routing produced
  C-indices of 72.72 (1.14), 72.16 (0.89), and 72.06 (1.18); ECE values of
  0.0365 (0.0028), 0.0391 (0.0034), and 0.0391 (0.0015); and Brier@Q50
  values of 0.1867 (0.0039), 0.1919 (0.0050), and 0.1886 (0.0054). Dense
  routing had the best predictive scores in this run.

For the default ``lambda_lb=0.01`` and dense routing, the mean pairwise
routing ARI across the ten seed pairs was 0.395 (0.111). These are descriptive
results from one local experiment, not evidence of statistical significance or
a comparison against the original paper's reported results. The experiment ran
all three studies with seeds ``11, 22, 33, 44, 55``. Do not commit the
SUPPORT2 source CSV or patient-level assignment outputs.

These are descriptive results from one local experiment, not evidence of
statistical significance or a comparison against the original paper's reported
results. The experiment ran all three studies with seeds ``11, 22, 33, 44, 55``.
For strict reproducibility, preserve the notebook configuration alongside the
results. Do not commit the SUPPORT2 source CSV or patient-level assignment
outputs.

.. autoclass:: pyhealth.models.survival_moe.SurvivalMoEModel
    :members:
    :undoc-members:
