pyhealth.models.ProtoECGNetH1
=============================

Paired prototype-similarity consistency for the local morphology branch of
`ProtoECGNet (Sethi et al., MLHC 2025)
<https://proceedings.mlr.press/v298/sethi25a.html>`_. Contributor: Chia Yang Tsai
(UIUC NetID: ct68).

This model extends the existing :class:`pyhealth.models.ProtoECGNet` stage and
projection interfaces. It adds the reference-compatible 2D morphology arithmetic,
two-view objective, ordered data/noise helpers, and bank-linked source persistence.
The existing general ProtoECGNet defaults are unchanged. H2 stability-oriented
projection is outside this contribution.

Computation and configuration
-----------------------------

The reference-compatible research configuration is ResNet18 without pretrained
weights, 52 ordered Category-3 labels, 18 prototypes per label, and a bank of
936 vectors with native shape ``(936, 512, 3)``. The author's extra singleton axis
is handled explicitly during checkpoint import. Similarities use epsilon
``1e-4`` inside normalization and effective scale 13, followed by temporal
**top-5 mean** pooling and a bias-free classifier.

The paired objective is

.. math::

   L_{\mathrm{paired}} = \tfrac12\bigl[
       L_{\mathrm{orig}}(X,y)+L_{\mathrm{orig}}(\widetilde X,y)\bigr]
       + \lambda_{\mathrm{cons}}\operatorname{mean}
       \bigl[(s(X)-s(\widetilde X))^2\bigr].

Each view receives the complete original objective: positive-weighted BCE,
clustering, separation, sparsity, diversity and co-occurrence contrastive terms.
Averaging retains the original regularization coefficients. Both similarity
branches receive gradients; neither is detached. ``lambda_cons=0`` is still a
paired augmentation control and still requires the noisy view during training.
The evaluated H1 coefficient is 117608, calibrated from training gradients and
fixed thereafter; this integration does not recalibrate it from validation scores.

Input and training stages
-------------------------

The research pipeline uses already-preprocessed 100 Hz ECGs with shape
``(N, 12, 1000)`` or ``(N, 1, 12, 1000)`` and binary targets ``(N, 52)``. The
ordered label vocabulary is fixed explicitly, including labels absent in a
small subset. All-zero target rows remain in the dataset.

* **Joint:** one combined clean/noisy forward, BN training, dropout off,
  classifier frozen. ``Trainer.train`` may call ``model.train``; the stage policy
  is reapplied automatically. Class weights and co-occurrence come from training.
* **Projection:** standard training-only projection from each arm's own joint
  bank; require ascending unique ECG IDs. Exact ties keep the earliest ECG ID,
  then latent window. No stability-oriented source search is performed.
* **Classifier:** clean inputs, only classifier updated; extractor, add-on,
  prototypes and BN buffers remain fixed. Refit loss is weighted BCE plus
  off-class classifier L1. Native checkpoints persist the stage and objective.

Use ``model.eval()`` for clean evaluation. ``classification_loss`` is BCE alone;
``loss`` follows the current stage and is not always a pure classification loss.
``logit``, ``y_prob`` and ``similarity`` always describe the clean view. Paired
forwards additionally expose noisy predictions, similarities and loss components.

Noise and dataset integration
-----------------------------

``make_ecg_noisy_view`` adds Gaussian noise after existing preprocessing, with
per-ECG/per-lead scale ``0.01 * clean AC RMS``; zero-AC-RMS leads receive zero
noise. NumPy ``SeedSequence([stream_seed, ecg_id, epoch_or_draw])`` makes noise
independent of batch order. Both arms share the training stream; evaluation
requires a separate stream.

``create_h1_ptbxl_dataset`` adapts preprocessed arrays to native SampleDataset,
retaining ECG/patient IDs, folds, label order and all-zero targets. It does not
reprocess signals or fit statistics. ``PairedECGLoader`` advances the noise epoch
once per complete pass. Omit ``steps_per_epoch`` or set it to ``len(loader)``.
Partial passes, overlapping iterators and mid-epoch training resume are unsupported.
Validation uses a separate clean loader. Split declarations enforce folds 1-8,
9 and 10 for train, validation and test respectively.

The runnable example reuses native :class:`pyhealth.datasets.PTBXLDataset` for
real metadata/events. Its narrow morphology adapter filters the original WFDB
float64 samples before casting to float32, with no extra z-scoring, and retains
all-zero targets. The existing general multilabel task has different defaults
and skips unlabeled rows; its output is not a drop-in replacement for historical
checkpoint parity. No dataset or general task is reimplemented in the library.

Executable ablation example
---------------------------

Install from the repository, then run the default CPU synthetic example:

.. code-block:: bash

   pip install -e ".[ptbxl]"
   python examples/ptbxl_multilabel_classification_protoecgnet_h1.py \
       --output-dir /tmp/h1_synthetic_example

The output directory must be new. The example runs lambda 0 and 117608 from the
same initialization, sample-order seed, noise stream and fixed epoch budgets.
Fingerprints of the initial bank/classifier and actual consumed clean/noisy
batch streams must agree across both arms.
It writes a comparison JSON/Markdown table, validation predictions, source-bound
Trainer checkpoints, and verifies two noise epochs, stage freezing and strict
reload. The tiny demonstration uses two artificial labels and four prototypes;
its performance numbers are pipeline checks, not evidence of H1 effectiveness.

To explicitly start a NEW training example on a local PTB-XL v1.0.3 download:

.. code-block:: bash

   python examples/ptbxl_multilabel_classification_protoecgnet_h1.py \
       --root /data/ptb-xl/1.0.3 --train-limit 128 --validation-limit 64 \
       --joint-epochs 2 --refit-epochs 1 --output-dir /tmp/h1_ptbxl_example

Real mode uses 52 labels and 936 prototypes. Set both limits to zero to use all
training and fold-9 records. Subsets may leave some labels absent: report macro
AUROC/AP over labels with both positive and negative targets, along with per-label
support, eligible-label count and projection coverage. Missing projection sources
remain explicit. Both arms use the same subset. Fold-10 waveforms are excluded
before reading; there is no automatic held-out evaluation or checkpoint selection.
These short example budgets do not reproduce the historical training schedule.

Sources, contributions and checkpoints
--------------------------------------

``model.prototype_contributions(signal)["morphology"]`` returns signed
per-prototype terms. Model forward outputs separately expose pooled
``similarity`` and ``best_window``. The terms are classifier weights times
similarities. Positive and negative terms are both useful; signed terms
can cancel. Source coordinates are latent positions, not validated clinical
waveform intervals or diagnostic explanations.

``model.get_projection_provenance()`` returns a checked copy of ECG/patient/fold,
label eligibility, preprocessed-input fingerprints and latent source coordinates.
The bundle is bound to prototype values, assignment, feature weights/BN state,
architecture and label order. Classifier-only refit preserves the binding.
Feature/bank changes invalidate it; explicitly clear old provenance before
resuming joint updates, then project again. This integrity check does not prove
authentic historical execution or clinical interpretation validity.

Trainer ``save_ckpt`` / ``load_ckpt`` use native state dictionaries, including the
H1 metadata and source binding. They do not serialize optimizer/scheduler/loader
state for exact mid-training resume. Historical reference import is a separate,
strict operation on a trusted in-memory checkpoint payload whose expected file
hash has already been checked. It maps all active keys explicitly, documents
unused reference tensors, and never searches or replaces the saved bank. Saved
sources are attached only after every patch has been reconstructed and checked.

Numerical verification and research attribution
-----------------------------------------------

The selected historical models were trained with the custom Colab reference
pipeline, not PyHealth Trainer. On fixed training probes in the same CPU runtime,
reference/port features, similarities, top-5, logits, individual contributions,
losses and refit classifier gradients met the declared compatibility checks;
all 1,872 saved patches were reconstructed and native reload preserved behavior.
This is scoped compatibility evidence, not a full-cohort metric recomputation or
proof of all joint gradients for the selected models. Separate nondegenerate
synthetic tests cover both-branch gradients and the complete H1 objective.

The original FP32 contribution-sum/GEMM comparison failed. The reference's own
sum also fails that fixed bound under cancellation. The documented post-failure
v2 audit retains these failures, preserves model and research arithmetic, and
checks the algebraic reconstruction with high precision accumulation of existing
FP32 inputs/terms. It does not claim bitwise equivalence or raw FP32 sum/GEMM
conformance at the original bound. Never change scaling/normalization or formal
metric arithmetic to hide these residuals.

Recorded research findings (custom pipeline, one fixed training seed):

.. list-table:: Fixed held-out model pair
   :header-rows: 1

   * - Metric
     - Paired lambda 0
     - H1
   * - Clean macro AUROC
     - 0.689971
     - 0.707543
   * - Clean macro AP
     - 0.056895
     - 0.062018

These recorded findings are separate from the runnable example and the CPU port
audit. Not all labels/cases improved; no multi-seed, universal robustness or
clinical-effectiveness claim follows. The main 52-label patient bootstrap had
346/2000 valid resamples and no CI was reported under the locked rule. Five noise
draws are not five independent training runs. The research execution manifest
SHA256 is ``524958b9b1223db6f4ef098859118d3f0297b99dc4975a1fc4c9fbb964136d6b``.

API
---

.. autoclass:: pyhealth.models.ProtoECGNetH1
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.protoecgnet_h1.MorphologyPrototypeBranch
   :members:

.. autofunction:: pyhealth.models.protoecgnet_h1.paired_similarity_objective

.. automodule:: pyhealth.models.protoecgnet_h1_data
   :members: create_h1_ptbxl_dataset, make_ecg_noisy_view, PairedECGLoader

.. automodule:: pyhealth.models.protoecgnet_h1_reference
   :members: convert_reference_checkpoint, bind_reconstructed_reference_sources
