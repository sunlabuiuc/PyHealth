CLIPStress
==========

``CLIPStress`` integrates the multimodal stress model from
`Yang, Yu, and Sano (CHIL 2025)
<https://proceedings.mlr.press/v287/yang25a.html>`_ with PyHealth's ``BaseModel``,
datasets, dataloaders and ``Trainer``. It implements the authors' Transformer,
convolutional tabular ResNet and residual projection heads independently.

Architecture and objectives
---------------------------

The input API accepts wearable arrays shaped ``(time, channels)``. Internally,
the fixed 24-hour window is transposed to ``(channels, time)``: the reference
Transformer attends across sensor channels, with model width 24, four heads,
three layers, feedforward width 64, ReLU and dropout 0.1. There is no positional
encoding or learned sensor input projection. Channel outputs are mean-pooled.

The tabular branch is a stride-one ResNet1D with width-three convolutions,
BatchNorm, ReLU and dropout 0.5. It starts with 128 filters, doubles them every
four residual blocks, and produces a 64-dimensional vector. Concatenated
sensor/table features are linearly fused to width 32. Both modalities use
128-dimensional residual GELU projection heads with dropout 0.5 and LayerNorm.
Downstream classification uses the sensor/table projection, ReLU and a linear
classifier. Binary tasks use two internal logits and cross-entropy; the returned
single logit is their difference, preserving PyHealth's binary output contract.

``objective="reference_soft"`` is the default. It matches the public code's
unnormalized dot products and differentiable soft targets derived from both
within-modality similarity matrices. Cross-modal logits divide by temperature;
the target similarities multiply by temperature. The targets are not detached.
``objective="paper_cosine"`` is an explicit alternative matching the paper's
paired cosine formulation, with the sign required to minimize the loss.
The printed equation and the public implementation are not identical; always
record which objective is used when reporting results.

Input and training stages
-------------------------

Use fitted ``tensor`` processors for ``signal`` (time, channels), ``tabular``
(features), and ``text_embedding`` (text width). Only fixed complete windows
are supported by the reference channel-attention architecture. An optional
mask must be all true. Zero is a valid measurement, not an inferred mask.
Inputs must be finite; impute observed missing values before training.

* ``pretrain`` aligns numeric predictors with cached frozen BERT vectors.
  Labels are not read. An unlabeled dataset with an empty output schema works.
* ``linear_probe`` freezes encoder, fusion and sensor projection parameters.
  As in the reference loop, training mode still enables dropout and updates
  BatchNorm buffers. Evaluation mode disables dropout and buffer updates.
* ``finetune`` trains the complete numeric encoder, projection and classifier.
  A newly initialized model in this stage is the supervised baseline.

Create a new Trainer after changing stages. ``load_pretrained_encoder`` copies
the numeric encoders, fusion and sensor projection into a model with its own
label schema and classifier. It checks parameter shapes and attention heads.
Text is not needed during classification or inference. Either numeric branch
can be disabled with ``signal_key=None`` or ``tabular_key=None``.

Text and preprocessing
----------------------

``build_reference_stress_prompt`` implements the LifeSnaps reference template,
excluding labels, as supported by the authors' experiment scripts. Statistics
are rounded to one decimal. ``statistics="reference"`` (default) uses the nine
statistics of the public code; ``statistics="paper"`` uses the eleven listed in
the paper, adding the 25th and 75th percentiles. The general-purpose
``build_stress_prompt`` helper also uses eleven statistics. The reference
template uses published personality categories and activity descriptions.

``encode_stress_prompts`` loads BERT only when explicitly called. Its default
``pooling="reference_mean"`` includes padding positions in the mean, matching
the author encoder. Text is truncated at 512 tokens with an audit warning.
Record encoding order and logical batch size: padding makes these relevant.
``micro_batch_size`` bounds device memory while preserving the logical batch's
padding. ``masked_mean`` and chunking are explicitly different alternatives;
chunking requires ``pooling="masked_mean"``.

For the authors' quantile-global LifeSnaps setup, numerical features are
normalized using training-fold quantiles. Each hour of each sensor channel is
a separate quantile-transform column. Only the four heart-rate-zone duration
fields in the 21-dimensional table receive numerical normalization; remaining
ordinal categorical fields retain their codes. Text must be regenerated from
the normalized sensor windows separately for each fold. A single globally
cached text archive cannot represent this protocol.

Runnable comparisons
--------------------

These commands run small offline synthetic demonstrations::

    python examples/clip_stress.py --ablation all --output smoke.json
    python examples/lifesnaps_stress_clip_stress.py --output-dir runs/demo
    python examples/support2_mortality_clip_stress.py --output-dir runs/support-demo

Synthetic outputs verify execution, not performance on the paper's task.
The LifeSnaps example follows the setting of the paper's Table 1: random
splitting with generalized models, comparing the supervised multimodal network
with contrastive pretraining followed by end-to-end fine-tuning. With
prepared fold archives::

    python examples/lifesnaps_stress_clip_stress.py \
        --fold-data-dir lifesnaps-paper-folds --output-dir runs/lifesnaps \
        --paper-budget --device cuda:0 --threads 2

The directory must contain ``fold-1.npz`` through ``fold-5.npz``, with numeric
``signal``, ``tabular``, ``text_embedding``, ``label`` (-1 for unlabeled),
``patient_id``, and JSON ``metadata`` identifying the fold's preprocessing and
test rows. The example checks those identities before training. The default
``paper_random`` protocol reproduces the public code's stratified 5-fold split
over all windows (shuffled, random state 42, stratified by stressed, relaxed
and unlabeled), so a participant can contribute windows to both training and
test folds. Pretraining uses every training window with its label removed;
fine-tuning uses all labeled training windows. ``reference_user`` (the
participant-disjoint split of the paper's Table 2) remains available.

``--paper-budget`` applies the paper's Section 3.2.3: five folds and four
random seeds (0-3); 300 supervised epochs; 300 pretraining and 150 fine-tuning
epochs; Adam at 0.001; batch size 512; a 3-layer, 4-head Transformer; and the
printed cosine objective (``paper_cosine``). The paper does not state the
temperature; ``--temperature`` defaults to the public code's 0.07. Fold
archives for this setting use the paper's eleven prompt statistics
(``build_reference_stress_prompt(..., statistics="paper")``).
``--workers N --worker-index I`` divides independent runs across devices.
``--summarize-only`` validates completeness and reports the mean and standard
deviation over the 20 fold/seed runs. Use identical scientific options when
resuming or summarizing a run.

Evaluation boundaries
---------------------

The paper optimizes hyperparameters separately for each fold and seed but does
not specify the search space. The example instead uses one fixed configuration,
fixed final checkpoints, and evaluates each test fold once. It does not adopt
the public training script's use of test metrics for checkpoint selection. Standard AUROC is computed from probabilities;
``reference_argmax_auc`` is additionally recorded for comparison with the public
script's hard-label score, but is not substituted for AUROC. Average precision,
F1 and accuracy are also recorded; thresholded metrics use 0.5. Fold standard
deviations are descriptive, not confidence intervals or seed-robustness claims.
The optional validated-user protocol reserves a separate validation subject
fold and is a different evaluation setting.

LifeSnaps results
-----------------

The command above (``paper_random`` protocol, ``--paper-budget``, temperature
0.07) produced the following test results: 20 runs per method (five folds,
seeds 0-3), mean and sample standard deviation, in percent. Each run used
the final checkpoint and evaluated its test fold once.

.. list-table::
   :header-rows: 1

   * - Method
     - AUROC
     - Hard-label AUC
     - Average precision
     - F1
     - Accuracy
   * - Supervised (multimodal)
     - 58.10 ± 7.63
     - 58.68 ± 5.12
     - 47.74 ± 8.58
     - 47.08 ± 5.93
     - 61.85 ± 5.72
   * - Contrastive pretraining + fine-tuning
     - 66.25 ± 9.12
     - 63.68 ± 7.23
     - 55.94 ± 10.96
     - 52.03 ± 9.70
     - 67.77 ± 7.16

Pretraining raised AUROC in 14 of the 20 paired fold/seed runs, by 8.15
points on average. The runs share data, so these figures are descriptive.
The 40 runs took 6.6 minutes on four GPUs.

The paper reports 67.18 ± 7.65 (supervised) and 72.45 ± 4.70 (contrastive)
on LifeSnaps. The authors' public results table (``lifesnaps.csv`` and the
aggregation notebook ``results.ipynb`` in their repository) reproduces these
values as follows. For each fold and seed, the supervised value is the highest
test AUC over the 300 training epochs. The contrastive value is the highest
test AUC over epochs, over the zero-shot, linear-probe and fine-tuning stages,
and over six runs: 10, 50 or 100 percent of the unlabeled data, each with BERT
prompt embeddings or with random embeddings (``exclude=all``). AUC is computed
from hard predicted labels, the recorded learning rate is 0.0001, and the
reported deviation is the population standard deviation. For the configuration
the paper describes (label-free prompts, all unlabeled data, end-to-end
fine-tuning), the same table gives 67.33 ± 7.13 for the contrastive model,
still using the best test epoch. Because this example selects no epoch or
configuration on test data, its values are not directly comparable to the
published ones.

SUPPORT2 extension
------------------

The SUPPORT2 example, an extension beyond the paper, uses PyHealth's native
``Support2Dataset`` and only six explicit predictor fields (age, sex,
comorbidity count, diabetes, dementia, cancer status) to predict in-hospital
death. Its tabular-only comparison varies the ResNet and projection dropout
together. With ``--csv support2.csv --epochs 50 --limit 1000 --seed 0``, the
1,000 patients with the lowest ``sno`` identifiers are randomly split
600/200/200 by patient (188, 66 and 55 deaths);
the epoch is chosen on the validation split and the test split is evaluated
once:

.. list-table::
   :header-rows: 1

   * - Dropout
     - AUROC
     - Average precision
     - F1
     - Accuracy
   * - 0
     - 65.72
     - 45.10
     - 30.77
     - 73.00
   * - 0.1
     - 64.58
     - 40.75
     - 35.79
     - 69.50
   * - 0.3
     - 67.00
     - 42.77
     - 36.59
     - 74.00

With one split, one seed and 200 test patients, these differences do not
establish an effect of dropout. The example demonstrates reuse of the model on
a native PyHealth dataset, not stress detection or multimodal pretraining.
Model alignment alone does not establish reproduction of the paper's PMData
experiments.

API
---

.. autoclass:: pyhealth.models.clip_stress.CLIPStress
   :members:
   :show-inheritance:

.. automodule:: pyhealth.models.clip_stress_utils
   :members:
