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
Old residual-MLP checkpoints are not compatible with this architecture.

Text and preprocessing
----------------------

``build_reference_stress_prompt`` implements the LifeSnaps reference template,
excluding labels, as supported by the authors' experiment scripts. It uses
nine statistics rounded to one decimal; the paper describes eleven, which the
separate general-purpose ``build_stress_prompt`` helper supports. The reference
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
The LifeSnaps comparison varies contrastive temperature (0.03, 0.07, 0.2)
against a supervised baseline. With reference-prepared fold archives::

    python examples/lifesnaps_stress_clip_stress.py \
        --fold-data-dir lifesnaps-reference-folds --output-dir runs/lifesnaps \
        --paper-budget --seed 0 --device cuda:0 --threads 2

The directory must contain ``fold-1.npz`` through ``fold-5.npz``, with numeric
``signal``, ``tabular``, ``text_embedding``, ``label`` (-1 for unlabeled),
``patient_id``, and JSON ``metadata`` identifying the fold's preprocessing and
subjects. The example checks those identities before training. Each outer fold
uses four subject groups for training and one for testing, matching the public
non-random user split. The reference profile also reproduces the author's
sampling of unlabeled training windows with replacement, while retaining all
labeled training windows. Labels are removed for the pretraining objective.

``--paper-budget`` selects 300 pretraining epochs, 150 downstream epochs,
300 supervised epochs, batch size 512, Adam at 0.001, and reference dimensions.
This is deliberately single-seed when ``--seed 0`` is used; the paper reports
four seeds and fold-specific tuning. ``--workers N --worker-index I`` divides
independent runs across devices. ``--summarize-only`` validates completeness.
Use identical scientific options when resuming or summarizing a run.

Evaluation boundaries
---------------------

The reference-user example uses fixed final checkpoints and evaluates each
test fold once. It does not adopt the public training script's use of test
metrics for checkpoint selection. Standard AUROC is computed from probabilities;
``reference_argmax_auc`` is additionally recorded for comparison with the public
script's hard-label score, but is not substituted for AUROC. Average precision,
F1 and accuracy are also recorded; thresholded metrics use 0.5. Fold standard
deviations are descriptive, not confidence intervals or seed-robustness claims.
The optional validated-user protocol reserves a separate validation subject
fold and is a different evaluation setting.

The SUPPORT2 example uses the existing native dataset and only six explicit
predictor fields. Its tabular-only dropout comparison varies the ResNet and
projection dropout together. This demonstrates reuse, not stress detection or
multimodal pretraining. Model alignment alone does not establish reproduction
of the paper's tuned, multi-seed results or its PMData experiments.

API
---

.. autoclass:: pyhealth.models.clip_stress.CLIPStress
   :members:
   :show-inheritance:

.. automodule:: pyhealth.models.clip_stress_utils
   :members:
