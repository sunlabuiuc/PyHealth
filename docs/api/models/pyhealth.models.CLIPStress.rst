CLIPStress
==========

``CLIPStress`` implements the multimodal contrastive learning method described
in `Yang, Yu, and Sano (CHIL 2025)
<https://proceedings.mlr.press/v287/yang25a.html>`_. It inherits ``BaseModel``
and supports PyHealth datasets, dataloaders, and ``Trainer``.

Workflow
--------

1. Split by participant before fitting preprocessing. Use only training
   participants during pretraining, including their unlabeled windows.
2. Build feature-only descriptions with ``build_stress_prompt``. Never include
   the day's stress label, label-derived scores, or held-out information.
3. Encode descriptions once with ``encode_stress_prompts`` (frozen BERT,
   attention-mask-aware mean pooling), then cache the returned vectors.
   Overlength descriptions emit a warning with the affected count. Set
   ``truncation="error"`` to reject them before loading encoder weights.
   ``truncation="chunk"`` preserves all tokens using nonoverlapping windows
   and a token-weighted mean across them. Each window contributes its special
   tokens; padding is excluded. This changes the text context relative to the
   reference implementation's single truncated input and must be reported.
4. Fit sensor/tabular normalization using training samples only. Describe
   signals in original units and normalize numeric model inputs separately.
5. Run ``Trainer`` with ``stage="pretrain"``. Use batches of at least two paired
   samples. The example wraps the PyHealth loader to omit partial batches;
   the current ``get_dataloader`` API has no ``drop_last`` argument.
   With ``mode=None``, evaluation reports loss.
6. Call ``set_stage("linear_probe")`` or ``set_stage("finetune")`` and create a
   new Trainer. A linear probe freezes the encoders and disables their dropout.
   Both downstream modes use the ordinary classification output contract.

Input contract
--------------

Use fitted ``tensor`` processors for ``signal`` (time, channels), ``tabular``
(features), and ``text_embedding`` (text width). Sensor ``timeseries``
processors are also supported. One binary or multiclass label is optional
during model construction; a wholly unlabeled model can pretrain, then transfer
its feature extractor with ``downstream.load_pretrained_encoder(pretrained)``.
The destination model keeps its own classifier and label processors. Both
models must use the same predictor ordering and encoder architecture.
Transfer checks attention head counts as well as parameter names and shapes;
different head counts change attention even when parameter shapes match.

Fixed-length windows need no mask. For variable windows, include an explicit
boolean tensor field and set ``mask_key``. True means valid. Do not infer
padding from zeros: zero is a valid sensor observation. Impute missing observed
values before training; entirely masked windows are rejected.

Text vectors are required only during pretraining. Inference needs sensor and
survey predictors only, and works without labels. Setting ``signal_key=None``
or ``tabular_key=None`` enables modality ablations. A fresh model with
``stage="finetune"`` is a supervised baseline without contrastive pretraining.

Method fidelity
---------------

This is an independent implementation of the paper's method, with explicit
engineering choices. It does not reproduce the reference repository exactly:

* The paper describes cosine similarities and paired positives. We minimize
  the mean of the two cross-entropies on L2-normalized embeddings. The printed
  equation omits the leading negative sign; minimization requires it.
  The reference code instead uses unnormalized dot products and soft targets
  formed from within-modality similarities.
* Tables use residual fully connected blocks with LayerNorm. The reference
  repository uses a 1-D convolutional ResNet. These are different architectures.
* Sensor inputs are projected to a configurable hidden width and receive
  sinusoidal positions. The reference Transformer has no positional encoding.
* Text features use masked mean pooling in evaluation mode. Padding does not
  affect a description's vector. Long descriptions are truncated at the
  configured token limit; audit this before experiments with many fields.
* Classification uses the fused encoder representation rather than the
  contrastive projection, following the paper's feature-extractor description.

The model never downloads weights. The text preprocessing helper downloads a
language model only when explicitly called. Automated tests use synthetic
tensors and a mocked language encoder, with no network access.

Example and evaluation
----------------------

Run the offline example from the repository root::

    python examples/clip_stress.py --ablation all --output smoke_results.json

These are synthetic integration results, not paper metrics. For real data,
pass ``--data prepared.npz`` containing numeric ``signal`` (N,T,C), ``tabular``
(N,F), ``text_embedding`` (N,D), integer ``label`` (N; -1 means unlabeled), and
string ``patient_id`` (N). Arrays must be readable with ``allow_pickle=False``. The supervised baseline
(``--ablation supervised``) also accepts archives without text embeddings.
The example separates subjects, fits normalization on training data only,
selects checkpoints using validation AUROC, and evaluates the test split once
per fixed recipe. ``--label-fraction`` subsamples only training labels.
This subject-disjoint example is not the paper's random-day experiment.

The example exposes batch size, learning rate, encoder dimensions, layer/head
counts, dropout and temperature as command-line options, and records these
settings in its output JSON. Its defaults use a small model for quick checks.

For a reproduction, use the specified dataset processing, split protocol,
five folds and four seeds, 300 pretraining epochs and 150 downstream epochs;
report architectural/protocol differences and uncertainty. Do not use test
AUROC to choose epochs or hyperparameters. This contribution supplies a model,
not native LifeSnaps/PMData dataset adapters or a completed empirical replication.

API
---

.. autoclass:: pyhealth.models.clip_stress.CLIPStress
   :members:
   :show-inheritance:

.. automodule:: pyhealth.models.clip_stress_utils
   :members:
