pyhealth.models.DescEmb
=======================

Overview
--------

DescEmb (Yoon et al., `Description-based Medical Code Embeddings for
Electronic Health Records <https://arxiv.org/abs/2208.06648>`_, Machine
Learning for Healthcare / MLHC 2022) embeds medical concepts directly from
their standardized clinical text descriptions.

Traditional EHR architectures assign arbitrary integer indices to medical
codes (such as ICD-9, ICD-10, or OMOP), creating models that cannot
generalize or transfer across distinct hospital datasets or coding systems.
DescEmb resolves this by representing codes via their clinical text descriptions
(e.g., "Essential (primary) hypertension"):

1. **Description Text Encoding**: Standardized textual descriptions are
   encoded using word-level tokenization and a bidirectional recurrent neural
   network (BiGRU) via :class:`~pyhealth.models.descemb.DescEmbCodeEncoder`.
   The endpoints are pooled to construct semantically grounded code
   representations.
2. **Feature Embedding & Intra-Visit Pooling**: Input features are
   embedded with :class:`~pyhealth.models.EmbeddingModel`. Nested visit
   sequences are pooled across events using configurable intra-visit
   pooling (``sum``, ``mean``, ``max``, or ``attention``).
3. **Description-Based Embedding Initialization**: By initializing code
   embeddings from textual descriptions via
   :meth:`~pyhealth.models.DescEmb.initialize_with_descriptions`, concept
   representations are initialized from description tokens. When paired with
   a user-supplied ``pretrained_text_encoder`` and matching ``text_vocab``,
   this injects learned clinical text semantics into the code embedding layer
   prior to task fine-tuning; when using the default un-pretrained encoder,
   initialization acts as a pseudo-random projection.
4. **Longitudinal Visit Modeling**: A temporal sequence model (GRU, LSTM,
   or Transformer) aggregates patient trajectories across visits.
5. **Outcome Prediction**: Last-visit patient representations are mapped
   to outcome predictions across binary, multiclass, and multilabel tasks.

.. note::

   When initialized without pretrained encoder weights,
   :class:`~pyhealth.models.descemb.DescEmbCodeEncoder` acts as a pseudo-random
   projection (dependent on the initialization seed) mapping description tokens
   to embedding space. Learned clinical semantics are only introduced when
   users provide a custom module via ``pretrained_text_encoder`` and matching
   vocabulary via ``text_vocab`` to retain semantics learned during text
   pretraining. Codes omitted from ``code_descriptions`` fall back to using
   their raw code strings.

Inputs and Outputs
------------------

DescEmb accepts standard PyHealth :class:`~pyhealth.datasets.SampleDataset`
batches. Both 2D sequences (``SequenceProcessor``) and 3D nested sequences
(``NestedSequenceProcessor``) are supported. Binary, multiclass, and
multilabel classification tasks are fully supported.

API
---

.. autoclass:: pyhealth.models.DescEmb
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.descemb.DescEmbCodeEncoder
   :members:
   :show-inheritance:

.. autofunction:: pyhealth.models.descemb.simple_word_tokenize

.. autofunction:: pyhealth.models.descemb.simple_subword_tokenize
