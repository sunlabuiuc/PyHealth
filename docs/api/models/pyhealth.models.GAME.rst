pyhealth.models.GAME
======================

Overview
--------

GAME (Han et al., "Representation learning to advance multi-institutional
studies with electronic health record data from US and France", `Nature
Communications 2025 <https://doi.org/10.1038/s41467-025-56837-1>`_) provides
globally aligned medical embeddings across multi-institutional EHR coding
systems.

Traditional electronic health record (EHR) studies across diverse hospitals
(such as MIMIC-IV and OMOP systems) encounter severe vocabulary heterogeneity:
distinct coding schemes (ICD-9, ICD-10, PheCode, LOINC, RxNorm, CCS, and local
institution-specific codes) create fragmented token spaces that do not align.
GAME resolves this by harmonizing clinical concepts into a shared embedding
space using biomedical knowledge graphs, language models, and multi-relational
graph alignment:

1. **Multi-Institutional Alignment**: Aligns clinical concepts across distinct
   vocabularies using positive pointwise mutual information (PPMI) and
   knowledge graph hierarchies.
2. **Harmonized Code Embeddings**: Produces unified representation vectors
   for standard and local clinical codes that capture both semantic similarity
   and medical relatedness.
3. **Drop-in Embedding Initializer**: With
   :func:`~pyhealth.models.init_embedding_with_game` and
   :class:`~pyhealth.models.GAMEEmbedding`, GAME embeddings can be directly
   loaded into any PyHealth model or embedding layer for downstream tasks on
   MIMIC-IV and OMOP.
4. **End-to-End Clinical Modeling**: :class:`~pyhealth.models.GAME` provides a
   complete PyHealth :class:`~pyhealth.models.BaseModel` integrating GAME
   embeddings with temporal sequence backbones (GRU, LSTM, Transformer) and
   configurable visit pooling for patient outcome prediction.

.. note::

   GAME embeddings can be loaded from precomputed dictionaries, PyTorch
   tensor files (``.pt`` or ``.pth``), NumPy archives (``.npy`` or ``.npz``),
   or text files. Codes not found in the pretrained GAME vectors retain their
   standard initialization, and the padding token (index 0) is explicitly
   preserved as zero.

API Reference
-------------

.. autofunction:: pyhealth.models.normalize_game_code

.. autofunction:: pyhealth.models.load_game_embeddings

.. autofunction:: pyhealth.models.init_embedding_with_game

.. autoclass:: pyhealth.models.GAMEEmbedding
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.GAME
    :members:
    :undoc-members:
    :show-inheritance:
