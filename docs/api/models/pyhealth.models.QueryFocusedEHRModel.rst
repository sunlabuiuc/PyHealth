
pyhealth.models.QueryFocusedEHRModel
====================================

QueryFocusedEHRModel: a diagnosis-conditioned attention model for
query-focused extractive summarization of longitudinal electronic
health records (EHRs), inspired by
`McInerney et al., MLHC 2020 <https://proceedings.mlr.press/v126/mcinerney20a.html>`_.

The model uses pretrained ClinicalBERT to encode historical clinical
sentences and diagnosis descriptions. Learned projections and
query-conditioned dot-product attention identify sentences that
contribute to downstream diagnosis prediction. Attention weights
are also used to rank sentences for extractive summarization.

The model supports frozen ClinicalBERT embeddings with optional
in-memory LRU caching, as well as end-to-end encoder fine-tuning.
Caching reduces repeated text encoding when ``freeze_encoder=True``.
Setting ``cache_embeddings=True`` with ``freeze_encoder=False``
raises a ``ValueError`` because caching requires a frozen encoder.

See ``examples/mimic3_query_focused_ehr_attention.py`` for an
implementation demonstrating MIMIC-III preprocessing, training,
TF-IDF baseline comparison, AUROC and average precision evaluation,
and history-length ablations using 30-, 90-, and 365-day windows.

.. autoclass:: pyhealth.models.QueryFocusedEHRModel
    :members:
    :undoc-members:
    :show-inheritance:
