pyhealth.models.MEME
====================

MEME, the Multiple Embedding Model for EHR, from Lee et al., "Clinical
decision support using pseudo-notes from multiple streams of EHR data" (npj
Digital Medicine 8, 394, 2025; `doi:10.1038/s41746-025-01777-x
<https://doi.org/10.1038/s41746-025-01777-x>`_; preprint `arXiv:2402.00160
<https://arxiv.org/abs/2402.00160>`_). Each domain of an ED visit is
written as a templated pseudo-note
(:class:`~pyhealth.tasks.EDDispositionMIMIC4`), embedded with a frozen clinical
BERT (MedBERT by default), and self-attention across the note embeddings feeds
a linear head. Example: ``examples/meme_mimic4_ed_disposition.ipynb``.

.. autoclass:: pyhealth.models.MEME
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.DomainSelfAttention
    :members:
    :undoc-members:
    :show-inheritance:
