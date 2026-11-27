pyhealth.models.MedTok
======================

Overview
--------

MedTok (Su et al., "Multimodal Medical Code Tokenizer", `ICML 2025
<https://arxiv.org/abs/2502.04397>`_; `code <https://github.com/mims-harvard/MedTok>`_,
MIT licence) tokenizes a medical code from both its text description and its
relations in a biomedical knowledge graph, quantized into one shared codebook
of 21,000 tokens. The authors release, under the MIT licence on `Hugging Face
<https://huggingface.co/mims-harvard/MedTok>`_, the 20 tokens and the
256-dimensional quantized embedding of about 616,000 codes (ICD-9, ICD-10-CM,
ICD-10-PCS, ATC, SNOMED CT, RxNorm and others).

This module plugs those released files into any PyHealth model built on
:class:`~pyhealth.models.EmbeddingModel`, the way the paper swaps the
tokenizer of existing EHR models:

1. :func:`~pyhealth.models.apply_medtok` replaces a built model's code
   embedding table, feature by feature, with a
   :class:`~pyhealth.models.MedTokEmbedding`.
2. ``mode="embedding"`` (the default, as the official EHR code) looks codes up
   in the frozen 256-d MedTok embeddings and learns a projection to the
   model's size, plus one shared vector for codes MedTok does not cover.
   ``mode="tokens"`` embeds a code as the mean of learned embeddings of its
   MedTok tokens.
3. :func:`~pyhealth.models.medtok_keys` converts dataset codes to MedTok's
   dotted keys (MIMIC stores ICD-9 ``428.0`` as ``4280``) and maps NDC drugs
   to ATC. For MIMIC-IV, whose readmission, length-of-stay and drug
   recommendation tasks write versioned codes such as ``9_4280`` and
   ``10_E119``, use ``"ICD_DX"`` and ``"ICD_PROC"``: the prefix picks ICD-9
   or ICD-10, so ``9_E8889`` (E888.9) and ``10_E8889`` (E88.89) stay
   distinct. ``MortalityPredictionMIMIC4`` drops the version, so its codes
   are ambiguous and are not supported.

.. note::

   Download ``code2embeddings.json`` (4 GB) or ``code2tokens.json`` (186 MB)
   from Hugging Face. :func:`~pyhealth.models.convert_medtok_embeddings`
   turns the embeddings into a 630 MB ``.npz`` that loads in seconds. MedTok
   does not cover every code. In the mortality task's samples on PyHealth's
   synthetic MIMIC-III, 76% of the distinct diagnosis codes (74% of
   diagnosis occurrences), 76% of procedure codes (72%) and 93% of drug codes
   (90%) have a MedTok entry; :func:`~pyhealth.models.apply_medtok` returns
   the distinct-code share for each feature. Some frequent codes, such as
   ICD-9 ``401.9``, are missing and use the shared missing-code vector.
   Training the tokenizer itself is not included.

Example
-------

.. code-block:: python

    from pyhealth.models import Transformer, apply_medtok, load_medtok

    model = Transformer(samples, embedding_dim=128)
    coverage = apply_medtok(
        model,
        "medtok.npz",
        {"conditions": "ICD9CM", "procedures": "ICD9PROC", "drugs": "NDC"},
    )

    # MIMIC-IV: samples from a task that keeps the ICD version, such as
    # ReadmissionPredictionMIMIC4 (codes like "9_4280" and "10_E119")
    model4 = Transformer(mimic4_samples, embedding_dim=128)
    coverage = apply_medtok(
        model4,
        "medtok.npz",
        {"conditions": "ICD_DX", "procedures": "ICD_PROC", "drugs": "NDC"},
    )

``examples/medtok_mimic.py`` compares learned and MedTok embeddings on
MIMIC-III (``--dataset mimic3``) or MIMIC-IV (``--dataset mimic4``).

API Reference
-------------

.. autofunction:: pyhealth.models.medtok_keys

.. autofunction:: pyhealth.models.load_medtok

.. autofunction:: pyhealth.models.convert_medtok_embeddings

.. autofunction:: pyhealth.models.apply_medtok

.. autoclass:: pyhealth.models.MedTokEmbedding
    :members:
    :show-inheritance:
