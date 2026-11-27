pyhealth.models.KeyClass
===================================

KeyClass classifies clinical notes into diagnostic categories without any
human-labeled notes. It builds keyword labeling functions automatically from
short category descriptions, combines their votes into soft weak labels, and
trains a multilabel classifier on frozen document embeddings.

Paper: Chufan Gao, Mononito Goswami, Jieshi Chen and Artur Dubrawski.
`Classifying Unstructured Clinical Notes via Automatic Weak Supervision
<https://arxiv.org/abs/2206.12088v2>`_. MLHC 2022.

The pipeline:

1. Clean notes with :func:`~pyhealth.weak_supervision.clean_note`.
2. Fit a :class:`~pyhealth.weak_supervision.KeywordLabeler` on training notes and
   category descriptions, then vote with
   :meth:`~pyhealth.weak_supervision.KeywordLabeler.transform`.
3. Turn votes into soft weak labels with
   :func:`~pyhealth.weak_supervision.aggregate_votes`, and pick confident notes with
   :func:`~pyhealth.weak_supervision.confident_balanced_mask`.
4. Embed notes, for example with
   :class:`~pyhealth.weak_supervision.ClinicalTextEncoder`, and train
   :class:`~pyhealth.models.KeyClass` on the embeddings and soft labels.

.. code-block:: python

    from pyhealth.models import KeyClass
    from pyhealth.weak_supervision import (
        ClinicalTextEncoder, KeywordLabeler, aggregate_votes, clean_note,
    )

    notes = [clean_note(text) for text in raw_notes]
    encoder = ClinicalTextEncoder(
        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    )
    labeler = KeywordLabeler(encoder, top_k=300).fit(notes, descriptions)
    votes = labeler.transform(notes)
    weak_labels, covered = aggregate_votes(votes, len(descriptions))
    # Build a sample dataset with "embedding" and "weak_labels" tensor fields, then:
    model = KeyClass(dataset, input_dim=768, category_ids=list(descriptions))
    model.fit_input_standardization(embeddings)

Model
-----

.. autoclass:: pyhealth.models.KeyClass
    :members:
    :undoc-members:
    :show-inheritance:

Weak supervision
----------------

.. autoclass:: pyhealth.weak_supervision.KeywordLabeler
    :members:

.. autoclass:: pyhealth.weak_supervision.LabelingFunction

.. autoclass:: pyhealth.weak_supervision.ClinicalTextEncoder
    :members:

.. autofunction:: pyhealth.weak_supervision.clean_note

.. autofunction:: pyhealth.weak_supervision.nltk_english_stopwords

.. autofunction:: pyhealth.weak_supervision.aggregate_votes

.. autofunction:: pyhealth.weak_supervision.confident_balanced_mask

.. autofunction:: pyhealth.weak_supervision.snorkel_memory_gb

.. autofunction:: pyhealth.weak_supervision.vote_diagnostics
