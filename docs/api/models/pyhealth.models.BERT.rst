pyhealth.models.BERT
===================================

BERT and BioBERT models for healthcare text classification.

This module provides BERT-based models specifically designed for healthcare NLP tasks.
It supports the following pre-trained models:

- Standard BERT (bert-base-uncased, bert-base-cased)
- BioBERT (dmis-lab/biobert-v1.1) - Pre-trained on PubMed abstracts
- Any other HuggingFace BERT-compatible model

Features
--------

- **Multiple pooling strategies**: CLS token, mean pooling, max pooling
- **Fine-tuning control**: Freeze entire encoder or specific layers
- **Differential learning rates**: Apply different LR to encoder vs classifier

Available Model Aliases
-----------------------

The following model aliases can be used with the ``model_name`` parameter:

- ``"bert-base-uncased"`` → ``bert-base-uncased``
- ``"bert-base-cased"`` → ``bert-base-cased``
- ``"biobert"`` → ``dmis-lab/biobert-v1.1``

Example Usage
-------------

Basic binary classification:

.. code-block:: python

    from pyhealth.datasets import SampleDataset, get_dataloader
    from pyhealth.models import BERT
    from pyhealth.trainer import Trainer

    # Create dataset
    samples = [
        {"patient_id": "p0", "visit_id": "v0", "note": "Chest pain...", "label": 1},
        {"patient_id": "p1", "visit_id": "v1", "note": "Wellness visit...", "label": 0},
    ]
    dataset = SampleDataset(
        samples=samples,
        input_schema={"note": "text"},
        output_schema={"label": "binary"},
        dataset_name="demo",
    )

    # Create BERT model
    model = BERT(
        dataset=dataset,
        model_name="bert-base-uncased",
        pooling="cls",
    )

    # Train
    trainer = Trainer(model=model, metrics=["accuracy", "f1"])
    trainer.train(train_dataloader, val_dataloader, epochs=5)

Using BioBERT for medical NLP:

.. code-block:: python

    model = BERT(
        dataset=dataset,
        model_name="biobert",  # Uses dmis-lab/biobert-v1.1
        pooling="cls",
        freeze_layers=6,  # Freeze bottom 6 layers
    )

BERTLayer
---------

The standalone BERT encoder layer that can be used in custom architectures.

.. autoclass:: pyhealth.models.BERTLayer
    :members:
    :undoc-members:
    :show-inheritance:

BERT Model
----------

The complete BERT model for text classification.

.. autoclass:: pyhealth.models.BERT
    :members:
    :undoc-members:
    :show-inheritance:

Biomedical Model Registry
-------------------------

Dictionary mapping model aliases to HuggingFace model identifiers.

.. autodata:: pyhealth.models.BIOMEDICAL_MODELS
    :annotation:

