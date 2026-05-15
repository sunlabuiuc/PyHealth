TDLSTMMortality
================

.. currentmodule:: pyhealth.models.td_lstm_mortality

.. autoclass:: TDLSTMMortality
    :members:
    :undoc-members:
    :show-inheritance:

Overview
--------

``TDLSTMMortality`` is a PyHealth-style reproduction model for ICU mortality
prediction inspired by the paper:

Frost, Li, and Harris. *Robust Real-Time Mortality Prediction in the ICU using
Temporal Difference Learning* (ML4H 2024).

This implementation provides a lightweight and contribution-friendly version of
the original idea using:

- an LSTM encoder over fixed-length time-series features
- binary mortality prediction
- supervised training with terminal binary cross-entropy loss
- temporal-difference (TD) training with bootstrapped future targets

Compared with the original paper, this implementation intentionally simplifies
the architecture to improve reproducibility and compatibility with the PyHealth
model contribution workflow.

Key Features
------------

- Supports ``training_mode="supervised"`` for standard binary mortality prediction
- Supports ``training_mode="td"`` for temporal-difference learning
- Uses a PyHealth-compatible ``BaseModel`` interface
- Accepts schema-based sample datasets created with ``create_sample_dataset``
- Returns standard PyHealth outputs including ``loss``, ``y_prob``,
  ``y_true``, and ``logit``

Constructor
-----------

.. code-block:: python

    from pyhealth.models.td_lstm_mortality import TDLSTMMortality

    model = TDLSTMMortality(
        dataset=dataset,
        feature_key="x",
        label_key="label",
        mode="binary",
        hidden_dim=32,
        gamma=0.95,
        alpha_terminal=0.10,
        n_step=1,
        training_mode="td",
    )

Parameters
----------

- ``dataset``:
  PyHealth sample dataset used to infer input/output structure.
- ``feature_key``:
  Key for the time-series feature tensor.
- ``label_key``:
  Key for the binary mortality label.
- ``mode``:
  Currently only ``"binary"`` is supported.
- ``hidden_dim``:
  Hidden size of the LSTM encoder.
- ``num_layers``:
  Number of LSTM layers.
- ``dropout``:
  Dropout rate used when ``num_layers > 1``.
- ``gamma``:
  Discount factor for TD target construction.
- ``alpha_terminal``:
  Weight for the terminal supervised anchor loss in TD mode.
- ``n_step``:
  Number of future steps used in TD target bootstrapping.
- ``lengths_key``:
  Optional key for sequence lengths when variable-length sequences are used.
- ``embedding_dim``:
  Reserved embedding dimension argument for compatibility/future extension.
- ``training_mode``:
  Either ``"supervised"`` or ``"td"``.

Input Format
------------

The model expects batched time-series input with shape ``[B, T, F]`` after
PyHealth collation.

For schema-based synthetic/sample datasets, the raw per-sample format can be:

.. code-block:: python

    {
        "patient_id": "p1",
        "visit_id": "v1",
        "x": [timestamps, values],
        "label": 1,
    }

where:

- ``timestamps`` is a list of Python ``datetime`` objects
- ``values`` is a list of length-``T`` feature vectors

Output
------

The forward pass returns a dictionary with keys such as:

- ``loss``: scalar training loss
- ``y_prob``: final mortality probability
- ``y_true``: binary ground-truth label
- ``logit``: final prediction logit
- ``logits_seq``: per-time-step logits
- ``probs_seq``: per-time-step probabilities

In TD mode, the output also includes:

- ``td_loss``: temporal-difference loss term
- ``terminal_loss``: supervised terminal BCE anchor

Example
-------

See the runnable example script:

``examples/mimic4_mortality_td_lstm.py``

This example demonstrates:

- synthetic ICU-style time-series sample generation
- train/validation/test split by patient
- supervised LSTM benchmark training
- TD-learning ablation sweep across discount factors
- final metric reporting with AUROC, F1, recall, and balanced accuracy

Notes
-----

This reproduction is aligned with a course project focused on implementing a
PyHealth model contribution based on a published healthcare ML paper. The
implementation emphasizes:

- clean integration with PyHealth APIs
- lightweight reproducible experiments
- fast synthetic tests
- clear separation between supervised and TD training modes