Contributor: Abdullah Rehman (arehman3), Benjamin Yang (bhyang2), Leny Pan (lenypan2)

Contribution Type: Model

Description:
This PR implements RNNAttention, a Recurrent Neural Network (RNN) with a multi-headed attention mechanism for predicting healthcare service utilization from individual disease trajectories. The model processes sequential patient visit data through an embedding layer, aggregates features, passes through a GRU with multi-headed attention, and produces predictions.

The implementation integrates with PyHealth's BaseModel API and supports PyHealth training and evaluation workflows.
Original Paper: Predicting utilization of healthcare services from individual disease trajectories using RNNs with multi-headed attention (https://dl.acm.org/doi/abs/10.1145/3291279.3332420)

Files to Review:

    pyhealth/models/rnn_with_attention.py — core model implementation
    tests/core/test_rnn_attention.py — unit tests using synthetic data
    examples/mimic3_readmission_rnn_attention.py — ablation study comparing model variants and baselines
    docs/api/models/pyhealth.models.RNNAttention.rst — API documentation
    docs/api/models.rst — updated model index
    pyhealth/models/__init__.py — model registration

Testing:

    All unit tests pass locally (python -m pytest -q tests/core/test_rnn_attention.py)
    Example ablation results (Synthetic MIMIC-III subset):
        RNNAttention (256d, 8h) AUROC: 0.0400
        RNN baseline (64d) AUROC: 0.3200
        Logistic Regression baseline (128d) AUROC: 0.0800

Notes:

    Synthetic data is used in tests and the ablation study to ensure fast CI execution and reproducibility without requiring MIMIC-III access.
