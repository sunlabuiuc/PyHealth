pyhealth.models.Unified_MLP
==========================

An MLP baseline for packed hierarchical records and static tensors.
Each level transforms its records before mean pooling into their parents.
Static tensors are flattened per sample and passed through independent MLPs.

.. autoclass:: pyhealth.models.Unified_MLP
    :members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import Unified_MLP, Unified_LogisticRegression

    dataset = create_sample_dataset(
        [
            {"patient_id": "p0", "age": [50.0], "visits": {
                "children": [{"tokens": ["A", "B"]}, {"tokens": []}]
            }, "label": 0},
            {"patient_id": "p1", "age": [65.0], "visits": {
                "children": [{"tokens": ["B"]}]
            }, "label": 1},
        ],
        {"age": "tensor", "visits": ("sequence_v2", {"levels": [
            {"ordering": "set", "tokens": 1}, {"ordering": "sequence"}
        ]})},
        {"label": "binary"},
    )
    dims = {"visits": [{"tokens": (8,)}, {}]}
    batch = next(iter(get_dataloader(dataset, batch_size=2)))
    model = Unified_MLP(dataset, dims, hidden_dim=16)
    result = model(**batch)
    linear_baseline = Unified_LogisticRegression(dataset, dims)
    linear_result = linear_baseline(**batch)

For per-step labels, give each sequence field a ``series`` top level.
At label time ``t``, both models mean-pool the available top-level record
representations with timestamps ``<= t``. No record yields a zero summary.
Resampled record ``i`` has timestamp ``i * step``. A top-level timestamp
must indicate when the complete record, including its children, is available.

Record ordering and timestamps are not input features. Spatial arrays are
flattened and projected within each record; these models do not learn spatial
locality. Nested means weight parent records equally, so a parent with many
children does not receive more weight than one with few children.

The existing :class:`~pyhealth.models.MLP` remains available for legacy schemas.
