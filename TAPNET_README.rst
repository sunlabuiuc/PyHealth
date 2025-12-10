TapNet: Time Series Attentional Prototype Network
===============================================

Overview
--------
- Location: ``pyhealth/models/tapnet.py``; registered via ``pyhealth.models.TapNet`` and ``TapNetLayer``.
- Purpose: lightweight time-series model that combines temporal convolution + attention pooling with a small set of learnable prototypes, then concatenates per-feature representations for prediction.
- Inputs: any mix of sequential categorical features (e.g., codes) and numeric time-series/tensor features supported by ``SampleDataset`` + ``EmbeddingModel``.
- Outputs: logits/probabilities and optional embeddings and prototype attention.

Minimal usage
-------------
.. code-block:: python

   from pyhealth.datasets import SampleDataset, get_dataloader
   from pyhealth.models import TapNet

   samples = [
       {"patient_id": "p1", "visit_id": "v1",
        "conditions": ["c1", "c2"], "labs": [[1.0, 2.0]], "label": 1},
       {"patient_id": "p2", "visit_id": "v1",
        "conditions": ["c2"], "labs": [[0.5, 1.5]], "label": 0},
   ]
   dataset = SampleDataset(
       samples=samples,
       input_schema={"conditions": "sequence", "labs": "tensor"},
       output_schema={"label": "binary"},
       dataset_name="demo",
   )
   model = TapNet(dataset, embedding_dim=128, hidden_dim=128, num_prototypes=8)

   loader = get_dataloader(dataset, batch_size=2, shuffle=True)
   batch = next(iter(loader))
   out = model(**batch)  # keys: loss, y_prob, y_true, logit
   out = model(**batch, embed=True, return_attn=True)  # adds embed + prototype_attention

Key arguments
-------------
- ``embedding_dim``: embedding size for all inputs (default 128).
- ``hidden_dim``: hidden size in TapNet layers (default 128).
- ``num_prototypes``: number of learnable prototypes per feature stream (default 8).
- ``kernel_size``: temporal conv kernel (default 3).
- ``dropout``: dropout after conv (default 0.1).

Test coverage
-------------
- File: ``tests/core/test_tapnet.py`` mirrors existing model tests.
- Covers: initialization metadata, forward outputs/shapes, backward/gradients, optional embedding + prototype attention outputs, custom hyperparameters.

How to run the tests
--------------------
1) Ensure pytest is in your active environment:
   - ``python -m pip install pytest``  (or use your project’s installer)
2) From repo root, run with the project venv and repo on ``PYTHONPATH``:
   - ``PYTHONPATH=. ./.venv/bin/python -m pytest tests/core/test_tapnet.py``
   - Or ``PYTHONPATH=. python -m pytest tests/core/test_tapnet.py`` if using another interpreter.
3) If your environment blocks creating ``~/.cache/pyhealth/``, set a writable HOME (e.g., ``HOME=$(pwd)/.cache_home``) before running so imports succeed.

Notes
-----
- Optional prototype attention can be returned by passing ``return_attn=True`` to ``model(**batch)``.
- Patient-level embedding can be returned with ``embed=True``.
