Unified Embedding E2E Protocol (MIMIC-IV)
=========================================

Objective
---------

Establish a reproducible end-to-end test path for unified multimodal embedding
with both ``MLP`` and ``RNN`` heads:

1. ingest MIMIC-IV data,
2. construct multimodal task samples,
3. train/infer with ``UnifiedMultimodalEmbeddingModel``,
4. emit concrete prediction rows.

This is designed to conceptually mirror the multimodal event-prediction setup
in `Lee et al., 2023 <https://arxiv.org/pdf/2305.02504>`_:

- multimodal EHR inputs with asynchronous events,
- clinical event prediction horizons,
- first milestone = mortality prediction within 12 hours.

Implemented First Milestone
---------------------------

Task:
``pyhealth.tasks.MultimodalMortalityHorizonMIMIC4``

- Observation window: configurable (default ``24h``)
- Prediction horizon: configurable (default ``12h``)
- Label: mortality within horizon
- Inputs:
  - ``icd_codes`` (diagnoses + procedures, StageNet)
  - ``labs`` (10-category lab vectors, StageNet tensor)
  - optional notes (discharge/radiology, tuple-time-text)

Runner:
``examples/mortality_prediction/unified_embedding_e2e_mimic4.py``

Acceptance Criteria
-------------------

For each model head (``MLP`` and ``RNN``):

1. ``MIMIC4Dataset(...).set_task(MultimodalMortalityHorizonMIMIC4(...))`` returns non-empty samples.
2. A forward pass on a batch returns ``y_prob`` and ``loss``.
3. Inference returns aligned arrays of ``patient_id``, ``y_true``, ``y_prob``.
4. Predictions are written to CSV with one row per sample.

Reference Test
--------------

``tests/core/test_unified_e2e_mimic4.py`` validates the end-to-end path on the
local MIMIC-IV demo resources.

Run:

.. code-block:: bash

   pytest tests/core/test_unified_e2e_mimic4.py -v

Run on Full MIMIC-IV
--------------------

RNN head:

.. code-block:: bash

   python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
     --ehr-root /path/to/mimiciv/2.2 \
     --model rnn \
     --observation-window-hours 24 \
     --prediction-horizon-hours 12 \
     --epochs 3 \
     --batch-size 64 \
     --output-dir ./output/unified_e2e

MLP head:

.. code-block:: bash

   python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
     --ehr-root /path/to/mimiciv/2.2 \
     --model mlp \
     --observation-window-hours 24 \
     --prediction-horizon-hours 12 \
     --epochs 3 \
     --batch-size 64 \
     --output-dir ./output/unified_e2e

Output Artifact
---------------

The runner writes:

- ``<output-dir>/predictions_rnn.csv`` or
- ``<output-dir>/predictions_mlp.csv``

with columns:

- ``patient_id``
- ``y_true``
- ``y_prob``
- ``y_pred_threshold_0_5``
