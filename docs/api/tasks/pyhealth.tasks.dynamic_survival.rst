DynamicSurvivalTask
==================

This module implements a dynamic survival analysis task for early event prediction.

The task follows the anchor-based discrete-time survival formulation proposed in:

Yèche et al. (2024), *Dynamic Survival Analysis for Early Event Prediction*.

Key Features
------------
- Multiple anchors per patient
- Discrete-time hazard prediction
- Support for censoring
- Configurable observation windows and anchor strategies

Output Format
-------------
Each processed sample contains:

- **patient_id**: unique patient identifier
- **visit_id**: unique anchor-based visit ID
- **x**: input features (temporal sequence)
- **y**: hazard label vector (0/1)
- **mask**: indicates valid risk set:
    - 1 = patient is at risk at this timestep
    - 0 = timestep excluded (post-event or post-censoring)

Usage Example
-------------

.. code-block:: python

    from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask

    # Minimal dataset wrapper (MockDataset or a real PyHealth dataset)
    class MockDataset:
        def __init__(self):
            self.patients = {}

    dataset = MockDataset()

    task = DynamicSurvivalTask(
        dataset=dataset,
        observation_window=24,
        horizon=24,
        anchor_strategy="fixed",
    )

    # Apply to a patient object
    samples = task(patient)

Example Output
--------------

Each sample:

- x: shape (T, d)
- y: shape (horizon,)
- mask: shape (horizon,)

API Reference
-------------

.. autoclass:: pyhealth.tasks.dynamic_survival.DynamicSurvivalTask
   :members:
   :undoc-members:
   :show-inheritance:
