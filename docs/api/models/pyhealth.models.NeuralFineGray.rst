pyhealth.models.NeuralFineGray
==============================

Overview
--------

Neural Fine-Gray directly models a cumulative incidence function for every
competing event. A softmax balancing network keeps their sum at or below one,
and smooth positive monotonic networks make every incidence curve
non-decreasing. Automatic differentiation supplies the exact event density,
avoiding time discretization and numerical quadrature.

Inputs and outputs
------------------

The model expects ``features: tensor`` as input and ``duration: tensor`` plus
``event: tensor`` as outputs. Event zero denotes right censoring; positive
integers select competing causes. ``y_prob`` and ``cif`` have shape
``[batch, num_risks, len(time_grid)]``. ``survival`` is one minus the CIF sum.

Example and ablation
--------------------

Run the complete synthetic example and CDF-parameterization ablation with::

    python examples/framingham_competing_risk_neural_fine_gray.py --epochs 20

Use ``--data-root /path/to/framingham --download`` for the public Framingham
route. Full setup and result notes are in
``examples/neural_fine_gray_results/README.md``.

API
---

.. autoclass:: pyhealth.models.NeuralFineGray
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.PositiveLinear
   :members:
   :show-inheritance:
