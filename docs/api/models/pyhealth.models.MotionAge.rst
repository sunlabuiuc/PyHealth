pyhealth.models.MotionAge
===================================

MotionAge (Song et al., MLHC 2026): a wear-aware recurrent model that predicts
mortality from epoch-level wearable activity. Hidden states are pooled using binary wear masks or continuous wear
fractions. Optional static covariates are combined by late fusion.
The standalone MotionAgeLayer encoder can be reused on its own.


Supported Dataset and Task
--------------------------

- :doc:`../datasets/pyhealth.datasets.NHANESDataset`
- :doc:`../tasks/pyhealth.tasks.NHANESMortalityPrediction`

Wear-aware Pooling
------------------

The masked pooling mode supports binary wear masks and continuous
wear-fraction weights. The continuous representation extends binary
masking by accounting for partial wear within each epoch.

Ablation Results
----------------

The following participant-level results were obtained using a 25%
stratified NHANES subset, a wearable-only GRU, and a single random seed.

.. list-table::
    :header-rows: 1

    * - Wear representation
      - AUROC
      - AUPRC
      - F1
    * - Binary
      - 0.8562
      - 0.3333
      - 0.3778
    * - Continuous fraction
      - 0.8667
      - 0.3605
      - 0.4186

Continuous weighting improved the observed metrics. However, paired
bootstrap confidence intervals for AUROC and AUPRC differences included
zero. These results do not establish statistically significant
superiority.

The experiment evaluates mortality prediction rather than biological
age calibration.

.. autoclass:: pyhealth.models.MotionAgeLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MotionAge
    :members:
    :undoc-members:
    :show-inheritance:
