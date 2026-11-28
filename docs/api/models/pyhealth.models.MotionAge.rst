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


Model Ablation
--------------

Pooling strategies and hidden dimensions were compared using
binary wear masks and the same participant split, training
configuration, and random seed. Continuous wear weighting is
included for comparison.

.. list-table::
    :header-rows: 1

    * - Configuration
      - Parameters
      - AUROC
      - AP
      - F1
      - Accuracy
    * - Binary baseline
      - 4765
      - 0.8562
      - 0.3333
      - 0.3778
      - 0.8250
    * - Continuous wear
      - 4765
      - 0.8667
      - 0.3605
      - 0.4186
      - 0.8438
    * - Mean pooling
      - 4765
      - 0.8530
      - 0.3781
      - 0.3855
      - 0.8406
    * - Last pooling
      - 4765
      - 0.7476
      - 0.2891
      - 0.2171
      - 0.6844
    * - Hidden dim 16
      - 1725
      - 0.8366
      - 0.3780
      - 0.3256
      - 0.8187
    * - Hidden dim 64
      - 15453
      - 0.8718
      - 0.3308
      - 0.3864
      - 0.8313

Mean pooling achieved the highest average precision, while
hidden dimension 64 achieved the highest AUROC at the cost
of additional parameters. Last pooling performed worse than
the masked baseline.

Continuous wear weighting improved all four observed metrics
over the binary baseline without additional parameters.

AUPRC is reported as average precision (AP).

Experimental Limitations
------------------------

The experiments used 2,129 participants from a 25% stratified
subset of NHANES 2003-2006, with 320 test participants and
21 positive test participants.

All comparisons used a single random seed (42).
Model ablation results are exploratory and were not subjected
to separate statistical significance testing.

The reproduction does not include the original paper's full
cross-validation protocol or biological age calibration.

.. autoclass:: pyhealth.models.MotionAgeLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MotionAge
    :members:
    :undoc-members:
    :show-inheritance:
