pyhealth.models.WatchSleepNet
===================================

WatchSleepNet is a sequence-level sleep-staging model adapted for PyHealth.
It combines residual CNN feature extraction, temporal convolutional
networks (TCN), bidirectional LSTM layers, and multi-head attention.

The implementation is designed for DREAMT inter-beat interval (IBI) data
with 750 samples per 30-second epoch at 25 Hz. It supports three sleep
stages: Wake, NREM, and REM, and handles variable-length patient sequences.

This DREAMT-focused implementation does not include the original paper's SHHS/MESA pretraining or transfer-learning pipeline.

The model is based on:

Wang et al., "WatchSleepNet: A Novel Model and Pretraining Approach for
Advancing Sleep Staging with Smartwatches," PMLR 287, 2025.

Paper: https://proceedings.mlr.press/v287/wang25a.html

Original implementation:
https://github.com/WillKeWang/WatchSleepNet_public/

Input and output
----------------

The model accepts batches of variable-length sequences of sleep epochs.
Each epoch contains 750 IBI samples. Valid epochs are classified into
Wake (0), NREM (1), and REM (2). Padded sequence positions are excluded
from the classification outputs.

Configuration and ablations
---------------------------

The default architecture uses 256 CNN channels, a kernel size of 5,
256 hidden units, 16 attention heads, four bidirectional LSTM layers,
and three TCN layers.

The following options support controlled architectural comparisons:

* ``use_tcn=False``: disable the temporal convolutional network.
* ``use_attention=False``: disable multi-head attention.
* ``num_layers``: change the number of LSTM layers.
* ``hidden_dim``: change the LSTM hidden dimension.
* ``class_weights``: configure class-weighted classification loss.
* ``residual_attention=True``: add a residual connection around the
  attention output. This option requires attention to be enabled and
  is disabled by default.

The residual attention option is an experimental architectural
extension. Its effectiveness must be assessed on held-out participants.

Usage
-----

.. code-block:: python

    from pyhealth.models import WatchSleepNet

    model = WatchSleepNet(dataset=sample_dataset)

    extended_model = WatchSleepNet(
        dataset=sample_dataset,
        residual_attention=True,
    )

Here, ``sample_dataset`` is a compatible processed PyHealth dataset
created using the DREAMT sequence sleep-staging task.

Reproduction scope
------------------

This implementation adapts the original architecture to the PyHealth
dataset and model interfaces. It does not claim numerical equivalence
to the pretrained model reported in the original paper.

Differences include the temporal convolution implementation,
sequence construction, and the DREAMT-only training protocol.
Evaluation should use participant-disjoint splits to prevent epochs
from the same participant appearing in both training and evaluation.

.. autoclass:: pyhealth.models.WatchSleepNet
    :members:
    :undoc-members:
    :show-inheritance:

Research reproduction and evaluation
------------------------------------

This implementation reproduces the WatchSleepNet architecture for
three-class sleep staging using DREAMT v2.2.0. It is a partial,
DREAMT-only reproduction rather than a numerical reproduction of the
complete published multi-dataset training protocol.

The original paper is available at
`WatchSleepNet (PMLR 287, 2025)
<https://proceedings.mlr.press/v287/wang25a.html>`_.

The accompanying example,
``examples/dreamt_sleep_staging_watchsleepnet.py``, demonstrates
participant-disjoint training, validation and testing. Its lightweight
defaults are intended for functional verification on limited hardware.

Example usage
~~~~~~~~~~~~~

Install the optional DREAMT dependency and obtain authorized access to
the dataset before running the example.

.. code-block:: bash

    pip install -e ".[dreamt]"

    python examples/dreamt_sleep_staging_watchsleepnet.py \
        --root /path/to/dreamt \
        --cache-dir /path/to/cache \
        --epochs 1 --max-folds 1 --device cpu

For a larger architecture with both experimental extensions enabled:

.. code-block:: bash

    python examples/dreamt_sleep_staging_watchsleepnet.py \
        --root /path/to/dreamt \
        --cache-dir /path/to/cache \
        --num-channels 256 --hidden-dim 256 \
        --kernel-size 5 --num-heads 16 \
        --num-layers 4 --tcn-layers 3 \
        --use-attention --residual-attention \
        --context-seconds 5 --learning-rate 0.00005 \
        --epochs 1 --max-folds 1 --device cpu

The second command demonstrates architecture and preprocessing
configuration, not the complete research training protocol.

Architectural ablations can be configured with ``--no-tcn``, by
omitting ``--use-attention``, or by reducing ``--num-layers``.
Residual attention requires attention to be enabled.

Research protocol
~~~~~~~~~~~~~~~~~

The separate research experiments used 20 DREAMT participants:
16 for development and four reserved for independent holdout evaluation.

The principal research configuration used a random seed of 42,
32-epoch sequence chunks, batch size 16, a maximum of 200 training
epochs, early stopping patience of 20, learning rate 0.00005,
and weight decay 0.0001. Training used Apple MPS.

The research architecture used 256 CNN channels, a hidden dimension
of 256, kernel size 5, 16 attention heads, four BiLSTM layers and
three TCN layers.

The selected research checkpoint additionally used:

* Residual attention.
* Five seconds of surrounding BVP context on each side of an epoch.
* Square-root inverse-frequency class weighting.
* REM-aware training sampling with strength 3.0.

Checkpoint selection used development validation macro-F1.
The selected checkpoint was from epoch 43. Holdout evaluation used
the existing checkpoint without retraining or holdout-based tuning,
without evaluation sampling, and with unweighted cross-entropy.

The research-specific sampling, weighting, checkpoint recovery and
holdout orchestration are not implemented by the lightweight example.
Consequently, its commands should not be interpreted as exact
reproduction commands for the reported research scores.

Development cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seven configurations were evaluated across five participant-level
development folds. The following values are exploratory fold means
with standard deviations for macro-F1.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Configuration
     - Accuracy
     - Macro-F1 (mean +/- SD)
   * - Original architecture
     - 0.6783
     - 0.3022 +/- 0.0433
   * - Residual attention
     - 0.6241
     - 0.3451 +/- 0.0453
   * - No attention
     - 0.6352
     - 0.3178 +/- 0.0390
   * - No TCN
     - 0.6392
     - 0.3412 +/- 0.0808
   * - One-layer BiLSTM
     - 0.5373
     - 0.3174 +/- 0.0177
   * - No TCN or attention
     - 0.5233
     - 0.3331 +/- 0.0293
   * - Epoch-level MLP
     - 0.6490
     - 0.3702 +/- 0.0366

Residual attention improved mean macro-F1 relative to the original
architecture configuration, but reduced mean accuracy. The epoch-level
MLP achieved the highest mean macro-F1 in this comparison while
producing zero mean REM F1. These comparisons do not establish
statistical significance or superiority across all metrics.

Independent holdout results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The selected research checkpoint was evaluated on four reserved
participants comprising 3046 evaluated epochs.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Selected model
     - Majority baseline
   * - Accuracy
     - 0.5489
     - 0.6060
   * - Macro-F1
     - 0.3928
     - 0.2516
   * - Weighted F1
     - 0.5317
     - 0.4574
   * - Balanced accuracy
     - 0.3904
     - 0.3333
   * - REM F1
     - 0.1453
     - 0.0000
   * - Cohen's kappa
     - 0.1221
     - 0.0000
   * - Macro-OVR AUROC
     - 0.5895
     - 0.5000

The selected model exceeded the majority baseline on six of seven
reported metrics, but its overall accuracy was lower.

Limitations
~~~~~~~~~~~

* SHHS and MESA pretraining and transfer learning were not reproduced.
* The participant cohort is small, and holdout metrics are pooled
  across epochs rather than accompanied by participant-level
  uncertainty estimates.
* Epoch segmentation depends on sleep-stage annotations.
* Context-aware extraction uses future signal context and is
  intended for retrospective analysis, not causal online inference.
* Development cross-validation is exploratory.
* Stored development validation metrics were not independently replayed.
* AUROC and loss were retained from evaluator outputs rather than
  reconstructed from confusion matrices.
* Wake and REM classification performance remains limited.
* External validation, statistical significance and clinical
  effectiveness have not been established.

These research results are distinct from the synthetic unit tests
and lightweight real-data smoke tests used to verify the PyHealth
implementation.
