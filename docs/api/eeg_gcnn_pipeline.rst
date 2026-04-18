EEG-GCNN Neurological Disease Classification Pipeline
======================================================

This page describes the end-to-end pipeline for EEG-based neurological
disease classification using the EEG-GCNN architecture introduced in:

    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    *Machine Learning for Health (ML4H) workshop, NeurIPS 2020*.
    https://arxiv.org/abs/2011.10432

Two model variants are provided:

- :class:`~pyhealth.models.EEGGraphConvNet` — Graph Convolutional Network (GCN)
- :class:`~pyhealth.models.EEGGATConvNet` — Graph Attention Network (GAT)

Example scripts are located in ``examples/eeg_gcnn/`` and
``examples/eeg_gatcnn/``.

Environment Setup
-----------------

A dedicated conda environment is recommended to avoid dependency conflicts,
particularly with PyTorch Geometric. The usage commands in this page assume
an environment named ``pyhealth`` is active — replace this with your own
environment name if different.

.. code-block:: bash

    conda create -n pyhealth python=3.12 -y
    conda activate pyhealth

Then install PyHealth and the EEG-specific dependencies:

.. code-block:: bash

    # PyHealth (from source for development)
    git clone https://github.com/sunlabuiuc/PyHealth.git
    cd PyHealth
    pip install -e .

    # PyTorch (CPU — replace with a CUDA build if using a GPU)
    pip install torch

    # PyTorch Geometric and its dependencies
    pip install torch-geometric
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

    # Additional pipeline dependencies
    pip install scikit-learn joblib tqdm pandas

.. note::

   PyTorch Geometric wheel URLs depend on your PyTorch and CUDA versions.
   See `<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
   for the correct installation command for your system.

For general PyHealth installation options see :doc:`../install`.

Overview
--------

Each EEG recording is segmented into fixed-length windows. Every window is
represented as a fully-connected graph over 8 bipolar EEG electrode pairs:

- **Node features** (shape ``(8, 6)``): six PSD band-power values per
  electrode (delta, theta, alpha, beta, low-gamma, high-gamma),
  L2-normalised across the full 48-dimensional feature vector.
- **Edge weights** (shape ``(8, 8)``): a weighted average of geodesic
  electrode distance and spectral coherence, controlled by ``alpha``.

The pipeline has four stages:

1. **Dataset** — :class:`~pyhealth.datasets.EEGGCNNDataset` loads the
   pre-computed feature arrays and builds per-window ``.npy`` caches.
2. **Task** — :class:`~pyhealth.tasks.EEGGCNNClassification` converts
   patient records into binary classification samples.
3. **Training** — 10-fold cross-validation with subject-level 70/30
   train+val / heldout-test split.
4. **Evaluation** — heldout test evaluation with patient-level aggregation
   and Youden's J threshold selection.

Required Data Files
-------------------

Place the following pre-computed files in a single directory (``DATA_ROOT``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``psd_features_data_X``
     - joblib array, shape ``(N, 48)`` — PSD band-power features
   * - ``labels_y``
     - joblib array, shape ``(N,)`` — class labels
   * - ``master_metadata_index.csv``
     - CSV with a ``patient_ID`` column
   * - ``spec_coh_values.npy``
     - numpy array, shape ``(N, 64)`` — spectral coherence values
   * - ``standard_1010.tsv.txt``
     - TSV with columns ``label``, ``x``, ``y``, ``z`` for electrode positions

These files are available from the Figshare repository:
`EEG-GCNN pre-computed features <https://figshare.com/articles/dataset/EEG-GCNN_Augmenting_Electroencephalogram-based_Neurological_Disease_Diagnosis_using_a_Domain-guided_Graph_Convolutional_Neural_Network/13251509>`_

Training (GCN)
--------------

Run from the ``examples/eeg_gcnn/`` directory:

.. code-block:: bash

    conda activate pyhealth
    python training_pipeline_shallow_gcnn.py

Key configuration options in the script:

.. code-block:: python

    ALPHA           = 0.5    # edge weight mix: 1.0=geodesic, 0.0=coherence
    NUM_FOLDS       = 10     # set to 2 for a quick smoke-test
    NUM_EPOCHS      = 100
    LEARNING_RATE   = 0.01
    MAX_PATIENTS    = None   # set to e.g. 20 for a fast dev run

Checkpoints are saved to ``output_data/`` as:

.. code-block:: text

    psd_gnn_shallow_ph_alpha0.50_fold_0.ckpt
    psd_gnn_shallow_ph_alpha0.50_fold_1.ckpt
    ...

Training (GAT)
--------------

Run from the ``examples/eeg_gatcnn/`` directory:

.. code-block:: bash

    conda activate pyhealth
    python training_pipeline_shallow_gatcnn.py

Key configuration options in the script:

.. code-block:: python

    DROPOUT_VALUES      = [0.2]      # classifier dropout ablation
    ATTN_DROPOUT_VALUES = [0.0]      # GAT attention dropout ablation
    NUM_FOLDS           = 10
    NUM_EPOCHS          = 100
    MAX_PATIENTS        = None

Checkpoints are saved to ``output_data/`` as:

.. code-block:: text

    psd_gat_shallow_ph_drop2_attn0_fold_0.ckpt
    psd_gat_shallow_ph_drop2_attn0_fold_1.ckpt
    ...

Heldout Test Evaluation
-----------------------

After training, evaluate on the held-out 30% test subjects:

.. code-block:: bash

    # GCN
    python heldout_test_run_gcnn.py

    # GAT
    python heldout_test_run_gatcnn.py

Both scripts load each fold checkpoint, run inference on the held-out
patients, aggregate window-level predictions to patient level by averaging
probabilities, select the optimal decision threshold via Youden's J
statistic, and report mean ± std across folds:

.. code-block:: text

    auroc_window  : 0.8342 ± 0.0079
    auroc_patient : 0.8970 ± 0.0110
    precision     : 0.9866 ± 0.0060
    recall        : 0.7198 ± 0.0320
    f1            : 0.8318 ± 0.0200
    bal_acc       : 0.8237 ± 0.0050

Frequency Band Ablation (GCN)
------------------------------

Run 13 ablation conditions (baseline + 6 leave-one-out + 6 keep-one-in)
without retraining, using the existing alpha=0.5 checkpoints:

.. code-block:: bash

    python run_band_ablation.py

The ``excluded_bands`` parameter of
:class:`~pyhealth.tasks.EEGGCNNClassification` zeros out the specified
frequency band columns in the node-feature matrix at inference time:

.. code-block:: python

    # Leave-one-out: exclude delta
    samples = dataset.set_task(EEGGCNNClassification(excluded_bands=["delta"]))

    # Keep-one-in: only delta active
    others = [b for b in BAND_NAMES if b != "delta"]
    samples = dataset.set_task(EEGGCNNClassification(excluded_bands=others))

Edge Weight Ablation
--------------------

Re-run training with a different ``ALPHA`` value to ablate the contribution
of geodesic distance vs. spectral coherence:

.. code-block:: python

    ALPHA = 1.0   # geodesic distance only
    ALPHA = 0.0   # spectral coherence only
    ALPHA = 0.5   # equal mix (default)

The heldout evaluation script accepts the same ``ALPHA`` setting to load
the corresponding checkpoints.

API Reference
-------------

- :doc:`datasets/pyhealth.datasets.EEGGCNNDataset`
- :doc:`tasks/pyhealth.tasks.EEGGCNNClassification`
- :doc:`models/pyhealth.models.EEGGraphConvNet`
- :doc:`models/pyhealth.models.EEGGATConvNet`
