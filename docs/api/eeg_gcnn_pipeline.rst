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

Datasets and Signal Processing
-------------------------------

Two data paths are supported. Both produce the same output schema
(``node_features``, ``adj_matrix``, ``label``) that feeds directly into
:class:`~pyhealth.models.EEGGraphConvNet` and
:class:`~pyhealth.models.EEGGATConvNet`.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Path
     - Dataset class
     - Task class
     - Input
   * - **Raw EEG**
     - :class:`~pyhealth.datasets.EEGGCNNRawDataset`
     - :class:`~pyhealth.tasks.EEGGCNNDiseaseDetection`
     - Raw EDF (TUAB) / BrainVision (LEMON) files
   * - **Pre-computed**
     - :class:`~pyhealth.datasets.EEGGCNNDataset`
     - :class:`~pyhealth.tasks.EEGGCNNClassification`
     - FigShare joblib/npy arrays (1,593 subjects)

Data Sources
~~~~~~~~~~~~

**TUAB — Temple University EEG Abnormal Corpus (patient class, label 0)**
    Recordings from patients whose EEGs appear clinically normal despite an
    underlying neurological condition. Only the ``normal`` split is used.
    Files follow the EDF format with ``EEG X-REF`` channel naming.

    Expected directory layout::

        <root>/train/normal/01_tcp_ar/<subject_dirs>/*.edf
        <root>/eval/normal/01_tcp_ar/<subject_dirs>/*.edf

**LEMON — MPI Leipzig Mind-Body-Emotion (healthy class, label 1)**
    Recordings from fully healthy volunteers with no neurological history.
    Files are in BrainVision format (``*.eeg``, ``*.vhdr``, ``*.vmrk``).

    Expected directory layout::

        <root>/lemon/sub-<id>/sub-<id>.vhdr

Signal Processing Steps (EEGGCNNDiseaseDetection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~pyhealth.tasks.EEGGCNNDiseaseDetection` task applies the
following preprocessing to each raw recording:

1. **Resampling** — all recordings are resampled to 250 Hz.
2. **Filtering** — a 1 Hz high-pass filter removes slow drift; a 50 Hz notch
   filter removes power-line interference.
3. **Bipolar montage** — 8 bipolar channels are derived from the standard
   10-20 layout:

   .. code-block:: text

       F7-F3  |  F8-F4
       T7-C3  |  T8-C4
       P7-P3  |  P8-P4
       O1-P3  |  O2-P4

   LEMON uses bare channel names (``F7``, ``T7`` …); TUAB uses
   ``EEG X-REF`` naming. The task resolves both via a built-in alias map.

4. **Windowing** — each recording is segmented into non-overlapping 10-second
   windows. Incomplete trailing windows are discarded.

5. **PSD feature extraction** — for each window, Welch's method computes
   band-power in six frequency bands per bipolar channel:

   .. list-table::
      :header-rows: 1
      :widths: 25 20

      * - Band
        - Range (Hz)
      * - delta
        - 0.5 – 4.0
      * - theta
        - 4.0 – 8.0
      * - alpha
        - 8.0 – 12.0
      * - lower_beta
        - 12.0 – 20.0
      * - higher_beta
        - 20.0 – 30.0
      * - gamma
        - 30.0 – 50.0

   This yields a ``(8, 6)`` node-feature matrix per window (8 channels ×
   6 bands), log-transformed and normalised.

6. **Graph adjacency construction** — an ``(8, 8)`` adjacency matrix is
   built by blending two connectivity measures:

   .. code-block:: python

       edge_weight = alpha * geodesic_distance + (1 - alpha) * spectral_coherence

   - **Geodesic distance** (``alpha = 1.0``): arc length between electrode
     positions on a unit sphere — a proxy for *spatial* proximity.
   - **Spectral coherence** (``alpha = 0.0``): average cross-spectral
     coherence — a proxy for *functional* connectivity.
   - **Combined** (``alpha = 0.5``, paper default): equal blend.

   Self-connections (diagonal) are set to 1.0. All values are normalised
   to ``[0, 1]``.

Output Schema
~~~~~~~~~~~~~

After calling ``dataset.set_task(EEGGCNNDiseaseDetection())``, each sample
in the resulting :class:`~pyhealth.datasets.SampleDataset` contains:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Shape / Type
     - Description
   * - ``node_features``
     - ``torch.Tensor (8, 6)``
     - Log PSD band-power per bipolar channel, normalised
   * - ``adj_matrix``
     - ``torch.Tensor (8, 8)``
     - Edge weights in ``[0, 1]``, self-connections = 1.0
   * - ``label``
     - ``int`` — 0 or 1
     - 0 = TUAB (patient/diseased), 1 = LEMON (healthy)
   * - ``patient_id``
     - ``str``
     - Unique patient identifier (e.g. ``tuab_aaaaaaav``)

These tensors are the direct inputs to both
:class:`~pyhealth.models.EEGGraphConvNet` and
:class:`~pyhealth.models.EEGGATConvNet`.

Quick Start — Raw EEG Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run against the sample data shipped in ``pyhealth/eeg-gcnn-data/``:

.. code-block:: bash

    # GCN
    python examples/eeg_gcnn/pre_compute_gcnn.py

    # GAT
    python examples/eeg_gatcnn/pre_compute_gatcnn.py

Or point to your own data root:

.. code-block:: bash

    python examples/eeg_gcnn/pre_compute_gcnn.py --root /path/to/eeg-gcnn-data

The script prints the tensor shapes at each stage and performs a single
forward pass to confirm end-to-end compatibility::

    Stage 1 — Dataset: EEGGCNNRawDataset
      6 patients, 689 windows

    Stage 2 — Task: EEGGCNNDiseaseDetection
      node_features : torch.Size([8, 6])
      adj_matrix    : torch.Size([8, 8])
      label         : 0 or 1

    Stage 3 — Model: EEGGraphConvNet
      y_prob shape  : torch.Size([32, 1])
      ✓ GCN forward pass successful

Programmatic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhealth.datasets import EEGGCNNRawDataset
    from pyhealth.tasks import EEGGCNNDiseaseDetection
    from pyhealth.models import EEGGraphConvNet

    # 1. Load raw TUAB + LEMON data
    dataset = EEGGCNNRawDataset(root="/path/to/eeg-gcnn-data")

    # 2. Apply task: segment windows, extract PSD + adjacency
    task = EEGGCNNDiseaseDetection(adjacency_type="combined")
    sample_dataset = dataset.set_task(task)

    # 3. Inspect outputs
    sample = sample_dataset[0]
    print(sample["node_features"].shape)  # torch.Size([8, 6])
    print(sample["adj_matrix"].shape)     # torch.Size([8, 8])
    print(sample["label"])                # 0 (TUAB) or 1 (LEMON)

    # 4. Feed directly into GCN or GAT model
    model = EEGGraphConvNet(dataset=sample_dataset, num_node_features=6)
    out = model(
        node_features=sample["node_features"].unsqueeze(0),
        adj_matrix=sample["adj_matrix"].unsqueeze(0),
        label=sample["label"],
    )
    print(out["y_prob"])  # predicted probability of healthy class

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
