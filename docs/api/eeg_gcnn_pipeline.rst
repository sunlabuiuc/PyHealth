EEG-GCNN Neurological Disease Classification Pipeline
======================================================

End-to-end pipeline for EEG-based neurological disease detection, based on:

    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    *Machine Learning for Health (ML4H) workshop, NeurIPS 2020*.
    https://proceedings.mlr.press/v136/wagh20a.html

**Contributors:** Jimmy Burhan (jburhan2) — Dataset & Task |
Robert Coffey (racoffey2) — Models & Training

Overview
--------

Each EEG recording is converted into a graph: nodes are bipolar electrode
pairs, edges encode brain connectivity. A GCN or GAT model classifies each
window as diseased (TUAB) or healthy (LEMON), then patient-level predictions
are aggregated for final evaluation.

The pipeline has five stages:

.. code-block:: text

    Raw EEG / FigShare arrays
          ↓
    Dataset  (EEGGCNNRawDataset  or  EEGGCNNDataset)
          ↓
    Task     (EEGGCNNDiseaseDetection  or  EEGGCNNClassification)
          ↓  node_features (8,6)  +  adj_matrix (8,8)  +  label
    Model    (EEGGraphConvNet  or  EEGGATConvNet)
          ↓
    Evaluation  (patient-level AUC, Youden's J)

Two data paths are supported — both produce identical output tensors:

.. list-table::
   :header-rows: 1
   :widths: 15 30 30 25

   * - Path
     - Dataset
     - Task
     - Input
   * - **Raw EEG**
     - :class:`~pyhealth.datasets.EEGGCNNRawDataset`
     - :class:`~pyhealth.tasks.EEGGCNNDiseaseDetection`
     - EDF (TUAB) + BrainVision (LEMON) files
   * - **Pre-computed**
     - :class:`~pyhealth.datasets.EEGGCNNDataset`
     - :class:`~pyhealth.tasks.EEGGCNNClassification`
     - FigShare arrays (1,593 subjects, 225,334 windows)

Environment Setup
-----------------

.. code-block:: bash

    conda create -n pyhealth python=3.12 -y
    conda activate pyhealth

    # PyHealth (from source)
    git clone https://github.com/sunlabuiuc/PyHealth.git
    cd PyHealth
    pip install -e .

    # PyTorch (CPU build — swap for CUDA if needed)
    pip install torch

    # PyTorch Geometric
    pip install torch-geometric
    pip install torch-scatter torch-sparse \
        -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

    # EEG dependencies
    pip install scikit-learn joblib tqdm pandas mne mne-connectivity

.. note::

   See `PyTorch Geometric installation <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
   for the correct wheel URL for your CUDA version.

Data Sources
------------

**TUAB — Temple University EEG Abnormal Corpus (label 0, patient class)**
    EEG recordings from patients whose EEGs appear clinically *normal* but
    who have an underlying neurological condition. Only the ``normal`` split
    is used. Files are EDF format with ``EEG X-REF`` channel naming.

    Directory layout::

        <root>/tuab/train/normal/01_tcp_ar/*.edf
        <root>/tuab/eval/normal/01_tcp_ar/*.edf

**LEMON — MPI Leipzig Mind-Body-Emotion Interactions (label 1, healthy class)**
    EEG recordings from fully healthy volunteers with no neurological history.
    BrainVision format (``*.eeg`` + ``*.vhdr`` + ``*.vmrk``).

    Directory layout::

        <root>/lemon/sub-<id>/sub-<id>.vhdr

**FigShare pre-computed arrays (large-scale training)**
    Pre-processed features for 1,593 subjects (225,334 windows), downloaded
    from `FigShare <https://figshare.com/articles/dataset/EEG-GCNN_Augmenting_Electroencephalogram-based_Neurological_Disease_Diagnosis_using_a_Domain-guided_Graph_Convolutional_Neural_Network/13251509>`_:

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
         - CSV with ``patient_ID`` column
       * - ``spec_coh_values.npy``
         - numpy array, shape ``(N, 64)`` — spectral coherence values
       * - ``standard_1010.tsv.txt``
         - TSV with electrode 3-D coordinates (``label``, ``x``, ``y``, ``z``)

Signal Processing (EEGGCNNDiseaseDetection)
-------------------------------------------

Applied to each raw recording in order:

1. **Resample** to 250 Hz.
2. **Filter** — 1 Hz high-pass (remove DC drift) + 60 Hz notch for TUAB
   (US power-line), 50 Hz notch for LEMON (EU power-line).
3. **Bipolar montage** — derive 8 bipolar channels from the 10-20 layout:

   .. code-block:: text

       F7-F3  |  F8-F4
       T7-C3  |  T8-C4
       P7-P3  |  P8-P4
       O1-P3  |  O2-P4

   TUAB uses ``EEG X-REF`` naming; LEMON uses bare names (``F7``, ``T7``, …).
   A built-in alias map resolves both automatically.

4. **Windowing** — non-overlapping 10-second windows (incomplete trailing
   window discarded).

5. **PSD feature extraction** — Welch's method per window, 6 bands per channel:

   .. list-table::
      :header-rows: 1
      :widths: 30 20

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

   Result: ``(8, 6)`` node-feature matrix per window, log-transformed and
   L2-normalised.

6. **Graph adjacency** — ``(8, 8)`` matrix blending spatial and functional
   connectivity:

   .. code-block:: python

       edge_weight = alpha * geodesic_distance + (1 - alpha) * spectral_coherence

   - ``alpha = 1.0`` — spatial only (geodesic arc length on unit sphere)
   - ``alpha = 0.0`` — functional only (spectral coherence)
   - ``alpha = 0.5`` — combined (paper default)

   Self-connections (diagonal) = 1.0. All values normalised to ``[0, 1]``.

Output Schema
-------------

Both tasks produce the same sample dict consumed directly by both models:

.. list-table::
   :header-rows: 1
   :widths: 20 22 58

   * - Key
     - Shape / Type
     - Description
   * - ``node_features``
     - ``Tensor (8, 6)``
     - Log PSD band-power, L2-normalised
   * - ``adj_matrix``
     - ``Tensor (8, 8)``
     - Edge weights in ``[0, 1]``; diagonal = 1.0
   * - ``label``
     - ``int`` — 0 or 1
     - 0 = TUAB (patient), 1 = LEMON (healthy)
   * - ``patient_id``
     - ``str``
     - e.g. ``tuab_aaaaaaav`` or ``lemon_sub-032301``

Models
------

Both models take ``node_features (B, 8, 6)`` and ``adj_matrix (B, 8, 8)``
and return a dict with keys ``loss``, ``y_prob``, ``y_true``, ``logit``.

**EEGGraphConvNet** (:class:`~pyhealth.models.EEGGraphConvNet`)
    Two-layer Graph Convolutional Network (Kipf & Welling, 2017).
    Node features are aggregated via normalised adjacency message-passing,
    followed by global mean pooling and a binary classification head.
    Uses ``float32``.

**EEGGATConvNet** (:class:`~pyhealth.models.EEGGATConvNet`)
    Two-layer Graph Attention Network (Veličković et al., 2018).
    Attention coefficients are computed per edge, allowing the model to
    weight neighbours differently. Uses ``float64`` on CPU.

Quick Start — Raw EEG
---------------------

Preprocess raw TUAB and LEMON recordings into the five files required by
``EEGGCNNDataset``, then run training:

.. code-block:: bash

    cd examples/eeg_gcnn

    # (Optional) download LEMON subjects from the INDI S3 bucket
    python download_lemon.py --n 10   # first 10 subjects for a quick test

    # Precompute features from raw TUAB + LEMON
    python pre_compute_gcnn.py --root raw_data --output precomputed_data

    # Train the GCN on the precomputed features
    python training_pipeline_shallow_gcnn.py

Expected output from ``pre_compute_gcnn.py``::

    EEG-GCNN Feature Precomputation
      raw data  : .../examples/eeg_gcnn/raw_data
      output    : .../examples/eeg_gcnn/precomputed_data
      subset    : both

    Done. Five pre-computed files written to: .../precomputed_data
    Next step: python training_pipeline_shallow_gcnn.py

Programmatic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhealth.datasets import EEGGCNNRawDataset

    # 1. Point at the directory containing raw_data/tuab/ and raw_data/lemon/
    dataset = EEGGCNNRawDataset(root="/path/to/raw_data")

    # 2. Run the full preprocessing pipeline and write the 5 output files
    dataset.precompute_features(output_dir="/path/to/precomputed_data")

    # 3. Load the pre-computed features with EEGGCNNDataset for training
    from pyhealth.datasets import EEGGCNNDataset
    train_ds = EEGGCNNDataset(root="/path/to/precomputed_data")

Training (GCN)
--------------

Uses the FigShare pre-computed dataset for full-scale training (1,593 subjects).
Run from ``examples/eeg_gcnn/``:

.. code-block:: bash

    conda activate pyhealth
    python training_pipeline_shallow_gcnn.py

Key options:

.. code-block:: python

    ALPHA         = 0.5    # adjacency mix: 1.0=spatial, 0.0=functional
    NUM_FOLDS     = 10     # cross-validation folds (set to 2 for a quick test)
    NUM_EPOCHS    = 100
    LEARNING_RATE = 0.01
    MAX_PATIENTS  = None   # set e.g. 20 for a fast dev run

Class imbalance (~7:1 TUAB:LEMON) is handled with
``WeightedRandomSampler`` during training.

Checkpoints saved to ``output_data/``:

.. code-block:: text

    psd_gnn_shallow_ph_alpha0.50_fold_0.ckpt
    psd_gnn_shallow_ph_alpha0.50_fold_1.ckpt
    ...

Training (GAT)
--------------

Run from ``examples/eeg_gatcnn/``:

.. code-block:: bash

    conda activate pyhealth
    python training_pipeline_shallow_gatcnn.py

Key options:

.. code-block:: python

    DROPOUT_VALUES      = [0.2]   # classifier dropout
    ATTN_DROPOUT_VALUES = [0.0]   # GAT attention dropout
    NUM_FOLDS           = 10
    NUM_EPOCHS          = 100

Checkpoints saved to ``output_data/``:

.. code-block:: text

    psd_gat_shallow_ph_drop2_attn0_fold_0.ckpt
    ...

Heldout Evaluation
------------------

Evaluate on the held-out 30% test subjects after training:

.. code-block:: bash

    python heldout_test_run_gcnn.py    # GCN
    python heldout_test_run_gatcnn.py  # GAT

Both scripts:

1. Load each fold checkpoint.
2. Run inference on held-out patients.
3. Aggregate window-level probabilities to patient level (mean).
4. Select optimal threshold via **Youden's J statistic** (J = sensitivity +
   specificity − 1) on the ROC curve.
5. Report mean ± std across folds.

Results matching the paper (Table 2, combined adjacency α=0.5):

.. code-block:: text

    auroc_patient : 0.8970 ± 0.0110
    precision     : 0.9866 ± 0.0060
    recall        : 0.7198 ± 0.0320
    f1            : 0.8318 ± 0.0200
    bal_acc       : 0.8237 ± 0.0050

Ablation Studies
----------------

Adjacency Type Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

We extended the paper by comparing four adjacency configurations on both
GCN and GAT, evaluated on the full FigShare dataset (477 test patients,
70/30 patient-level split).

**GCN** (our best model — matches paper's Shallow EEG-GCNN):

.. list-table::
   :header-rows: 1
   :widths: 25 8 9 11 9 9 9

   * - Config
     - Alpha
     - AUC
     - Youden's J
     - Bal. Acc
     - Recall
     - F1
   * - Functional only
     - 0.0
     - 0.900
     - 0.645
     - 0.823
     - 0.879
     - 0.493
   * - Combined (paper)
     - 0.5
     - 0.902
     - 0.655
     - 0.828
     - 0.879
     - 0.503
   * - **Spatial-heavy**
     - **0.75**
     - **0.903**
     - **0.660**
     - **0.830**
     - **0.879**
     - **0.508**
   * - Spatial only
     - 1.0
     - 0.898
     - 0.623
     - 0.812
     - 0.862
     - 0.481

**GAT**:

.. list-table::
   :header-rows: 1
   :widths: 25 8 9 11 9 9 9

   * - Config
     - Alpha
     - AUC
     - Youden's J
     - Bal. Acc
     - Recall
     - F1
   * - Functional only
     - 0.0
     - 0.844
     - 0.599
     - 0.799
     - 0.759
     - 0.521
   * - Combined (paper)
     - 0.5
     - 0.842
     - 0.588
     - 0.794
     - 0.793
     - 0.484
   * - Spatial-heavy
     - 0.75
     - 0.841
     - 0.548
     - 0.774
     - 0.741
     - 0.473
   * - Spatial only
     - 1.0
     - 0.849
     - 0.588
     - 0.794
     - 0.810
     - 0.475

**Comparison to paper (Table 2, combined adjacency, 10-fold CV)**:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Model
     - AUC
     - Bal. Acc
   * - Shallow EEG-GCNN (paper)
     - 0.90 ± 0.02
     - 0.83 ± 0.02

GCN with α=0.75 achieves the highest AUC (0.903) and Youden's J (0.660),
marginally outperforming the paper's combined default (α=0.5). Spatial-only
(α=1.0) is the weakest GCN configuration, confirming that functional
coherence adds discriminative signal. GCN consistently outperforms GAT by
approximately 6% AUC across all configurations.

Re-run with a different alpha:

.. code-block:: python

    ALPHA = 0.0   # functional only — in training_pipeline_shallow_gcnn.py
    ALPHA = 0.5   # combined (paper default)
    ALPHA = 0.75  # spatial-heavy
    ALPHA = 1.0   # spatial only

Spectral Frequency Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To determine which frequency bands carry the most discriminative signal,
we ran a leave-one-out (LOO) and keep-one-in (KOI) analysis at inference
time using trained GCN checkpoints — no retraining required.  The 13
conditions are:

- **Baseline**: all 6 bands active
- **Leave-one-out** (×6): one band zeroed, remaining 5 active
- **Keep-one-in** (×6): only one band active, remaining 5 zeroed

Results (mean ± std across 10 folds, patient-level):

.. list-table::
   :header-rows: 1
   :widths: 30 24 24 22

   * - Condition
     - AUC
     - Bal. Acc
     - F1
   * - **Baseline**
     - **0.898 ± 0.010**
     - **0.828 ± 0.012**
     - **0.840 ± 0.028**
   * - LOO delta
     - 0.802 ± 0.059
     - 0.746 ± 0.043
     - 0.772 ± 0.055
   * - LOO theta
     - 0.800 ± 0.062
     - 0.752 ± 0.050
     - 0.785 ± 0.078
   * - LOO alpha
     - 0.807 ± 0.070
     - 0.751 ± 0.061
     - 0.786 ± 0.069
   * - LOO beta
     - 0.807 ± 0.070
     - 0.751 ± 0.061
     - 0.786 ± 0.069
   * - **LOO low gamma**
     - **0.594 ± 0.153**
     - **0.589 ± 0.144**
     - **0.793 ± 0.286**
   * - LOO high gamma
     - 0.785 ± 0.066
     - 0.737 ± 0.042
     - 0.800 ± 0.070
   * - KOI delta
     - 0.623 ± 0.207
     - 0.605 ± 0.241
     - 0.644 ± 0.214
   * - KOI theta
     - 0.575 ± 0.232
     - 0.569 ± 0.194
     - 0.675 ± 0.358
   * - KOI alpha
     - 0.661 ± 0.258
     - 0.639 ± 0.218
     - 0.856 ± 0.116
   * - KOI beta
     - 0.579 ± 0.106
     - 0.565 ± 0.138
     - 0.856 ± 0.116
   * - **KOI low gamma**
     - **0.723 ± 0.245**
     - **0.682 ± 0.202**
     - **0.781 ± 0.153**
   * - KOI high gamma
     - 0.656 ± 0.240
     - 0.633 ± 0.200
     - 0.840 ± 0.158

**Key finding**: removing low gamma causes the largest AUC drop (0.898 →
0.594), and low gamma alone (KOI) achieves the highest single-band AUC
(0.723). This confirms that the low gamma band (30–50 Hz) carries the most
discriminative signal for neurological disease detection.

Run from ``examples/eeg_gcnn/``:

.. code-block:: bash

    python run_band_ablation.py

Uses the ``excluded_bands`` parameter of
:class:`~pyhealth.tasks.EEGGCNNClassification`:

.. code-block:: python

    from pyhealth.tasks.eeg_gcnn_classification import EEGGCNNClassification, BAND_NAMES

    # Leave-one-out: remove low_gamma
    samples = dataset.set_task(EEGGCNNClassification(excluded_bands=["low_gamma"]))

    # Keep-one-in: only low_gamma active
    others = [b for b in BAND_NAMES if b != "low_gamma"]
    samples = dataset.set_task(EEGGCNNClassification(excluded_bands=others))

API Reference
-------------

Datasets:

- :doc:`datasets/pyhealth.datasets.EEGGCNNRawDataset` — raw EEG (TUAB + LEMON)
- :doc:`datasets/pyhealth.datasets.EEGGCNNDataset` — FigShare pre-computed

Tasks:

- :doc:`tasks/pyhealth.tasks.eeg_gcnn_disease_detection` — raw EEG task
- :doc:`tasks/pyhealth.tasks.EEGGCNNClassification` — FigShare task

Models:

- :doc:`models/pyhealth.models.EEGGraphConvNet` — GCN
- :doc:`models/pyhealth.models.EEGGATConvNet` — GAT
