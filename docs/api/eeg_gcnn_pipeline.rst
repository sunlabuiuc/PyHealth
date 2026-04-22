EEG-GCNN Neurological Disease Classification Pipeline
======================================================

End-to-end pipeline for EEG-based neurological disease detection, based on:

    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    *Machine Learning for Health (ML4H) workshop, NeurIPS 2020*.
    https://proceedings.mlr.press/v136/wagh20a.html

**Contributors:** Jimmy Burhan (jburhan2@illinois.edu) — Dataset & Task |
Robert Coffey (rc37@illinois.edu) — Models & Training

Overview
--------

Each EEG recording is converted into a graph: nodes are bipolar electrode
pairs, edges encode brain connectivity. A GCN or GAT model classifies each
window as diseased (TUAB) or healthy (LEMON), then patient-level predictions
are aggregated for final evaluation.

There are two independent data paths into the model. Both feed
``EEGGCNNDataset`` + ``EEGGCNNClassification`` and produce identical output
tensors — the difference is whether the five required arrays come from raw
EEG files (preprocessed locally) or from FigShare (downloaded directly).

.. code-block:: text

    Path A — Raw EEG (run preprocessing locally)
    ────────────────────────────────────────────
    Raw TUAB EDF + LEMON BrainVision files
          ↓  EEGGCNNRawDataset.precompute_features()
    5 arrays (psd_features_data_X, labels_y, …)
          ↓  EEGGCNNDataset
          ↓  EEGGCNNClassification
    node_features (8,6) + adj_matrix (8,8) + label
          ↓  EEGGraphConvNet  or  EEGGATConvNet
    Patient-level AUC, Youden's J

    Path B — FigShare pre-computed arrays
    ─────────────────────────────────────
    Download 5 arrays from FigShare (1,593 subjects)
          ↓  EEGGCNNDataset
          ↓  EEGGCNNClassification
    node_features (8,6) + adj_matrix (8,8) + label
          ↓  EEGGraphConvNet  or  EEGGATConvNet
    Patient-level AUC, Youden's J

.. list-table::
   :header-rows: 1
   :widths: 18 28 28 26

   * - Path
     - Input
     - Preprocessing
     - Output (consumed by EEGGCNNDataset)
   * - **A. Raw EEG**
     - EDF (TUAB) + BrainVision (LEMON)
     - :class:`~pyhealth.datasets.EEGGCNNRawDataset` runs the full pipeline
     - 5 arrays written to ``precomputed_data/``
   * - **B. FigShare**
     - Pre-computed download (1,593 subjects, 225,334 windows)
     - None — already done by the paper authors
     - Same 5 arrays

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

There are two raw sources (TUAB + LEMON) used by **Path A**, and one
pre-computed source (FigShare) used by **Path B**.

Path A — Raw inputs
~~~~~~~~~~~~~~~~~~~~

**TUAB — Temple University EEG Abnormal Corpus (raw, label 0 = patient class)**
    EEG recordings from patients whose EEGs appear clinically *normal* but
    who have an underlying neurological condition. Only the ``normal`` split
    is used. Files are EDF format with ``EEG X-REF`` channel naming.

    Download: `TUH EEG Abnormal Corpus <https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/>`_
    (requires `registration <https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php>`_).

    Directory layout::

        <root>/tuab/train/normal/01_tcp_ar/*.edf
        <root>/tuab/eval/normal/01_tcp_ar/*.edf

**LEMON — MPI Leipzig Mind-Body-Emotion Interactions (raw, label 1 = healthy class)**
    EEG recordings from fully healthy volunteers with no neurological history.
    BrainVision format (``*.eeg`` + ``*.vhdr`` + ``*.vmrk``).

    Download: `MPI LEMON <http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html>`_
    (no registration needed). Alternatively use the provided download script::

        python download_lemon.py          # all 213 subjects
        python download_lemon.py --n 10   # first 10 subjects only

    Directory layout::

        <root>/lemon/sub-<id>/sub-<id>.vhdr

Path B — Pre-computed input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FigShare pre-computed arrays (precomputed, large-scale training)**
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

Signal Processing
-----------------

The pipeline below is applied to each raw recording. There are two
implementations depending on the use case:

- **``EEGGCNNRawDataset.precompute_features()``** (batch) — runs once over
  all raw files and saves the five output arrays to disk.  This is what
  ``pre_compute.py`` calls, and it is the path used for all training,
  holdout evaluation, and ablation runs described in this document.

- **``EEGGCNNDiseaseDetection.__call__()``** (streaming) — processes one
  patient at a time on-the-fly during training without saving intermediate
  files.  Supports configurable ``adjacency_type`` (``spatial``,
  ``functional``, ``combined``) and a custom ``bands`` dict for band-level
  ablation.  Available as an alternative entry point but not used in the
  training or ablation runs described here — those use ``EEGGCNNDataset``
  with the ``alpha`` parameter and ``EEGGCNNClassification`` with
  ``excluded_bands`` instead.

Steps applied in order:

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

5. **PSD feature extraction** — Welch's method per window, 6 bands per channel.
   Band names and ranges match the FigShare arrays and ``EEGGCNNClassification``
   (used for the spectral ablation study):

   .. list-table::
      :header-rows: 1
      :widths: 30 20

      * - Band
        - Range (Hz)
      * - delta
        - 0 – 4.0
      * - theta
        - 4.0 – 7.5
      * - alpha
        - 7.5 – 13.0
      * - beta
        - 13.0 – 30.0
      * - low_gamma
        - 30.0 – 40.0
      * - high_gamma
        - 40.0 – 50.0

   Result: ``(8, 6)`` node-feature matrix per window (48 values total),
   log-transformed and L2-normalised.

6. **Graph adjacency** — ``(8, 8)`` matrix blending spatial and functional
   connectivity, computed in ``EEGGCNNDataset``:

   .. code-block:: python

       edge_weight = alpha * geodesic_distance + (1 - alpha) * spectral_coherence

   - ``alpha = 1.0`` — spatial only (geodesic distance between electrodes)
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

    # Precompute features from raw TUAB + LEMON
    python eeg_gcnn_classification_gcn_precompute.py --root raw_data --output precomputed_data

    # Limit subjects for a fast validation run
    python eeg_gcnn_classification_gcn_precompute.py --max-tuab 10 --max-lemon 10

    # Train the GCN on the precomputed features
    python eeg_gcnn_classification_gcn_training.py

Expected output from ``eeg_gcnn_classification_gcn_precompute.py``::

    EEG-GCNN Feature Precomputation
      raw data  : .../examples/eeg_gcnn/raw_data
      output    : .../examples/eeg_gcnn/precomputed_data
      subset    : both

    Done. Five pre-computed files written to: .../precomputed_data
    Next step: python eeg_gcnn_classification_gcn_training.py

Programmatic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhealth.datasets import EEGGCNNRawDataset

    # 1. Point at the directory containing raw_data/tuab/ and raw_data/lemon/
    dataset = EEGGCNNRawDataset(root="/path/to/raw_data")

    # 2. Run the full preprocessing pipeline and write the 5 output files
    dataset.precompute_features(output_dir="/path/to/precomputed_data")

    # Optional: limit subjects for a faster dev run
    dataset.precompute_features(
        output_dir="/path/to/precomputed_data",
        max_tuab=10,
        max_lemon=10,
    )

    # 3. Load the pre-computed features with EEGGCNNDataset for training
    from pyhealth.datasets import EEGGCNNDataset
    train_ds = EEGGCNNDataset(root="/path/to/precomputed_data")

Training (GCN)
--------------

Trains on whatever 5-array dataset sits in ``DATA_ROOT`` — either the
locally pre-computed output of ``eeg_gcnn_classification_gcn_precompute.py`` (Path A) or the FigShare
download (Path B, 1,593 subjects). The script does not care which one it
is, since both produce the same five files.
Run from ``examples/eeg_gcnn/``:

.. code-block:: bash

    conda activate pyhealth
    python eeg_gcnn_classification_gcn_training.py

Key options:

.. code-block:: python

    ALPHA         = 0.5    # adjacency mix: 1.0=spatial, 0.0=functional
    NUM_FOLDS     = 10     # cross-validation folds (set to 2 for a quick test)
    NUM_EPOCHS    = 100
    LEARNING_RATE = 0.01
    MAX_PATIENTS  = None   # set e.g. 20 for a fast dev run

.. note::

   To limit how many subjects are included at the *precompute* stage (before
   training), use the ``--max-tuab`` / ``--max-lemon`` flags of
   ``eeg_gcnn_classification_gcn_precompute.py`` or the ``max_tuab`` / ``max_lemon`` arguments of
   :meth:`~pyhealth.datasets.EEGGCNNRawDataset.precompute_features`.

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
    python eeg_gcnn_classification_gat_training.py

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

    python eeg_gcnn_classification_gcn_evaluation.py    # GCN
    python eeg_gcnn_classification_gat_evaluation.py  # GAT

Both scripts:

1. Load each fold checkpoint.
2. Run inference on held-out patients.
3. Aggregate window-level probabilities to patient level (mean).
4. Select optimal threshold via **Youden's J statistic** (J = sensitivity +
   specificity − 1) on the ROC curve.
5. Report mean ± std across folds.

Reported metrics: ``auroc_patient``, ``precision``, ``recall``, ``f1``,
``bal_acc``.

Results
-------

All results use the full raw dataset (TUAB normal + MPI LEMON), 70/30
patient-level train/test split, 10-fold cross-validation, combined
adjacency (α = 0.5). GCN trained with SGD + MultiStepLR; GAT trained
with Adam + ReduceLROnPlateau (lr=1e-3, patience=5, factor=0.5).

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - Model
     - AUROC
     - Precision
     - Recall
     - F1
     - Bal. Acc
   * - Shallow EEG-GCNN (paper, FigShare)
     - 0.90 ± 0.02
     - —
     - —
     - —
     - 0.83 ± 0.02
   * - **GCN** (raw data, α = 0.5)
     - 0.914 ± 0.008
     - 0.978 ± 0.006
     - 0.822 ± 0.043
     - 0.892 ± 0.023
     - 0.849 ± 0.007
   * - **GAT** (raw data, Adam + Plateau)
     - **0.942 ± 0.025**
     - **0.986 ± 0.009**
     - **0.864 ± 0.041**
     - **0.920 ± 0.023**
     - **0.885 ± 0.032**

Both models trained on the raw pipeline exceed the paper's FigShare
baseline. The GAT with Adam + ReduceLROnPlateau outperforms the GCN on
every metric, reversing the earlier result where GAT underperformed due
to SGD being a poor fit for attention-based architectures.

Ablation Studies
----------------

Adjacency Type Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

We extended the paper by varying the adjacency mix coefficient ``alpha``
across five configurations and evaluating both models on the full FigShare
dataset (477 test patients, 70/30 patient-level split):

- α = 0.0  — functional only (spectral coherence)
- α = 0.25 — coherence-heavy
- α = 0.5  — combined (paper default)
- α = 0.75 — spatial-heavy
- α = 1.0  — spatial only (geodesic distance)

For each α we trained both ``EEGGraphConvNet`` and ``EEGGATConvNet`` and
reported AUC, Youden's J, balanced accuracy, recall, and F1. Numerical
results are in the project slides.

Re-run with a different alpha:

.. code-block:: python

    ALPHA = 0.0   # functional only — in eeg_gcnn_classification_gcn_training.py
    ALPHA = 0.25  # coherence-heavy
    ALPHA = 0.5   # combined (paper default)
    ALPHA = 0.75  # spatial-heavy
    ALPHA = 1.0   # spatial only

Spectral Frequency Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To determine which frequency bands carry the most discriminative signal,
we ran a leave-one-out (LOO) and keep-one-in (KOI) analysis at inference
time using trained GCN checkpoints — no retraining required. The 13
conditions are:

- **Baseline**: all 6 bands active
- **Leave-one-out** (×6): one band zeroed, remaining 5 active
- **Keep-one-in** (×6): only one band active, remaining 5 zeroed

Numerical results (AUC, balanced accuracy, F1 — mean ± std across 10
folds) are in the project slides.

Run from ``examples/eeg_gcnn/``:

.. code-block:: bash

    python eeg_gcnn_classification_gcn_band_ablation.py

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
