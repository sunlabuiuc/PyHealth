"""PTB-XL multi-label ECG classification task.

This module provides :class:`PTBXLMultilabelClassification`, a
:class:`~pyhealth.tasks.BaseTask` subclass that turns a
:class:`~pyhealth.datasets.PTBXLDataset` into a multi-label classification
problem.

Two label spaces are supported, selected via the ``label_type`` constructor
argument.  This design enables the **ablation study** described in the project
paper: hold the model and training hyper-parameters constant and vary only the
label granularity (and optionally the signal sampling rate) to observe how
label coarseness affects downstream ROC-AUC and F1 performance.

Mathematical framing
--------------------
Let :math:`X \\in \\mathbb{R}^{C \\times T}` be a single ECG recording with
:math:`C = 12` leads and :math:`T` time-steps (1,000 at 100 Hz or 5,000 at
500 Hz).  Given a label universe of :math:`K` classes, the ground-truth
annotation is a binary vector :math:`y \\in \\{0, 1\\}^K` (multi-hot).

A model :math:`f_\\theta` maps the ECG to per-class logit scores:

.. math::

    \\hat{y} = \\sigma\\!\\left(f_\\theta(X) W^\\top + b\\right) \\in [0,1]^K

Training minimises the element-wise **binary cross-entropy**:

.. math::

    \\mathcal{L} = -\\frac{1}{K} \\sum_{k=1}^{K}
        \\Bigl[ y_k \\log \\hat{y}_k + (1 - y_k) \\log (1 - \\hat{y}_k) \\Bigr]

Evaluation uses **macro-averaged ROC-AUC**:

.. math::

    \\overline{\\text{AUC}} = \\frac{1}{K} \\sum_{k=1}^{K}
        \\int_0^1 \\text{TPR}_k(t)\\, d\\text{FPR}_k(t)

and **macro-averaged F1** (at threshold 0.5):

.. math::

    \\overline{F_1} = \\frac{1}{K} \\sum_{k=1}^{K}
        \\frac{2 \\cdot \\text{TP}_k}{2 \\cdot \\text{TP}_k + \\text{FP}_k + \\text{FN}_k}

Label spaces
------------
``"superdiagnostic"`` (:data:`SUPERDIAG_CLASSES` — 5 classes)
    Directly mirrors the five PTB-XL superdiagnostic categories from
    Strodthoff et al. (2020).  SNOMED-CT codes from every recording's
    ``# Dx:`` list are mapped to one or more of NORM / MI / STTC / CD / HYP
    using :data:`SNOMED_TO_SUPERDIAG`.  Records with no mappable code are
    skipped.

``"diagnostic"`` (:data:`CHALLENGE_SNOMED_CLASSES` — 27 classes)
    Uses the 27 SNOMED-CT codes that were officially scored in the
    PhysioNet/CinC Challenge 2020.  Each code present in a recording's
    ``# Dx:`` list that falls within this vocabulary becomes a positive label.
    Records with no scored codes are skipped.

Ablation axes
-------------
The two constructor arguments create the natural ablation grid:

+-------------------+-----------+------------------------+
| ``label_type``    | ``sampling_rate`` | Description   |
+===================+===========+========================+
| ``"superdiagnostic"`` | 100   | 5-class / 100 Hz       |
+-------------------+-----------+------------------------+
| ``"superdiagnostic"`` | 500   | 5-class / 500 Hz       |
+-------------------+-----------+------------------------+
| ``"diagnostic"``  | 100       | 27-class / 100 Hz      |
+-------------------+-----------+------------------------+
| ``"diagnostic"``  | 500       | 27-class / 500 Hz      |
+-------------------+-----------+------------------------+

Author:
    CS-598 DLH Project Team
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label-space definitions
# ---------------------------------------------------------------------------

#: Mapping from SNOMED-CT code (string) to one of the 5 PTB-XL superdiagnostic
#: classes.  Codes absent from this dict are silently ignored during label
#: construction.  The mapping follows Table 1 of Strodthoff et al. (2020) and
#: the PhysioNet Challenge 2020 label alignment documented in the challenge
#: description paper.
SNOMED_TO_SUPERDIAG: Dict[str, str] = {
    # ------ NORM — Normal sinus rhythm ----------------------------------- #
    "426783006": "NORM",
    # ------ MI — Myocardial Infarction ----------------------------------- #
    "57054005":  "MI",   # Acute myocardial infarction
    "164865005": "MI",   # Myocardial infarction
    "413444003": "MI",   # Acute MI of anterolateral wall
    "413867000": "MI",   # Acute MI of inferior wall
    "164861001": "MI",   # Anterior MI
    "164857002": "MI",   # Inferior MI
    "164860000": "MI",   # Anteroseptal MI
    "164864009": "MI",   # Posterior MI
    "164867002": "MI",   # Lateral MI
    # ------ STTC — ST/T-wave Change -------------------------------------- #
    "164931005": "STTC",  # ST elevation
    "164934002": "STTC",  # ST depression
    "59931005":  "STTC",  # Inverted T-wave / T-wave abnormality
    "164947007": "STTC",  # Prolonged PR interval
    "164917005": "STTC",  # Prolonged QT interval
    "251268003": "STTC",  # Early repolarisation pattern
    "428750005": "STTC",  # Non-specific ST-T change
    # ------ CD — Conduction Disturbance / Rhythm Disorder ---------------- #
    "270492004": "CD",    # First-degree AV block
    "195042002": "CD",    # Second-degree AV block
    "27885002":  "CD",    # Third-degree AV block
    "6374002":   "CD",    # Bundle branch block (unspecified)
    "713427006": "CD",    # Complete right bundle branch block (CRBBB)
    "713426002": "CD",    # Complete left bundle branch block (CLBBB)
    "164909002": "CD",    # Left bundle branch block
    "59118001":  "CD",    # Right bundle branch block
    "698252002": "CD",    # Non-specific intraventricular conduction disturbance
    "445118002": "CD",    # Left anterior fascicular block (LAFB)
    "10370003":  "CD",    # Pacing rhythm
    "164889003": "CD",    # Atrial fibrillation
    "164890007": "CD",    # Atrial flutter
    "426627000": "CD",    # Bradycardia
    "427393009": "CD",    # Sinus arrhythmia
    "426177001": "CD",    # Sinus bradycardia
    "427084000": "CD",    # Sinus tachycardia
    "63593006":  "CD",    # Supraventricular premature beats
    "17338001":  "CD",    # Ventricular premature beats
    "284470004": "CD",    # Premature atrial contraction
    "427172004": "CD",    # Premature ventricular contraction
    # ------ HYP — Hypertrophy / Axis Deviation --------------------------- #
    "55827005":  "HYP",   # Left ventricular hypertrophy
    "446358003": "HYP",   # Right ventricular hypertrophy
    "73282002":  "HYP",   # Biventricular hypertrophy
    "67751000119106": "HYP",  # Left atrial enlargement
    "446813000": "HYP",   # Right atrial enlargement
    "39732003":  "HYP",   # Left axis deviation
    "47665007":  "HYP",   # Right axis deviation
    "251146004": "HYP",   # Low QRS voltage
}

#: Ordered list of the 5 superdiagnostic class names.  The ordering is
#: deterministic so that model outputs are consistently interpretable.
SUPERDIAG_CLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

#: The 27 SNOMED-CT codes officially scored in the PhysioNet/CinC Challenge
#: 2020 (alphabetically sorted by their clinical abbreviation for readability).
#: These form the label universe for ``label_type="diagnostic"``.
CHALLENGE_SNOMED_CLASSES: List[str] = sorted(
    [
        "270492004",  # IAVB  — First-degree atrioventricular block
        "164889003",  # AF    — Atrial fibrillation
        "164890007",  # AFL   — Atrial flutter
        "6374002",    # BBB   — Bundle branch block (unspecified)
        "426627000",  # Brady — Bradycardia
        "713427006",  # CRBBB — Complete right bundle branch block
        "713426002",  # CLBBB — Complete left bundle branch block
        "445118002",  # LAnFB — Left anterior fascicular block
        "39732003",   # LAD   — Left axis deviation
        "164909002",  # LBBB  — Left bundle branch block
        "251146004",  # LQRSV — Low QRS voltage
        "698252002",  # NSIVCB — Non-specific intraventricular conduction dist.
        "10370003",   # PR    — Pacing rhythm
        "164947007",  # LPR   — Prolonged PR interval
        "164917005",  # LQT   — Prolonged QT interval
        "47665007",   # RAD   — Right axis deviation
        "427393009",  # SA    — Sinus arrhythmia
        "426177001",  # SB    — Sinus bradycardia
        "426783006",  # NSR   — Normal sinus rhythm
        "427084000",  # ST    — Sinus tachycardia
        "63593006",   # SVPB  — Supraventricular premature beats
        "164934002",  # STD   — ST depression
        "59931005",   # TWA   — T-wave abnormality
        "164931005",  # STE   — ST elevation
        "17338001",   # VPB   — Ventricular premature beats
        "284470004",  # PAC   — Premature atrial contraction
        "427172004",  # PVC   — Premature ventricular contraction
    ]
)

_CHALLENGE_SET: frozenset = frozenset(CHALLENGE_SNOMED_CLASSES)


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class PTBXLMultilabelClassification(BaseTask):
    """Multi-label 12-lead ECG classification on PTB-XL.

    For each ECG recording this task:

    1. Loads the ``.mat`` signal matrix via :func:`scipy.io.loadmat`
       (shape ``(12, 5000)`` at 500 Hz).
    2. Optionally decimates the signal to 100 Hz (shape ``(12, 1000)``).
    3. Parses SNOMED-CT codes from the ``scp_codes`` event attribute.
    4. Maps those codes to the chosen label space (superdiagnostic or
       full Challenge 27-class).
    5. Returns one sample dict per valid recording::

       {
           "signal": np.ndarray,  # shape (12, T), float32
           "labels": List[str],   # positive class names / SNOMED strings
       }

    Args:
        sampling_rate (int): Target sampling rate in Hz.  Accepted values are
            ``100`` (decimation ×5 from the native 500 Hz; yields ``T = 1000``)
            and ``500`` (no resampling; yields ``T = 5000``).
            Defaults to ``100``.
        label_type (str): Label vocabulary to use.  ``"superdiagnostic"``
            yields 5 classes (NORM, MI, STTC, CD, HYP);
            ``"diagnostic"`` yields 27 SNOMED-CT classes from the PhysioNet
            Challenge 2020 scoring list.  Defaults to ``"superdiagnostic"``.

    Raises:
        ValueError: If ``sampling_rate`` is not 100 or 500.
        ValueError: If ``label_type`` is not ``"superdiagnostic"`` or
            ``"diagnostic"``.

    Examples:
        Superdiagnostic task at 100 Hz (default)::

            >>> from pyhealth.datasets import PTBXLDataset
            >>> from pyhealth.tasks import PTBXLMultilabelClassification
            >>> dataset = PTBXLDataset(root="/data/.../training/ptb-xl/")
            >>> task = PTBXLMultilabelClassification()
            >>> sample_ds = dataset.set_task(task)
            >>> sample_ds[0]["labels"]      # e.g. ["NORM"] or ["CD", "STTC"]

        27-class diagnostic task at 500 Hz (ablation variant)::

            >>> task_27 = PTBXLMultilabelClassification(
            ...     sampling_rate=500, label_type="diagnostic"
            ... )
            >>> sample_ds_27 = dataset.set_task(task_27)

    See Also:
        :data:`SNOMED_TO_SUPERDIAG`, :data:`SUPERDIAG_CLASSES`,
        :data:`CHALLENGE_SNOMED_CLASSES`
    """

    task_name: str = "PTBXLMultilabelClassification"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"labels": "multilabel"}

    def __init__(
        self,
        sampling_rate: int = 100,
        label_type: str = "superdiagnostic",
    ) -> None:
        super().__init__()

        if sampling_rate not in (100, 500):
            raise ValueError(
                f"sampling_rate must be 100 or 500, got {sampling_rate}."
            )
        if label_type not in ("superdiagnostic", "diagnostic"):
            raise ValueError(
                "label_type must be 'superdiagnostic' or 'diagnostic', "
                f"got '{label_type}'."
            )

        self.sampling_rate = sampling_rate
        self.label_type = label_type

        # Disambiguate the task_name so that cached SampleDatasets from
        # different configurations do not collide on disk.
        self.task_name = (
            f"PTBXLSuperDiagnostic_{sampling_rate}Hz"
            if label_type == "superdiagnostic"
            else f"PTBXLDiagnostic27_{sampling_rate}Hz"
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def __call__(self, patient: Patient) -> List[Dict]:
        """Extract samples from one patient (= one ECG recording in PTB-XL).

        Args:
            patient: A :class:`~pyhealth.data.Patient` object whose events
                have ``event_type="ptbxl"`` and carry attributes
                ``mat_file``, ``scp_codes``, ``age``, and ``sex``.

        Returns:
            A list with at most one sample dict
            ``{"signal": np.ndarray, "labels": List[str]}``, or an empty list
            if the recording should be skipped (missing file, unrecognised
            codes, etc.).
        """
        # In PTBXLDataset each patient has exactly one event in the "ptbxl"
        # table (record == patient).
        events = patient.get_events(event_type="ptbxl")
        samples = []

        for event in events:
            # ---- 1. Load the .mat signal --------------------------------
            mat_file = getattr(event, "ptbxl/mat", None)
            if not mat_file:
                logger.debug("Skip %s: no *.mat file", event)
                continue

            try:
                from scipy.io import loadmat as _loadmat
                mat = _loadmat(mat_file)
                signal = mat["val"].astype(np.float32)  # (12, 5000) @ 500 Hz
            except Exception as exc:
                logger.warning("Cannot load signal from %s: %s", mat_file, exc)
                continue

            if signal.ndim != 2 or signal.shape[0] != 12:
                logger.warning(
                    "Unexpected signal shape %s in %s; skipping.",
                    signal.shape,
                    mat_file,
                )
                continue

            # ---- 2. Resample if needed (decimation only) ----------------
            # Native rate is 500 Hz (5000 samples / 10 s).
            # Decimation by 5 gives 100 Hz (1000 samples / 10 s).
            if self.sampling_rate == 100:
                signal = signal[:, ::5]  # shape (12, 1000)

            # ---- 3. Parse SNOMED-CT codes --------------------------------
            dx_codes: str = str(getattr(event, "ptbxl/dx_codes", "") or "")
            codes = [c.strip() for c in dx_codes.split(",") if c.strip()]

            # ---- 4. Map to chosen label space ---------------------------
            if self.label_type == "superdiagnostic":
                labels = list(
                    {
                        SNOMED_TO_SUPERDIAG[c]
                        for c in codes
                        if c in SNOMED_TO_SUPERDIAG
                    }
                )
            else:  # "diagnostic" — 27-class Challenge vocabulary
                labels = [c for c in codes if c in _CHALLENGE_SET]

            if not labels:
                # No recognised labels → skip (consistent with other tasks).
                continue

            samples.append({"signal": signal, "labels": labels})

        return samples