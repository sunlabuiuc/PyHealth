import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import scipy.signal
from joblib import dump
from mne.time_frequency import psd_array_welch

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bipolar montage constants
# ---------------------------------------------------------------------------

# TUAB 01_tcp_ar channel names (T3/T4/T5/T6 = older 10-20 names for T7/T8/P7/P8)
_TUAB_FIRST  = ["EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF",
                "EEG T5-REF", "EEG T6-REF", "EEG O1-REF", "EEG O2-REF"]
_TUAB_SECOND = ["EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF",
                "EEG P3-REF", "EEG P4-REF", "EEG P3-REF", "EEG P4-REF"]

# LEMON BrainVision channel names (uses updated T7/T8/P7/P8 naming)
_LEMON_FIRST  = ["F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"]
_LEMON_SECOND = ["F3", "F4", "C3", "C4", "P3", "P4", "P3", "P4"]

_NUM_NODES = 8
_NUM_BANDS = 6
_NUM_EDGES = _NUM_NODES ** 2

# delta lower bound is None (freqs <= 4.0), matching original eeg_pipeline.py
_BAND_RANGES: List[Tuple[Optional[float], float]] = [
    (None, 4.0),
    (4.0,  7.5),
    (7.5,  13.0),
    (13.0, 16.0),
    (16.0, 30.0),
    (30.0, 40.0),
]

_STANDARD_1010_ROWS = (
    "label\tx\ty\tz\n"
    "F5\t-0.5878\t0.8090\t0.0000\n"
    "F6\t0.5878\t0.8090\t0.0000\n"
    "C5\t-0.8090\t0.0000\t0.5878\n"
    "C6\t0.8090\t0.0000\t0.5878\n"
    "P5\t-0.5878\t-0.8090\t0.0000\n"
    "P6\t0.5878\t-0.8090\t0.0000\n"
    "O1\t-0.3090\t-0.9511\t0.0000\n"
    "O2\t0.3090\t-0.9511\t0.0000\n"
)


class EEGGCNNRawDataset(BaseDataset):
    """EEG-GCNN raw EEG dataset pooling TUAB normal-subset and MPI LEMON.

    Processes raw EDF/BrainVision files directly (as opposed to
    :class:`~pyhealth.datasets.EEGGCNNDataset` which uses pre-computed
    FigShare features).

    This dataset supports the EEG-GCNN paper (Wagh & Varatharajah, ML4H @
    NeurIPS 2020) which distinguishes "normal-appearing" patient EEGs (from
    TUAB) from truly healthy EEGs (from MPI LEMON).

    **TUAB (normal subset):** The Temple University EEG Abnormal Corpus
    provides EDF recordings labelled normal/abnormal. Only the *normal*
    recordings are used here — these are the "patient" class (label 0).

    **MPI LEMON:** The Leipzig Study for Mind-Body-Emotion Interactions
    provides BrainVision EEG recordings from healthy controls — these form
    the "healthy" class (label 1).

    Paper:
        Wagh, N. & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
        Electroencephalogram-based Neurological Disease Diagnosis using a
        Domain-guided Graph Convolutional Neural Network. *Proceedings of
        Machine Learning for Health (ML4H) at NeurIPS 2020*, PMLR 136.
        https://proceedings.mlr.press/v136/wagh20a.html

    Authors' code: https://github.com/neerajwagh/eeg-gcnn

    Args:
        root: Root directory containing TUAB and/or LEMON data.
            Expected structure::

                <root>/tuab/train/normal/01_tcp_ar/*.edf
                <root>/tuab/eval/normal/01_tcp_ar/*.edf   (optional)
                <root>/lemon/sub-<ID>/sub-<ID>.vhdr

        dataset_name: Name of the dataset. Defaults to ``"eeg_gcnn"``.
        config_path: Path to the YAML config. Defaults to the built-in
            ``eeg_gcnn_raw.yaml``.
        subset: Which data source(s) to load. One of ``"tuab"``,
            ``"lemon"``, or ``"both"`` (default).
        dev: If ``True``, limit to a small subset for quick iteration.

    Examples:
        >>> from pyhealth.datasets import EEGGCNNRawDataset
        >>> dataset = EEGGCNNRawDataset(root="raw_data/")
        >>> dataset.precompute_features(output_dir="precomputed_data/")
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subset: Optional[str] = "both",
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = (
                Path(__file__).parent / "configs" / "eeg_gcnn_raw.yaml"
            )

        self.root = root

        if subset == "tuab":
            tables = ["tuab"]
        elif subset == "lemon":
            tables = ["lemon"]
        elif subset == "both":
            tables = ["tuab", "lemon"]
        else:
            raise ValueError(
                "subset must be one of 'tuab', 'lemon', or 'both'"
            )

        self.prepare_metadata()

        root_path = Path(root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "eeg_gcnn"

        use_cache = False
        for table in tables:
            shared_csv = root_path / f"eeg_gcnn-{table}-pyhealth.csv"
            cache_csv = cache_dir / f"eeg_gcnn-{table}-pyhealth.csv"
            if not shared_csv.exists() and cache_csv.exists():
                use_cache = True
                break

        if use_cache:
            logger.info("Using cached metadata from %s", cache_dir)
            root = str(cache_dir)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "eeg_gcnn",
            config_path=config_path,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Metadata preparation (file scanning)
    # ------------------------------------------------------------------

    def prepare_metadata(self) -> None:
        """Scan raw files and write metadata CSVs for TUAB and LEMON.

        Writes:
            - ``<root>/eeg_gcnn-tuab-pyhealth.csv``
            - ``<root>/eeg_gcnn-lemon-pyhealth.csv``
        """
        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "eeg_gcnn"

        # --- TUAB normal subset ---
        shared_csv = root / "eeg_gcnn-tuab-pyhealth.csv"
        cache_csv = cache_dir / "eeg_gcnn-tuab-pyhealth.csv"

        if not shared_csv.exists() and not cache_csv.exists():
            tuab_rows = []
            # Support both <root>/train/ and <root>/tuab/train/ layouts.
            tuab_base = root / "tuab" if (root / "tuab").is_dir() else root
            for split in ("train", "eval"):
                normal_dir = tuab_base / split / "normal" / "01_tcp_ar"
                if not normal_dir.is_dir():
                    logger.debug("TUAB normal dir not found: %s", normal_dir)
                    continue
                for edf in sorted(normal_dir.rglob("*.edf")):
                    parts = edf.stem.split("_")
                    patient_id = f"tuab_{parts[0]}"
                    record_id = parts[1] if len(parts) > 1 else "0"
                    tuab_rows.append({
                        "patient_id": patient_id,
                        "record_id": record_id,
                        "signal_file": str(edf),
                        "source": "tuab",
                        "label": 0,
                    })

            if tuab_rows:
                df = pd.DataFrame(tuab_rows)
                df.sort_values(["patient_id", "record_id"], inplace=True,
                               na_position="last")
                df.reset_index(drop=True, inplace=True)
                self._write_csv(df, shared_csv, cache_dir, "tuab")

        # --- LEMON healthy controls ---
        shared_csv = root / "eeg_gcnn-lemon-pyhealth.csv"
        cache_csv = cache_dir / "eeg_gcnn-lemon-pyhealth.csv"

        if not shared_csv.exists() and not cache_csv.exists():
            lemon_rows = []
            lemon_dir = root / "lemon"
            if lemon_dir.is_dir():
                for subject_dir in sorted(lemon_dir.iterdir()):
                    if not subject_dir.is_dir():
                        continue
                    for vhdr in sorted(subject_dir.glob("*.vhdr")):
                        patient_id = f"lemon_{subject_dir.name}"
                        record_id = vhdr.stem
                        lemon_rows.append({
                            "patient_id": patient_id,
                            "record_id": record_id,
                            "signal_file": str(vhdr),
                            "source": "lemon",
                            "label": 1,
                        })

            if lemon_rows:
                df = pd.DataFrame(lemon_rows)
                df.sort_values(["patient_id", "record_id"], inplace=True,
                               na_position="last")
                df.reset_index(drop=True, inplace=True)
                self._write_csv(df, shared_csv, cache_dir, "lemon")

    # ------------------------------------------------------------------
    # Feature precomputation — saves the 5 files EEGGCNNDataset expects
    # ------------------------------------------------------------------

    def precompute_features(
        self,
        output_dir: str,
        sfreq: float = 250.0,
        window_seconds: float = 10.0,
        standard_1010_src: Optional[str] = None,
    ) -> None:
        """Preprocess all raw recordings and save features to *output_dir*.

        Runs the full EEG-GCNN pipeline on every recording found in *root*:
        bipolar montage, resampling, high-pass + notch filtering, 10-second
        windowing, Welch PSD band powers, and pairwise spectral coherence.

        Writes five files consumed by :class:`EEGGCNNDataset`:

        - ``psd_features_data_X``       joblib array ``(N, 48)``
        - ``labels_y``                   joblib array ``(N,)``
        - ``master_metadata_index.csv``  one row per window
        - ``spec_coh_values.npy``        numpy array ``(N, 64)``
        - ``standard_1010.tsv.txt``      electrode coordinates

        Args:
            output_dir: Directory to write the five output files.
            sfreq: Target sampling frequency in Hz. Defaults to 250.0.
            window_seconds: Window length in seconds. Defaults to 10.0.
            standard_1010_src: Optional path to a full
                ``standard_1010.tsv.txt``.  When omitted an 8-row subset is
                written automatically.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        window_samples = int(sfreq * window_seconds)

        all_patient_ids: List[str] = []
        all_X: List[np.ndarray] = []
        all_coh: List[np.ndarray] = []
        all_labels: List[str] = []

        root = Path(self.root)

        # --- collect TUAB recordings ---
        tuab_base = root / "tuab" if (root / "tuab").is_dir() else root
        for split in ("train", "eval"):
            for dir_label, py_label in (("normal", "healthy"), ("abnormal", "diseased")):
                edf_dir = tuab_base / split / dir_label / "01_tcp_ar"
                if not edf_dir.is_dir():
                    continue
                for edf_path in sorted(edf_dir.glob("*.edf")):
                    patient_id = edf_path.stem.split("_")[0]
                    n_before = len(all_X)
                    self._process_recording(
                        edf_path, patient_id, py_label, "tuab",
                        sfreq, window_samples,
                        all_patient_ids, all_X, all_coh, all_labels,
                    )
                    logger.info("[TUAB] %s → %d windows (%s).",
                                edf_path.name, len(all_X) - n_before, py_label)

        # --- collect LEMON recordings ---
        lemon_dir = root / "lemon"
        if lemon_dir.is_dir():
            for sub_dir in sorted(lemon_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                vhdr_path = sub_dir / f"{sub_dir.name}.vhdr"
                if not vhdr_path.exists():
                    continue
                n_before = len(all_X)
                self._process_recording(
                    vhdr_path, sub_dir.name, "healthy", "lemon",
                    sfreq, window_samples,
                    all_patient_ids, all_X, all_coh, all_labels,
                )
                logger.info("[LEMON] %s → %d windows (healthy).",
                            vhdr_path.name, len(all_X) - n_before)

        if not all_X:
            raise RuntimeError(
                "No windows extracted. Check that raw_data/ contains TUAB "
                "and/or LEMON recordings in the expected layout."
            )

        # --- save the five output files ---
        X   = np.vstack(all_X).astype(np.float32)
        coh = np.vstack(all_coh).astype(np.float32)
        y   = np.array(all_labels, dtype=object)

        dump(X, out / "psd_features_data_X")
        dump(y, out / "labels_y")
        np.save(str(out / "spec_coh_values.npy"), coh)
        pd.DataFrame({"patient_ID": all_patient_ids}).to_csv(
            out / "master_metadata_index.csv", index=False
        )

        if standard_1010_src and Path(standard_1010_src).exists():
            shutil.copy2(standard_1010_src, out / "standard_1010.tsv.txt")
        else:
            (out / "standard_1010.tsv.txt").write_text(_STANDARD_1010_ROWS)

        counts = dict(zip(*np.unique(y, return_counts=True)))
        logger.info(
            "Saved %d windows to %s — label counts: %s", len(y), out, counts
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_recording(
        self,
        path: Path,
        patient_id: str,
        label: str,
        source: str,
        sfreq: float,
        window_samples: int,
        patient_ids: List[str],
        X_list: List[np.ndarray],
        coh_list: List[np.ndarray],
        label_list: List[str],
    ) -> None:
        try:
            raw = self._load_raw(path, source, sfreq)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            return

        bipolar = self._make_bipolar(raw, source)
        n_windows = bipolar.shape[1] // window_samples

        for w in range(n_windows):
            start = w * window_samples
            window = bipolar[:, start: start + window_samples]
            patient_ids.append(patient_id)
            X_list.append(self._psd_features(window, sfreq))
            coh_list.append(self._coherence_features(window, sfreq))
            label_list.append(label)

    @staticmethod
    def _load_raw(path: Path, source: str, sfreq: float) -> mne.io.BaseRaw:
        if source == "tuab":
            raw = mne.io.read_raw_edf(str(path), verbose=False, preload=True)
            line_freq = 60.0
        else:
            raw = mne.io.read_raw_brainvision(str(path), verbose=False, preload=True)
            line_freq = 50.0

        raw.pick(picks="eeg")
        if abs(raw.info["sfreq"] - sfreq) > 0.5:
            raw.resample(sfreq=sfreq, verbose=False)
        raw.filter(l_freq=1.0, h_freq=None, verbose=False)
        notch_freqs = np.arange(line_freq, raw.info["sfreq"] / 2, line_freq)
        raw.notch_filter(freqs=notch_freqs, picks="eeg", verbose=False)
        return raw

    @staticmethod
    def _make_bipolar(raw: mne.io.BaseRaw, source: str) -> np.ndarray:
        first_names  = _TUAB_FIRST  if source == "tuab" else _LEMON_FIRST
        second_names = _TUAB_SECOND if source == "tuab" else _LEMON_SECOND

        all_needed = list(dict.fromkeys(first_names + second_names))
        ch_upper = [c.upper() for c in raw.ch_names]
        missing = [n for n in all_needed if n.upper() not in ch_upper]
        if missing:
            raise RuntimeError(f"Channels missing: {missing}")

        name_to_idx = {c.upper(): i for i, c in enumerate(raw.ch_names)}
        data = raw.get_data()
        bipolar = np.zeros((8, data.shape[1]), dtype=np.float32)
        for i, (a, b) in enumerate(zip(first_names, second_names)):
            bipolar[i] = data[name_to_idx[a.upper()]] - data[name_to_idx[b.upper()]]
        return bipolar

    @staticmethod
    def _psd_features(window: np.ndarray, sfreq: float) -> np.ndarray:
        psd, freqs = psd_array_welch(
            window, sfreq=sfreq, fmax=50.0,
            n_per_seg=150, average="mean", verbose=False,
        )
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))
        band_powers = np.zeros((_NUM_NODES, _NUM_BANDS), dtype=np.float32)
        for band_idx, (flo, fhi) in enumerate(_BAND_RANGES):
            mask = freqs <= fhi if flo is None else (freqs >= flo) & (freqs <= fhi)
            band_powers[:, band_idx] = psd_db[:, mask].sum(axis=1)
        return band_powers.flatten()

    @staticmethod
    def _coherence_features(window: np.ndarray, sfreq: float) -> np.ndarray:
        coh_vec = np.zeros(_NUM_EDGES, dtype=np.float32)
        nperseg = int(sfreq)
        for i in range(_NUM_NODES):
            for j in range(_NUM_NODES):
                f, cxy = scipy.signal.coherence(
                    window[i], window[j], fs=sfreq, nperseg=nperseg,
                )
                band_mask = (f >= 1.0) & (f <= 40.0)
                coh_vec[i * _NUM_NODES + j] = float(cxy[band_mask].mean())
        return coh_vec

    @staticmethod
    def _write_csv(
        df: pd.DataFrame,
        shared_path: Path,
        cache_dir: Path,
        table_name: str,
    ) -> None:
        try:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(shared_path, index=False)
            logger.info("Wrote %s metadata to %s", table_name, shared_path)
        except (PermissionError, OSError):
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / shared_path.name
            df.to_csv(cache_path, index=False)
            logger.info("Wrote %s metadata to cache: %s", table_name, cache_path)

    @property
    def default_task(self):
        from pyhealth.tasks import EEGGCNNDiseaseDetection
        return EEGGCNNDiseaseDetection()
