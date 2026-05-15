# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Paper link: https://arxiv.org/abs/2408.07773
# Description: BIDMC dataset — 53 ICU patients with 8-minute
#     recordings of ECG, PPG, and respiratory signals at 125 Hz.
#     Two annotators manually annotated individual breaths.
# Source: https://physionet.org/content/bidmc/1.0.0/

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.datasets._medtsllm_cache import (
    compute_fingerprint,
    load_or_build,
)

logger = logging.getLogger(__name__)

# Paper's BIDMC split: 85/15 by patient, np.random.RandomState(0).
_PAPER_SPLIT_RATIO = 0.85
_PAPER_SPLIT_SEED = 0

# Subdirectory under ``root`` for preprocessed ``.npz`` caches.
_PROCESSED_SUBDIR = "processed"

# RESP, PLETH, and ECG lead II: the 3 channels used in the paper.
_TARGET_CHANNELS = ["RESP,", "PLETH,", "II,"]


class BIDMCDataset(BaseDataset):
    """BIDMC respiratory signal dataset for boundary detection.

    53 ICU patients with 8-minute recordings of ECG, PPG, and
    respiratory impedance signals at 125 Hz. Breath boundaries are
    manually annotated by two annotators.

    Dataset is available at https://physionet.org/content/bidmc/1.0.0/

    Paper: Pimentel, M.A.F. et al. "Towards a Robust Estimation of
    Respiratory Rate from Pulse Oximeters." IEEE TBME, 2016.

    Args:
        root: Root directory of the raw BIDMC data. Should contain
            wfdb record files (bidmc01.dat, bidmc01.hea, etc.).
        dataset_name: Name of the dataset. Default is ``"bidmc"``.
        config_path: Path to the YAML config file.
        dev: Whether to enable dev mode (first 5 patients).
        paper_split: If True, populate a ``split`` column with an
            85/15 train/test assignment per patient, using the
            legacy NumPy seed=0 RNG from the paper. Otherwise the
            ``split`` column is left blank. Default False.
        preprocess: If True, decode each record once, extract the
            RESP/PLETH/II channels, and cache
            ``(signal, ann_sample, ann_aux)`` to
            ``{root}/processed/{record}.npz``. Subsequent runs skip
            wfdb. Default False.

    Examples:
        >>> from pyhealth.datasets import BIDMCDataset
        >>> dataset = BIDMCDataset(root="/path/to/bidmc/")
        >>> dataset.stat()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        paper_split: bool = False,
        preprocess: bool = False,
    ) -> None:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "bidmc.yaml"
            )

        metadata_path = os.path.join(root, "bidmc-pyhealth.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(
                root,
                dev=dev,
                paper_split=paper_split,
                preprocess=preprocess,
            )

        super().__init__(
            root=root,
            tables=["respiratory"],
            dataset_name=dataset_name or "bidmc",
            config_path=config_path,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(
        root: str,
        dev: bool = False,
        paper_split: bool = False,
        preprocess: bool = False,
    ) -> None:
        """Prepare metadata CSV from raw wfdb files.

        Args:
            root: Root directory containing wfdb files.
            dev: If True, only process first 5 patients.
            paper_split: If True, assign records to train/test via the
                paper's 85/15 seed=0 split (written to the ``split``
                column). Otherwise the column is blank.
            preprocess: If True, build per-record ``.npz`` caches and
                populate the ``processed_file`` column.
        """
        import wfdb

        # Find record files (exclude numerics *n.hea)
        hea_files = sorted(
            f.replace(".hea", "")
            for f in os.listdir(root)
            if f.endswith(".hea")
            and not f.endswith("n.hea")
            and f.startswith("bidmc")
        )
        if dev:
            hea_files = hea_files[:5]

        split_by_record = (
            _assign_paper_split(hea_files) if paper_split else {}
        )
        processed_dir = os.path.join(root, _PROCESSED_SUBDIR)

        rows = []
        for rec_name in hea_files:
            rec_path = os.path.join(root, rec_name)
            try:
                record = wfdb.rdrecord(rec_path)
            except Exception:
                continue

            patient_id = rec_name.replace("bidmc", "")

            # Parse header for demographics
            age, sex, location = "", "", ""
            if record.comments:
                comment = " ".join(record.comments)
                for field, var in [("age", "age"), ("sex", "sex"),
                                   ("location", "location")]:
                    tag = f"<{field}>:"
                    if tag in comment:
                        idx = comment.index(tag) + len(tag)
                        end = comment.find("<", idx)
                        val = (comment[idx:end].strip()
                               if end > 0 else comment[idx:].strip())
                        if field == "age":
                            age = val
                        elif field == "sex":
                            sex = val
                        elif field == "location":
                            location = val

            processed_file = ""
            if preprocess:
                processed_file = _build_record_cache(
                    processed_dir=processed_dir,
                    rec_path=os.path.join(root, rec_name),
                    rec_name=rec_name,
                )

            rows.append({
                "patient_id": patient_id,
                "signal_file": os.path.join(root, rec_name),
                "annotation_file": "breath",
                "age": age,
                "sex": sex,
                "location": location,
                "split": split_by_record.get(rec_name, ""),
                "processed_file": processed_file,
            })

        df = pd.DataFrame(rows)
        out_path = os.path.join(root, "bidmc-pyhealth.csv")
        df.to_csv(out_path, index=False)
        logger.info(
            "BIDMC metadata: %d patients -> %s", len(df), out_path
        )

    @property
    def default_task(self):
        """Returns the default task for this dataset."""
        from pyhealth.tasks.respiratory_boundary_detection import (
            RespiratoryBoundaryDetection,
        )

        return RespiratoryBoundaryDetection()


def _build_record_cache(
    processed_dir: str,
    rec_path: str,
    rec_name: str,
) -> str:
    """Cache (signal, ann_sample, ann_aux) for one BIDMC record.

    Signal is the 3-channel (RESP, PLETH, II) array at native 125 Hz
    — no downsampling, no trim, matching the paper's recipe.
    ``ann_sample`` / ``ann_aux`` preserve the raw annotation stream
    so the task can pick annotator 1 or 2 at windowing time.
    """
    cache_path = os.path.join(processed_dir, f"{rec_name}.npz")
    raw_paths = [
        rec_path + ".dat",
        rec_path + ".hea",
        rec_path + ".breath",
    ]
    fingerprint = compute_fingerprint(raw_paths, {"channels": _TARGET_CHANNELS})

    def _build() -> dict[str, np.ndarray]:
        import wfdb

        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, extension="breath")

        col_idx = [
            record.sig_name.index(ch)
            for ch in _TARGET_CHANNELS
            if ch in record.sig_name
        ]
        signal = record.p_signal[:, col_idx].astype(np.float32)

        return {
            "signal": signal,
            "ann_sample": np.asarray(ann.sample, dtype=np.int64),
            "ann_aux": np.asarray(ann.aux_note).astype("U8"),
        }

    load_or_build(cache_path, fingerprint, _build)
    return cache_path


def _assign_paper_split(records: list[str]) -> dict[str, str]:
    """Assign each record to train/test per the paper's 85/15 seed=0 split.

    Mirrors the cs598 BIDMCSegmentationDataset recipe: shuffle record
    names with ``np.random.RandomState(0)``, take the first 85% as
    train and the remainder as test.
    """
    rng = np.random.RandomState(_PAPER_SPLIT_SEED)
    order = rng.permutation(len(records))
    cutoff = int(len(records) * _PAPER_SPLIT_RATIO)
    assignment: dict[str, str] = {}
    for rank, idx in enumerate(order):
        assignment[records[idx]] = "train" if rank < cutoff else "test"
    return assignment
