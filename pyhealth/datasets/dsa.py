"""
PyHealth dataset for the UCI Daily and Sports Activities (DSA) dataset.

Dataset link:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

Dataset paper: (please cite if you use this dataset)
    Kerem Altun, Billur Barshan, and Orkun Tunçel. "Comparative Study on
    Classifying Human Activities with Miniature Inertial and Magnetic Sensors."
    Pattern Recognition 43(10): 3605-3620, 2010.

Dataset paper link:
    https://doi.org/10.1016/j.patcog.2010.04.019

Author:
    Edward Guan (edwardg2@illinois.edu)
"""

import logging
import os
import random
import zipfile
from pathlib import Path
from typing import List, Optional
import urllib.request

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class DSADataset(BaseDataset):
    """Dataset class for the UCI Daily and Sports Activities (DSA) dataset.

    The dataset contains motion sensor data of 19 daily and sports activities,
    each performed by 8 subjects for 5 minutes. Five Xsens MTx sensor units
    are placed on the torso, right arm, left arm, right leg, and left leg.
    Each unit records 9-channel data (x/y/z accelerometer, gyroscope, and
    magnetometer) at 25 Hz, segmented into 5-second (125-timestep) windows.

    The dataset is structured to support multi-source transfer learning: all
    five sensor domains are recorded simultaneously, so every segment across
    domains is paired by the same subject performing the same activity at the
    same moment. This pairwise structure is preserved in the ``pair_id`` field
    of each indexed row.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        target_domain (str): Sensor placement used as the classification target.
        activities (List[str]): Ordered list of activity names (1-indexed).
        domains (List[str]): Ordered list of sensor domain keys.
        domain_full_names (dict): Mapping from domain key to descriptive name.

    Example::

        >>> dataset = DSADataset(root="./data/DSA")
        >>> print(len(dataset))   # number of indexed rows
        >>> task_ds = dataset.set_task()
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    activities: List[str] = [
        "sitting",
        "standing",
        "lying_back",
        "lying_right",
        "ascending_stairs",
        "descending_stairs",
        "elevator_standing",
        "elevator_moving",
        "walking_parking_lot",
        "walking_treadmill_flat",
        "walking_treadmill_inclined",
        "running",
        "stepper",
        "cross_trainer",
        "cycling_horizontal",
        "cycling_vertical",
        "rowing",
        "jumping",
        "basketball",
    ]

    domains: List[str] = ["T", "RA", "LA", "RL", "LL"]

    domain_full_names: dict = {
        "T":  "Torso",
        "RA": "Right Arm",
        "LA": "Left Arm",
        "RL": "Right Leg",
        "LL": "Left Leg",
    }

    # Column slices within each 45-column row (0-indexed, end exclusive)
    _domain_cols: dict = {
        "T":  (0,  9),
        "RA": (9,  18),
        "LA": (18, 27),
        "RL": (27, 36),
        "LL": (36, 45),
    }

    _N_ACTIVITIES = 19
    _N_SUBJECTS   = 8
    _N_SEGMENTS   = 60
    _N_CHANNELS   = 9    # per sensor unit
    _N_TIMESTEPS  = 125  # 5 sec at 25 Hz

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(Path(__file__).parent / "configs" / "dsa.yaml"),
        download: bool = False,
        target_domain: str = "LA",
        scale: bool = True,
        **kwargs,
    ) -> None:
        """Initialises the DSA dataset.

        Args:
            root (str): Root directory of the raw data. Must contain folders
                ``a01`` through ``a19`` after download or manual extraction.
                Defaults to the working directory.
            config_path (Optional[str]): Path to a PyHealth YAML configuration
                file. Defaults to "../configs/dsa.yaml"
            download (bool): If ``True``, download and extract the dataset from
                the UCI ML Repository into ``root``. Defaults to ``False``.
            target_domain (str): Sensor domain treated as the target for
                classification. Must be one of ``["T", "RA", "LA", "RL", "LL"]``.
                Defaults to ``"LA"`` (Left Arm, simulating a wrist wearable).
            scale (bool): If ``True``, apply per-channel min-max scaling to
                ``[-1, 1]`` when loading time series arrays. Defaults to ``True``.

        Raises:
            ValueError: If ``target_domain`` is not a valid domain key.
            FileNotFoundError: If ``root`` does not exist or lacks ``a01``.
            FileNotFoundError: If any expected segment file is missing.

        Example::

            >>> dataset = DSADataset(root="./data/DSA", target_domain="LA")
        """
        if target_domain not in self.domains:
            raise ValueError(
                f"target_domain must be one of {self.domains}, "
                f"got '{target_domain}'."
            )

        self.target_domain = target_domain
        self.scale = scale
        self._metadata_path = os.path.join(root, "dsa-metadata-pyhealth.csv")

        if download:
            self._download(root)

        self._verify_data(root)
        self._index_data(root)

        super().__init__(
            root=root,
            tables=["dsa"],
            dataset_name="DSA",
            config_path=config_path,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            DSAActivityClassification: The default classification task using the target domain time series.

        Example::

            >>> dataset = DSADataset(root="./data/DSA")
            >>> task = dataset.default_task
        """
        # Import here to avoid circular imports
        from pyhealth.tasks.dsa import DSAActivityClassification
        return DSAActivityClassification()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self, root: str) -> None:
        """Downloads and extracts the DSA dataset from the UCI ML Repository.

        Downloads the zip archive (~163 MB), extracts it into ``root``, and
        removes the archive afterwards.

        Args:
            root (str): Destination directory for the extracted dataset.

        Raises:
            FileNotFoundError: If extraction produces no ``a01`` folder.
        """
        os.makedirs(root, exist_ok=True)
        url = (
            "https://archive.ics.uci.edu/static/public/256/"
            "daily+and+sports+activities.zip"
        )
        zip_path = os.path.join(root, "dsa.zip")

        logger.info(f"Downloading DSA dataset from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        logger.info("Download complete. Extracting ...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate paths before extraction (safety check)
            for member in zf.namelist():
                member_path = os.path.realpath(os.path.join(root, member))
                if not member_path.startswith(os.path.realpath(root)):
                    raise ValueError(
                        f"Unsafe path detected in zip: '{member}'"
                    )
            zf.extractall(root)

        os.remove(zip_path)
        logger.info("Extraction complete.")

        # The zip may contain a top-level subfolder — move contents up if so
        extracted_dirs = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.startswith("data")
        ]
        if extracted_dirs and not os.path.isdir(os.path.join(root, "a01")):
            inner = os.path.join(root, extracted_dirs[0])
            for item in os.listdir(inner):
                os.rename(os.path.join(inner, item), os.path.join(root, item))
            os.rmdir(inner)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_data(self, root: str) -> None:
        """Verifies the presence and structure of the dataset directory.

        Checks that ``root`` exists and contains the expected activity folders
        ``a01`` through ``a19``, each with 8 subject subdirectories containing
        60 segment files.

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If ``root`` does not exist.
            FileNotFoundError: If the expected folder ``a01`` is missing.
        """
        if not os.path.exists(root):
            msg = (
                f"Dataset root '{root}' does not exist. "
                "Pass download=True to download it automatically."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        expected_dir = os.path.join(root, "a01")
        if not os.path.isdir(expected_dir):
            msg = (
                f"Expected activity folder '{expected_dir}' not found. "
                "Ensure 'root' points to the directory containing a01..a19."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Dataset structure verified.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_data(self, root: str) -> pd.DataFrame:
        """Parses the dataset directory structure into a metadata index.

        Walks all activity, subject, and segment folders to build a flat
        DataFrame where each row represents one segment file. No time series
        data is loaded at this stage — only file paths and identifiers.

        The ``pair_id`` column encodes the pairwise synchronisation structure:
        all five domains share the same ``pair_id`` for a given
        (activity, segment) combination, regardless of subject. This field is
        required for Inter-domain Pairwise Distance (IPD) computation.

        Args:
            root (str): Root directory of the raw data.

        Returns:
            pd.DataFrame: Metadata index saved to ``dsa-metadata-pyhealth.csv``.

        Raises:
            FileNotFoundError: If any expected segment file is missing.
        """
        rows = []

        for activity_idx in range(1, self._N_ACTIVITIES + 1):
            activity_folder = f"a{activity_idx:02d}"
            activity_name   = self.activities[activity_idx - 1]

            for subject_id in range(1, self._N_SUBJECTS + 1):
                subject_folder = f"p{subject_id}"
                subject_dir = os.path.join(root, activity_folder, subject_folder)

                if not os.path.isdir(subject_dir):
                    raise FileNotFoundError(
                        f"Expected subject directory not found: {subject_dir}"
                    )

                for segment_id in range(1, self._N_SEGMENTS + 1):
                    filename = f"s{segment_id:02d}.txt"
                    filepath = os.path.join(subject_dir, filename)

                    if not os.path.isfile(filepath):
                        raise FileNotFoundError(
                            f"Expected segment file not found: {filepath}"
                        )

                    rows.append({
                        # --- PyHealth standard fields ---
                        "patient_id":    f"p{subject_id}",
                        "visit_id":      f"a{activity_idx:02d}_p{subject_id}_s{segment_id:02d}",

                        # --- DSA-specific fields ---
                        "activity_id":   activity_idx,
                        "activity_name": activity_name,
                        # 0-indexed label for model output layers
                        "label":         activity_idx - 1,
                        "segment_id":    segment_id,
                        "filepath":      filepath,

                        # pair_id links the same (activity, segment) across all
                        # domains and subjects — the foundation of IPD computation
                        "pair_id":       f"a{activity_idx:02d}_s{segment_id:02d}",
                    })

        df = pd.DataFrame(rows)
        df.to_csv(self._metadata_path, index=False)
        logger.info(
            f"Indexed {len(df):,} segment files → {self._metadata_path}"
        )
        return df

    # ------------------------------------------------------------------
    # Time series loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_segment(filepath: str) -> np.ndarray:
        """Load one segment file into a (125, 45) float32 array.

        Args:
            filepath (str): Path to a ``s{segment}.txt`` file.

        Returns:
            np.ndarray: Shape ``(125, 45)``, dtype ``float32``.
        """
        return np.loadtxt(filepath, delimiter=",", dtype=np.float32)

    def _slice_domain(self, raw: np.ndarray, domain: str) -> np.ndarray:
        """Extract one domain's channels from a raw segment array.

        Args:
            raw (np.ndarray): Shape ``(125, 45)`` full segment array.
            domain (str): Domain key, one of ``["T", "RA", "LA", "RL", "LL"]``.

        Returns:
            np.ndarray: Shape ``(9, 125)`` — channels × timesteps.
        """
        start, end = self._domain_cols[domain]
        # raw[:, start:end] is (125, 9); transpose to (9, 125)
        return raw[:, start:end].T

    @staticmethod
    def _minmax_scale(ts: np.ndarray) -> np.ndarray:
        """Scale each channel of a ``(K, T)`` array independently to ``[-1, 1]``.

        Channels with zero range (flat signal) are left as zeros.

        Args:
            ts (np.ndarray): Shape ``(K, T)``.

        Returns:
            np.ndarray: Shape ``(K, T)``, values in ``[-1, 1]``.
        """
        scaled = np.zeros_like(ts)
        for k in range(ts.shape[0]):
            mn, mx = ts[k].min(), ts[k].max()
            if mx > mn:
                scaled[k] = 2.0 * (ts[k] - mn) / (mx - mn) - 1.0
        return scaled

    def load_time_series(
        self,
        filepath: str,
        domain: Optional[str] = None,
    ) -> dict:
        """Load and preprocess all domain time series from one segment file.

        This is the primary method for retrieving time series data. It is
        called by task functions when building model-ready samples.

        Args:
            filepath (str): Path to the segment ``.txt`` file.
            domain (Optional[str]): If provided, return only this domain's
                array. If ``None``, return all five domains.

        Returns:
            dict: Mapping ``{domain_key: np.ndarray (9, 125)}``.
                  If ``domain`` is specified, the dict has one entry.
                  All arrays are scaled to ``[-1, 1]`` if ``self.scale=True``.
        """
        raw = self._load_segment(filepath)
        domains_to_load = [domain] if domain else self.domains

        result = {}
        for d in domains_to_load:
            ts = self._slice_domain(raw, d)
            if self.scale:
                ts = self._minmax_scale(ts)
            result[d] = ts

        return result

    # ------------------------------------------------------------------
    # Subject-level split utilities
    # ------------------------------------------------------------------

    def get_subject_split(
        self,
        train_subjects: List[int],
        test_subjects: List[int],
    ) -> tuple:
        """Return metadata DataFrames split by subject.

        Args:
            train_subjects (List[int]): Subject IDs (1–8) for training.
            test_subjects (List[int]): Subject IDs (1–8) for testing.

        Returns:
            tuple: ``(train_df, test_df)`` as pandas DataFrames.

        Raises:
            ValueError: If train and test subject sets overlap.

        Example::

            >>> dataset = DSADataset(root="./data/DSA")
            >>> train_df, test_df = dataset.get_subject_split(
            ...     train_subjects=[1,2,3,4,5,6],
            ...     test_subjects=[7,8],
            ... )
        """
        if set(train_subjects) & set(test_subjects):
            raise ValueError("train_subjects and test_subjects must not overlap.")

        df = pd.read_csv(self._metadata_path)
        train_ids = {f"p{s}" for s in train_subjects}
        test_ids  = {f"p{s}" for s in test_subjects}

        train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
        test_df  = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

        return train_df, test_df

    def random_subject_splits(
        self,
        n_repeats: int = 15,
        n_train: int = 6,
        random_seed: int = 0,
    ):
        """Generator yielding repeated random train/test subject splits.

        Replicates the paper's evaluation protocol: randomly choose ``n_train``
        of 8 subjects for training, reserve the rest for testing, and repeat
        ``n_repeats`` times. Report mean ± std of the metric across repeats.

        Args:
            n_repeats (int): Number of random repetitions. Paper uses 15.
            n_train (int): Number of training subjects. Paper uses 6.
            random_seed (int): Base random seed; repeat ``i`` uses
                ``random_seed + i`` for reproducibility.

        Yields:
            tuple: ``(repeat_idx, train_subjects, test_subjects,
                      train_df, test_df)``

        Example::

            >>> dataset = DSADataset(root="./data/DSA")
            >>> results = []
            >>> for i, train_sub, test_sub, train_df, test_df in \\
            ...         dataset.random_subject_splits(n_repeats=15):
            ...     rcc = run_experiment(train_df, test_df)
            ...     results.append(rcc)
            >>> print(f"{np.mean(results):.4f} ± {np.std(results):.4f}")
        """
        all_subjects = list(range(1, self._N_SUBJECTS + 1))
        for i in range(n_repeats):
            rng = random.Random(random_seed + i)
            shuffled = all_subjects.copy()
            rng.shuffle(shuffled)
            train_subjects = shuffled[:n_train]
            test_subjects  = shuffled[n_train:]
            train_df, test_df = self.get_subject_split(train_subjects, test_subjects)
            yield i, train_subjects, test_subjects, train_df, test_df