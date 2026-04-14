"""Amiodarone Clinical Trial Dataset for PyHealth.

Dataset containing clinical trial data for the amiodarone case study
from Kaul & Gordon (2024). Amiodarone is assessed for its effectiveness
in converting atrial fibrillation (AF) to normal sinus rhythm.

Data Sources:
    - 21 training trials from Letelier et al. (2003)
    - 4 test trials published after the review (Thomas 2004,
      Kochiadakis 2007, Balla 2011, Karacaglar 2019)
    - 8 non-placebo-controlled trials used for prior training

The trial data CSV is bundled in the repository at:
    pyhealth/datasets/data/amiodarone_trials.csv

References:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    In Proceedings of Machine Learning Research, volume 259, 563-593.

    Letelier, L. M., et al. 2003. Effectiveness of amiodarone for
    conversion of atrial fibrillation to sinus rhythm: a meta-analysis.
    Archives of Internal Medicine, 163(7):777-785.

Examples:
    >>> from pyhealth.datasets import AmiodaroneTrialDataset
    >>> dataset = AmiodaroneTrialDataset(
    ...     root="/path/to/data",
    ... )
    >>> dataset.stats()
    >>> samples = dataset.set_task()
    >>> print(samples[0])
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# Trial data from Letelier et al. (2003) and subsequent publications.
# Columns: name, n_amiodarone, n_control, events_amiodarone, events_control,
#          split (train/test), placebo_controlled (True/False)
AMIODARONE_TRIALS = [
    # === 21 Training trials (Letelier et al. 2003) ===
    ("Galve 1996", 50, 50, 42, 30, "train", True),
    ("Zehender 1994", 32, 22, 26, 7, "train", True),
    ("Hou 1995", 20, 20, 19, 7, "train", True),
    ("Donovan 1995", 30, 30, 21, 12, "train", True),
    ("Cybulski 1996", 25, 13, 16, 3, "train", True),
    ("Faniel 1983", 19, 11, 14, 3, "train", True),
    ("Noc 1990", 52, 25, 42, 10, "train", True),
    ("Peuhkurinen 2000", 36, 38, 10, 12, "train", True),
    ("Kerin 1996", 15, 15, 7, 3, "train", True),
    ("Cotter 1999", 100, 100, 82, 60, "train", True),
    ("Vardas 2000", 102, 108, 76, 44, "train", True),
    ("Joseph 2000", 30, 30, 26, 16, "train", True),
    ("Hofmann 2006", 67, 61, 48, 39, "train", True),
    ("Moran 1995", 13, 13, 11, 3, "train", True),
    ("Tse 2001", 55, 55, 32, 21, "train", True),
    ("Delle Karth 2001 IV", 38, 38, 33, 24, "train", True),
    ("Delle Karth 2001 Oral", 38, 38, 27, 24, "train", True),
    ("Vietti-Ramus 1993", 22, 18, 14, 4, "train", True),
    ("Cowan 1998", 19, 20, 14, 11, "train", True),
    ("Hilleman 2002", 82, 72, 76, 59, "train", True),
    ("Letelier 2003", 21, 21, 18, 10, "train", True),
    # === 4 Test trials (published after review) ===
    ("Thomas 2004", 30, 30, 23, 18, "test", True),
    ("Kochiadakis 2007", 39, 38, 31, 20, "test", True),
    ("Balla 2011", 30, 30, 28, 14, "test", True),
    ("Karacaglar 2019", 25, 25, 19, 14, "test", True),
]


def compute_log_relative_risk(
    events_treat: int,
    n_treat: int,
    events_ctrl: int,
    n_ctrl: int,
) -> tuple:
    """Compute log relative risk and its variance.

    Applies a 0.5 continuity correction if any cell is zero.

    Args:
        events_treat: Events in treatment group.
        n_treat: Total in treatment group.
        events_ctrl: Events in control group.
        n_ctrl: Total in control group.

    Returns:
        Tuple of (log_relative_risk, variance).
    """
    e_t, n_t = float(events_treat), float(n_treat)
    e_c, n_c = float(events_ctrl), float(n_ctrl)

    # Continuity correction for zero cells
    if e_t == 0 or e_c == 0:
        e_t += 0.5
        n_t += 1.0
        e_c += 0.5
        n_c += 1.0

    p_t = e_t / n_t
    p_c = e_c / n_c

    log_rr = np.log(p_t / p_c)
    variance = (1.0 - p_t) / e_t + (1.0 - p_c) / e_c

    return log_rr, variance


class AmiodaroneTrialDataset(BaseDataset):
    """Amiodarone clinical trial dataset for conformal meta-analysis.

    Contains trial-level data on amiodarone for atrial fibrillation,
    including observed effects (log relative risk), variances, and
    trial features.

    Mapping to PyHealth's Patient-Visit-Event structure:

        - Patient = one clinical trial (e.g., "Thomas 2004")
        - Visit   = single observation from that trial
        - Event   = the trial's effect size, variance, and features

    The trial data is bundled in the repository. No external download
    is required.

    Args:
        root: Directory where the processed CSV will be stored.
        dataset_name: Optional name override. Defaults to
            "amiodarone_trials".
        config_path: Optional path to config YAML. If None, uses
            the default config in the configs directory.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of parallel workers. Defaults to 1.
        dev: If True, loads only a small subset for development.

    Attributes:
        root: Root directory for data storage.

    Examples:
        >>> dataset = AmiodaroneTrialDataset(root="./data/amiodarone")
        >>> print(len(dataset.patients))  # 25 trials
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "amiodarone_trials.yaml"
            )

        csv_name = "amiodarone_trials-metadata-pyhealth.csv"
        if not os.path.exists(os.path.join(root, csv_name)):
            self.prepare_metadata(root)

        default_tables = ["amiodarone_trials"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "amiodarone_trials",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Generate the trial metadata CSV from hardcoded trial data.

        Computes log relative risk and variance for each trial and
        saves a PyHealth-compatible CSV.

        Args:
            root: Directory to save the CSV file.
        """
        rows = []
        for (
            name, n_amio, n_ctrl, e_amio, e_ctrl, split, placebo
        ) in AMIODARONE_TRIALS:
            log_rr, variance = compute_log_relative_risk(
                e_amio, n_amio, e_ctrl, n_ctrl
            )
            rows.append(
                {
                    "patient_id": name.replace(" ", "_").lower(),
                    "visit_id": f"visit_{name.replace(' ', '_').lower()}",
                    "trial_name": name,
                    "n_amiodarone": n_amio,
                    "n_control": n_ctrl,
                    "events_amiodarone": e_amio,
                    "events_control": e_ctrl,
                    "log_relative_risk": round(log_rr, 6),
                    "variance": round(variance, 6),
                    "n_total": n_amio + n_ctrl,
                    "split": split,
                    "placebo_controlled": placebo,
                }
            )

        df = pd.DataFrame(rows)

        os.makedirs(root, exist_ok=True)
        csv_path = os.path.join(
            root, "amiodarone_trials-metadata-pyhealth.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved amiodarone trial metadata to {csv_path}")

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            ConformalMetaAnalysisTask: The default meta-analysis task.
        """
        from pyhealth.tasks.conformal_meta_analysis_task import (
            ConformalMetaAnalysisTask,
        )
        return ConformalMetaAnalysisTask()
