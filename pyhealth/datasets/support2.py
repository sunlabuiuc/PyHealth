import csv
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from pyhealth.tasks.survival_time_to_event import SUPPORT2_FEATURES, _support2_values

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class Support2Dataset(BaseDataset):
    """
    A dataset class for handling SUPPORT2 (Study to Understand Prognoses and Preferences 
    for Outcomes and Risks of Treatments) data.

    The SUPPORT2 dataset contains data on 9,105 seriously ill hospitalized adults from
    five U.S. medical centers (1989-1994), including patient demographics, diagnoses,
    clinical measurements, and outcomes.

    Dataset is available for download from:
        - UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/880/support2
        - Vanderbilt Biostatistics: https://hbiostat.org/data/repo/supportdesc
        - Hugging Face: https://huggingface.co/datasets/jarrydmartinx/support2
        - R packages: "rms" and "Hmisc"

    Citation:
        Knaus WA, Harrell FE, Lynn J, et al. The SUPPORT prognostic model: 
        Objective estimates of survival for seriously ill hospitalized adults. 
        Ann Intern Med. 1995;122(3):191-203.

    Args:
        root (str): The root directory where the dataset CSV file is stored.
        tables (List[str]): A list of tables to be included (typically ["support2"]).
        dataset_name (Optional[str]): The name of the dataset. Defaults to "support2".
        config_path (Optional[str]): The path to the configuration file. If not provided,
            uses the default config.
        **kwargs: Additional arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import Support2Dataset
        >>> dataset = Support2Dataset(
        ...     root="/path/to/support2/data",
        ...     tables=["support2"]
        ... )
        >>> dataset.stats()

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (str): The name of the dataset.
        config_path (str): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "support2.yaml"
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "support2",
            config_path=config_path,
            **kwargs
        )
        return

# Added for Survival MDN by Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html

# The Survival MDN authors' released SUPPORT files store d.time / 365.25;
# their data loader adds 0.001 before training and evaluation.
_DAYS_PER_YEAR = 365.25
SUPPORT2_PAPER_TIME_OFFSET = 0.001
_SPLITS = ("train", "valid", "test")


def support2_paper_time_transform() -> dict:
    """Time conversion used by the Survival MDN authors for SUPPORT2.

    Pass the result to :class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`
    to get times in years (``d.time / 365.25``) plus the 0.001 their loader
    adds, as in their released splits.

    Returns:
        ``{"time_divisor": 365.25, "time_offset": 0.001}``.

    Examples:
        >>> from pyhealth.datasets import support2_paper_time_transform
        >>> support2_paper_time_transform()
        {'time_divisor': 365.25, 'time_offset': 0.001}
    """
    return {"time_divisor": _DAYS_PER_YEAR, "time_offset": SUPPORT2_PAPER_TIME_OFFSET}


def support2_released_split_ids(
    root: str | Path,
    released_dir: str | Path,
    repeat: int,
    tolerance: float = 1e-6,
) -> dict[str, list[str]]:
    """Map one of the authors' released SUPPORT splits to patient numbers.

    The Survival MDN authors released ten train/validation/test splits of
    the 8,873-patient benchmark cohort as ``{train,valid,test}_{repeat}.npz``
    (``arr_2``: the 27 unscaled features, ``arr_1``: ``[years, death]``).
    The files carry no patient IDs, so each released row is matched to the
    SUPPORT2 patient (``sno``) with the same features and death indicator
    and the same time in years. Every cohort patient must be used exactly
    once across the three files. All ten released repeats map one-to-one.

    The released files come from the authors' repository, which has no
    license; download them yourself. This function only reads local files.

    Args:
        root: Directory containing ``support2.csv`` with an ``sno`` column,
            as :class:`Support2Dataset` reads it.
        released_dir: Directory containing the released ``.npz`` files.
        repeat: Split number, 1 to 10 in the authors' release.
        tolerance: Largest allowed relative difference for features, and
            absolute difference for times in years. Default 1e-6.

    Returns:
        ``{"train": [...], "valid": [...], "test": [...]}`` patient numbers
        as strings, in each file's row order.

    Raises:
        ValueError: If a released row matches no unused patient, or the
            three files do not cover the cohort exactly once.

    Examples:
        >>> from pyhealth.datasets import support2_released_split_ids
        >>> ids = support2_released_split_ids(  # doctest: +SKIP
        ...     "/data/support2", "/data/Survival-MDN/data/support", repeat=1
        ... )
        >>> [len(ids[s]) for s in ("train", "valid", "test")]  # doctest: +SKIP
        [5370, 1776, 1727]
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if (
        isinstance(repeat, bool)
        or not isinstance(repeat, (int, np.integer))
        or not 1 <= repeat <= 10
    ):
        raise ValueError("repeat must be an integer from 1 through 10")
    path = Path(root).expanduser() / "support2.csv"
    ids, features, outcomes = [], [], []
    seen = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if "sno" not in row:
                raise ValueError("support2.csv needs an sno column")
            source_id = str(row["sno"]).strip()
            if not source_id or source_id in seen:
                raise ValueError("SUPPORT2 sno values must be nonempty and unique")
            seen.add(source_id)
            parsed = _support2_values(row, row["sno"])
            if parsed is not None:
                ids.append(str(row["sno"]).strip())
                features.append(parsed[0])
                outcomes.append((parsed[1] / _DAYS_PER_YEAR, parsed[2]))
    features = np.asarray(features, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    # Index by float32 features and event for speed; check times exactly.
    index: dict[tuple, list[int]] = {}
    for i, (row, event) in enumerate(zip(features.astype(np.float32), outcomes[:, 1])):
        index.setdefault((row.tobytes(), event), []).append(i)
    used = np.zeros(len(ids), dtype=bool)
    memberships = {}
    for split in _SPLITS:
        file = Path(released_dir).expanduser() / f"{split}_{repeat}.npz"
        with np.load(file, allow_pickle=False) as data:
            x = np.asarray(data["arr_2"], dtype=float)
            y = np.asarray(data["arr_1"], dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(SUPPORT2_FEATURES)
            or y.shape != (len(x), 2)
        ):
            raise ValueError(f"{file.name}: expected arr_2 [N, 27] and arr_1 [N, 2]")
        members = []
        for r in range(len(x)):
            key = (x[r].astype(np.float32).tobytes(), y[r, 1])
            candidates = [i for i in index.get(key, []) if not used[i]]
            if not candidates:  # fall back to a tolerance search
                close = (
                    ~used
                    & (outcomes[:, 1] == y[r, 1])
                    & (
                        np.abs(features - x[r])
                        <= tolerance * np.maximum(1.0, np.abs(x[r]))
                    ).all(axis=1)
                )
                candidates = list(np.flatnonzero(close))
            candidates = [
                i for i in candidates if abs(outcomes[i, 0] - y[r, 0]) <= tolerance
            ]
            if not candidates:
                raise ValueError(f"{file.name} row {r}: no matching SUPPORT2 patient")
            used[candidates[0]] = True
            members.append(ids[candidates[0]])
        memberships[split] = members
    if not used.all():
        raise ValueError(
            f"Released repeat {repeat} leaves {int((~used).sum())} cohort patients unused"
        )
    return memberships
