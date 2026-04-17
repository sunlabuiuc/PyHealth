"""PMLB Meta-Analysis Dataset for PyHealth.

Dataset for conformal meta-analysis experiments using regression
datasets from the Penn Machine Learning Benchmark (PMLB).

PMLB is freely accessible at:
    https://github.com/EpistasisLab/pmlb
    Install: pip install pmlb

The three datasets used in Kaul & Gordon (2024) are:
    - 1196_BNG_pharynx
    - 1201_BNG_breastTumor
    - 1193_BNG_lowbwt

These regression datasets provide features X and target values Y
(used as true effects U in the meta-analysis simulations). When
``synthesize_noise=True``, synthetic observed effects Y, within-trial
variances V, and prior means M are generated at CSV creation time
according to controlled parameters ``prior_error`` and
``effect_noise``, following Appendix B.6 of the paper. When False
(default), the dataset is a plain regression dataset with just
features and the true effect.

Reference:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    In Proceedings of Machine Learning Research, volume 259, 563-593.

    Olson, R. S., La Cava, W., Orzechowski, P., Urbanowicz, R. J.,
    and Moore, J. H. 2017. PMLB: a large benchmark suite for machine
    learning evaluation and comparison. BioData Mining, 10:1-13.

Examples:
    >>> from pyhealth.datasets import PMLBMetaAnalysisDataset
    >>> # Plain regression dataset
    >>> dataset = PMLBMetaAnalysisDataset(
    ...     root="./data/pmlb",
    ...     pmlb_dataset_name="1196_BNG_pharynx",
    ... )
    >>> # With synthetic meta-analysis noise
    >>> dataset = PMLBMetaAnalysisDataset(
    ...     root="./data/pmlb",
    ...     pmlb_dataset_name="1196_BNG_pharynx",
    ...     synthesize_noise=True,
    ...     prior_error=0.9,
    ...     effect_noise=0.5,
    ...     seed=42,
    ... )
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

SUPPORTED_PMLB_DATASETS = [
    "1196_BNG_pharynx",
    "1201_BNG_breastTumor",
    "1193_BNG_lowbwt",
]


class PMLBMetaAnalysisDataset(BaseDataset):
    """PMLB regression dataset for conformal meta-analysis.

    Each row = one simulated trial. Mapping to PyHealth's structure:
        - Patient = one data point
        - Visit   = single observation
        - Event   = features + target (+ optional Y, V, M)

    Args:
        root: Directory for processed CSV.
        pmlb_dataset_name: One of the three supported PMLB datasets.
        dataset_name: Optional override for the dataset name.
        config_path: Optional path to the YAML config.
        cache_dir: Optional cache directory.
        num_workers: Parallel workers (default 1).
        dev: Load only a small subset if True.
        synthesize_noise: If True, add observed_effect, variance,
            and prior_mean columns during CSV creation for
            meta-analysis simulations. Defaults to False.
        prior_error: Prior quality parameter. 0 = perfect,
            higher = worse. Only used when synthesize_noise=True.
        effect_noise: Noise scale. 0 = no noise, higher = more.
            Only used when synthesize_noise=True.
        seed: Random seed. Only used when synthesize_noise=True.

    Examples:
        >>> dataset = PMLBMetaAnalysisDataset(
        ...     root="./data/pmlb",
        ...     pmlb_dataset_name="1196_BNG_pharynx",
        ...     synthesize_noise=True,
        ... )
    """

    def __init__(
            self,
            root: str,
            pmlb_dataset_name: str = "1196_BNG_pharynx",
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
            cache_dir: Optional[str] = None,
            num_workers: int = 1,
            dev: bool = False,
            synthesize_noise: bool = False,
            prior_error: float = 0.9,
            effect_noise: float = 0.5,
            seed: Optional[int] = None,
            n_samples: Optional[int] = 2000,  # NEW
    ) -> None:
        if pmlb_dataset_name not in SUPPORTED_PMLB_DATASETS:
            raise ValueError(
                f"pmlb_dataset_name must be one of {SUPPORTED_PMLB_DATASETS}, "
                f"got '{pmlb_dataset_name}'"
            )

        self.pmlb_dataset_name = pmlb_dataset_name
        self.synthesize_noise = synthesize_noise
        self.prior_error = prior_error
        self.effect_noise = effect_noise
        self.seed = seed

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_name = (
                "pmlb_meta_analysis_noisy.yaml" if synthesize_noise
                else "pmlb_meta_analysis.yaml"
            )
            config_path = Path(__file__).parent / "configs" / config_name

        # Separate filenames so noisy and clean versions don't collide
        if synthesize_noise:
            csv_name = "pmlb_meta_analysis_noisy-metadata-pyhealth.csv"
        else:
            csv_name = "pmlb_meta_analysis-metadata-pyhealth.csv"

        if not os.path.exists(os.path.join(root, csv_name)):
            self.prepare_metadata(
                root=root,
                pmlb_dataset_name=pmlb_dataset_name,
                synthesize_noise=synthesize_noise,
                prior_error=prior_error,
                effect_noise=effect_noise,
                seed=seed,
                n_samples=n_samples,  # NEW
            )

        default_tables = ["pmlb_meta_analysis"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or f"pmlb_{pmlb_dataset_name}",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(
        root: str,
        pmlb_dataset_name: str,
        synthesize_noise: bool = False,
        prior_error: float = 0.9,
        effect_noise: float = 0.5,
        seed: Optional[int] = None,
        n_samples: Optional[int] = 2000,
    ) -> None:
        """Fetch PMLB data and save as a PyHealth-compatible CSV.

        Args:
            root: Directory to save the CSV file.
            pmlb_dataset_name: Name of the PMLB dataset to fetch.
            synthesize_noise: Add observed_effect, variance, and
                prior_mean columns if True.
            prior_error: Prior quality parameter.
            effect_noise: Noise scale parameter.
            seed: Random seed.

        Raises:
            ImportError: If the ``pmlb`` package is not installed.
        """
        try:
            from pmlb import fetch_data
        except ImportError:
            raise ImportError(
                "The 'pmlb' package is required. "
                "Install with: pip install pmlb"
            )

        logger.info(f"Fetching PMLB dataset: {pmlb_dataset_name}")
        data = fetch_data(pmlb_dataset_name, local_cache_dir="./data/pmlb_raw_cache")

        if n_samples is not None and n_samples < len(data):
            data = data.sample(
                n=n_samples, random_state=seed or 0
            ).reset_index(drop=True)

        df = pd.DataFrame(data.values, columns=data.columns)
        df.insert(0, "patient_id", [f"trial_{i}" for i in range(len(df))])
        df.insert(1, "visit_id", [f"visit_{i}" for i in range(len(df))])
        df = df.rename(columns={"target": "true_effect"})

        if synthesize_noise:
            df = PMLBMetaAnalysisDataset._add_synthetic_noise(
                df,
                prior_error=prior_error,
                effect_noise=effect_noise,
                seed=seed,
            )
            csv_name = "pmlb_meta_analysis_noisy-metadata-pyhealth.csv"
        else:
            csv_name = "pmlb_meta_analysis-metadata-pyhealth.csv"

        os.makedirs(root, exist_ok=True)
        csv_path = os.path.join(root, csv_name)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved PMLB metadata to {csv_path}")

    @staticmethod
    def _add_synthetic_noise(
        df: pd.DataFrame,
        prior_error: float,
        effect_noise: float,
        seed: Optional[int],
    ) -> pd.DataFrame:
        """Add synthetic observed_effect, variance, and prior_mean cols.

        Follows Appendix B.6:
            V ~ Exp(1) * sqrt(effect_noise * E|U|)
            Y ~ N(U, V)
            M = p * offset + (1 - p) * U, with p chosen so that
                MSE(M, U) = prior_error * Var(U).

        Args:
            df: DataFrame with a 'true_effect' column.
            prior_error: Target prior error.
            effect_noise: Target effect noise.
            seed: Random seed.

        Returns:
            DataFrame with added columns.
        """
        rng = np.random.RandomState(seed)
        U = df["true_effect"].to_numpy(dtype=np.float64)
        n = len(U)

        mean_abs_U = float(np.mean(np.abs(U))) if n > 0 else 1.0
        scale = np.sqrt(effect_noise * mean_abs_U)
        V = rng.exponential(1.0, size=n) * scale

        Y = U + rng.randn(n) * np.sqrt(np.maximum(V, 1e-12))

        var_U = float(np.var(U)) if n > 1 else 1.0
        offset = rng.randn(n) * np.sqrt(var_U + 1e-12)
        mse_offset = float(np.mean((U - offset) ** 2))
        if mse_offset > 0 and var_U > 0:
            p = min(np.sqrt(prior_error * var_U / mse_offset), 1.0)
        else:
            p = 0.0
        M = p * offset + (1.0 - p) * U

        df = df.copy()
        df["observed_effect"] = Y
        df["variance"] = V
        df["prior_mean"] = M
        return df

    @property
    def default_task(self):
        """Returns the default task for this dataset."""
        from pyhealth.tasks.conformal_meta_analysis_task import (
            ConformalMetaAnalysisTask,
        )
        return ConformalMetaAnalysisTask()