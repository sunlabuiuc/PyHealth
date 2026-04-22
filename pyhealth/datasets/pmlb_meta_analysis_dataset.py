"""PMLB Meta-Analysis Dataset for PyHealth.

Dataset for conformal meta-analysis experiments using regression
datasets from the Penn Machine Learning Benchmark (PMLB).

PMLB is freely accessible at:
    https://github.com/EpistasisLab/pmlb
    Install: pip install pmlb

The only one of the three datasets used in Kaul & Gordon (2024) is compatible:
    - 1196_BNG_pharynx - compatible
    - 1201_BNG_breastTumor - needs yaml config
    - 1193_BNG_lowbwt - needs yaml config

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
    "1196_BNG_pharynx"
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
            n_samples: Optional[int] = 2000,
    ) -> None:
        if pmlb_dataset_name not in SUPPORTED_PMLB_DATASETS:
            raise ValueError(
                f"pmlb_dataset_name must be one of {SUPPORTED_PMLB_DATASETS}, "
                f"got '{pmlb_dataset_name}' to add update SUPPORTED_PMLB_DATASETS"
                f" and create config YAML File and update config logic"
            )
        if prior_error < 0:
            raise ValueError(
                f"prior_error must be non-negative, got {prior_error}. "
                f"It parameterizes the distance between the prior mean M "
                f"and the true effect U (0 = perfect prior, larger = worse)."
            )
        if effect_noise < 0:
            raise ValueError(
                f"effect_noise must be non-negative, got {effect_noise}. "
                f"It scales the within-trial variances V (0 = noise-free "
                f"observations Y = U, larger = noisier observations)."
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
            n_samples: If provided and smaller than the PMLB dataset,
                randomly subsample this many rows. PMLB datasets can
                have hundreds of thousands of rows; subsampling keeps
                processing fast and fits the meta-analysis setting
                (n <= 500 trials) studied in the paper. Defaults to
                2000. Pass None to use all available rows.

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
        seed: int = 0,
        gamma: float = 0.2,
        kernel_type: str = "gaussian",
        num_prior: int = 1000,
    ) -> pd.DataFrame:
        """Generate synthetic meta-analysis noise following Appendix B.6.

        This reproduces the ``generate_problem`` routine from the
        authors' reference implementation at
        https://github.com/shivak/conformal-meta. Given a regression
        dataset with features X and true effects U, it simulates the
        observables of a meta-analysis:

            - ``prior_mean`` (M): an imperfect predictor of U, built as
              a convex combination of U and a random RKHS function
              ``prior_noise = K[:, prior_indices] @ prior_weights``,
              where ``K`` is the kernel matrix over all rows of the
              dataset. The mixing fraction is calibrated so that
              ``MSE(M, U) = prior_error * Var(U)``.
            - ``variance`` (V): within-trial variance drawn as
              ``V ~ Exp(1) * sqrt(effect_noise * E|U|)``. Scales with
              ``effect_noise`` and is floored at 1e-6 for numerical
              stability.
            - ``observed_effect`` (Y): noisy observation
              ``Y ~ N(U, V)`` of the true effect.

        The kernel matrix is built from features standardized with
        zero mean and unit variance, using the same ``gamma = 0.2``
        and mean-normalized-distance convention as the reference.

        Args:
            df: Input DataFrame containing at least ``true_effect``
                and feature columns. Must also contain ``patient_id``
                and ``visit_id`` (excluded from the feature matrix).
            prior_error: Non-negative parameter controlling the gap
                between the synthetic prior mean M and the true
                effect U. 0 means M = U (perfect prior); 1 means
                ``Var(M - U) = Var(U)`` (useless prior).
            effect_noise: Non-negative parameter scaling the within-
                trial variance V. 0 yields near-zero variances; larger
                values yield noisier Y.
            seed: Random seed controlling prior-index selection, RKHS
                weights, variance sampling, and label noise.
            gamma: Kernel bandwidth parameter. 0.2 matches the
                reference implementation. Only used when
                ``kernel_type='gaussian'`` or ``'laplace'``.
            kernel_type: Either ``'gaussian'`` (default) or
                ``'laplace'``. Matches the reference's per-dataset
                kernel choice for PMLB (``1196_BNG_pharynx`` is
                gaussian in the reference).
            num_prior: Number of rows used as prior anchors. Capped
                at ``len(df) // 2`` so at least half the rows remain
                for the train/test split.

        Returns:
            A copy of ``df`` with three new columns appended:
            ``observed_effect`` (Y), ``variance`` (V), and
            ``prior_mean`` (M).
        """
        rng = np.random.RandomState(seed)
        U = df["true_effect"].to_numpy(dtype=np.float64)
        n = len(U)

        # Feature matrix and standardization (reference cell 7:
        # StandardScaler().fit_transform(X))
        feature_cols = [
            c for c in df.columns
            if c not in ("patient_id", "visit_id", "true_effect")
        ]
        X = df[feature_cols].to_numpy(dtype=np.float64)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

        # --- Kernel matrix over all rows (reference cell 7) ---
        # dist_matrix = mean over features of pairwise distance
        if kernel_type == "gaussian":
            diff = X[:, None, :] - X[None, :, :]
            dist_matrix = np.mean(diff ** 2, axis=-1)
        elif kernel_type == "laplace":
            diff = X[:, None, :] - X[None, :, :]
            dist_matrix = np.mean(np.abs(diff), axis=-1)
        else:
            raise ValueError(
                f"kernel_type must be 'gaussian' or 'laplace', "
                f"got '{kernel_type}'"
            )
        K = np.exp(-dist_matrix * gamma)

        # --- Prior M via RKHS mixing (reference cell 11) ---
        # prior_noise = K[:, prior_indices] @ random_weights; M mixes
        # this with U so that MSE(M, U) = prior_error * Var(U).
        num_prior_eff = min(num_prior, n // 2)
        shuffled = rng.permutation(n)
        prior_indices = shuffled[:num_prior_eff]
        K_prior = K[:, prior_indices]
        prior_weights = rng.randn(num_prior_eff)
        prior_noise = K_prior @ prior_weights

        U_var = float(np.var(U))
        mse = float(np.mean((prior_noise - U) ** 2))
        if mse < 1e-12 or prior_error == 0.0:
            mixing_frac = 0.0
        else:
            mixing_frac = np.sqrt(prior_error * U_var / mse)
        M = mixing_frac * prior_noise + (1.0 - mixing_frac) * U

        # --- Variance V ~ Exp(1) * sqrt(effect_noise * E|U|) ---
        # (reference cell 11, exact formula)
        mean_abs_U = float(np.mean(np.abs(U)))
        V = (
            rng.exponential(scale=1.0, size=n)
            * np.sqrt(effect_noise * mean_abs_U)
        )
        V = np.maximum(V, 1e-6)

        # --- Observed Y ~ N(U, V) ---
        Y = U + rng.randn(n) * np.sqrt(V)

        df = df.copy()
        df["observed_effect"] = Y
        df["variance"] = V
        df["prior_mean"] = M
        return df

    @property
    def default_task(self) -> "ConformalMetaAnalysisTask":
        """Returns the default task for this dataset.

        The configuration depends on ``synthesize_noise``:
            - ``synthesize_noise=True``: the CSV contains
              ``observed_effect``, ``variance``, and ``prior_mean``
              columns, so the task carries all of them through.
            - ``synthesize_noise=False``: the CSV only has
              ``true_effect``, so the optional source columns are
              set to ``None`` and the task behaves as a plain
              regression task.
        """
        from pyhealth.tasks.conformal_meta_analysis import (
            ConformalMetaAnalysisTask,
        )
        if self.synthesize_noise:
            return ConformalMetaAnalysisTask()
        return ConformalMetaAnalysisTask(
            observed_column=None,
            variance_column=None,
            prior_column=None,
        )