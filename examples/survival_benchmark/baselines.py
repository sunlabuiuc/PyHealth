"""Authors: Aritra Saha and Ugochukwu Njoku.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html

Cox and DeepSurv adapters for the same patient/curve contract as SurvivalMDN.

The fitted baseline hazards use training outcomes only. Evaluation samples
never contribute to hazard fitting. Predictions are right-continuous steps,
equal to one before the first training time and held constant after the last.
Checkpoint files are trusted local artifacts, not an untrusted import format.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import torch
from numpy.typing import ArrayLike


def _data(features: ArrayLike, outcomes: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(features, np.float32), np.asarray(outcomes, np.float32)
    if x.ndim != 2 or not x.shape[0] or y.shape != (len(x), 2):
        raise ValueError(
            "features [N,D] and one [time,event] outcome per patient required"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and outcomes must be finite")
    if (y[:, 0] <= 0).any() or not np.isin(y[:, 1], [0, 1]).all():
        raise ValueError("times must be positive and events must be 0 or 1")
    if not y[:, 1].any():
        raise ValueError("training/validation needs at least one observed event")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _prediction_input(
    features: ArrayLike, grid: ArrayLike, width: int
) -> tuple[np.ndarray, np.ndarray]:
    x, t = np.asarray(features, np.float32), np.asarray(grid, float)
    if x.ndim != 2 or not len(x) or x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError("prediction features must be finite with the fitted width")
    if (
        t.ndim != 1
        or not t.size
        or not np.isfinite(t).all()
        or (t < 0).any()
        or (np.diff(t) <= 0).any()
    ):
        raise ValueError("prediction grid must be finite, nonnegative and increasing")
    return np.ascontiguousarray(x), t


def _at_grid(times: np.ndarray, curves: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate fitted right-continuous step curves without patient reordering."""
    indices = np.searchsorted(times, grid, side="right") - 1
    result = np.ones((len(curves), len(grid)))
    selected = indices >= 0
    result[:, selected] = curves[:, indices[selected]]
    if (
        not np.isfinite(result).all()
        or (result < -1e-7).any()
        or (result > 1 + 1e-7).any()
    ):
        raise ValueError("baseline returned invalid survival probabilities")
    return result.clip(0, 1)


def _scaler(x: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return np.zeros(x.shape[1]), np.ones(x.shape[1])
    mean, scale = x.astype(float).mean(0), x.astype(float).std(0)
    scale[scale == 0] = 1.0
    return mean, scale


def _scaled(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((x - mean) / scale, dtype=np.float32)


class CoxAdapter:
    """Classical Cox baseline with Breslow ties and optional L2 regularization.

    Args:
        alpha: Nonnegative finite regularization strength.
        standardize_input: Fit feature means/scales on training rows only.

    Examples:
        >>> adapter = CoxAdapter(alpha=0.1)
        >>> adapter.config["standardize_input"]
        True
    """

    def __init__(self, alpha: float = 0.1, standardize_input: bool = True) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.config = {
            "alpha": float(alpha),
            "standardize_input": bool(standardize_input),
        }

    def fit(
        self,
        features: ArrayLike,
        outcomes: ArrayLike,
        validation: tuple[ArrayLike, ArrayLike] | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> Self:
        """Fit the baseline using training patients and separate validation data.

        Args:
            features: Finite training covariates with shape [N, D].
            outcomes: Positive observed times and binary events, shape [N, 2].
            validation: Separate features/outcomes for neural early stopping.
                Cox accepts this argument for the shared interface and ignores it.
            checkpoint_dir: Directory for DeepSurv's best-validation weights.
                Required for DeepSurv; unused by Cox.

        Returns:
            This fitted adapter. Baseline hazards use training outcomes only.

        Raises:
            ValueError: If shapes, values or required validation inputs are invalid.
        """
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        x, y = _data(features, outcomes)
        self.width = x.shape[1]
        self.mean, self.scale = _scaler(x, self.config["standardize_input"])
        x = _scaled(x, self.mean, self.scale)
        self.model = CoxPHSurvivalAnalysis(
            alpha=self.config["alpha"], ties="breslow", n_iter=200
        )
        self.model.fit(x, Surv.from_arrays(y[:, 1].astype(bool), y[:, 0]))
        self.fit_info = {
            "backend": "scikit-survival.CoxPHSurvivalAnalysis",
            "ties": "breslow",
            "training_rows": len(x),
        }
        return self

    def predict_survival(self, features: ArrayLike, grid: ArrayLike) -> np.ndarray:
        """Predict survival curves in the same order as the feature rows.

        Args:
            features: Finite covariates of the fitted width, shape [N, D].
            grid: Finite, nonnegative, strictly increasing times, shape [T].

        Returns:
            NumPy probabilities of shape [N, T]. Curves equal one before the
            first training knot and stay constant beyond the last training knot.
        """
        x, t = _prediction_input(features, grid, self.width)
        curves = self.model.predict_survival_function(
            _scaled(x, self.mean, self.scale), return_array=True
        )
        return _at_grid(self.model.unique_times_, curves, t)

    def save(self, directory: str | Path) -> None:
        """Save the model, training-derived feature transform and configuration.

        Args:
            directory: Checkpoint destination, created if needed.

        Notes:
            Cox uses joblib. Load only checkpoints from a trusted source.
            DeepSurv stores tensor weights and NumPy baseline-hazard arrays.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "cox.joblib")
        (path / "adapter.json").write_text(
            json.dumps(
                {
                    "kind": "cox",
                    "config": self.config,
                    "width": self.width,
                    "feature_mean": self.mean.tolist(),
                    "feature_scale": self.scale.tolist(),
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> Self:
        """Restore a trusted checkpoint with its fitted feature transform.

        Args:
            directory: Directory previously written by ``save``.
            device: CPU/CUDA override supported by DeepSurv only. If omitted,
                DeepSurv preserves the original device.

        Returns:
            A fitted adapter ready for prediction.
        """
        path = Path(directory)
        saved = json.loads((path / "adapter.json").read_text())
        result = cls(**saved["config"])
        result.model = joblib.load(path / "cox.joblib")
        result.width = saved["width"]
        result.mean, result.scale = (
            np.asarray(saved["feature_mean"]),
            np.asarray(saved["feature_scale"]),
        )
        return result


class DeepSurvAdapter:
    """pycox CoxPH neural model, using full risk-set batches and early stopping.

    pycox uses ordered cumulative risk sets and an approximate order for ties,
    unlike the classical adapter's Breslow ties. Record this implementation
    convention with experiments. This adapter does not claim exact parity with
    the original Theano DeepSurv release.

    Args:
        hidden: Positive integer hidden-layer widths.
        lr: Positive Adam learning rate.
        epochs: Positive maximum number of full-risk-set epochs.
        patience: Positive early-stopping patience on validation loss.
        seed: Nonnegative random initialization seed.
        device: Torch device string, such as "cpu" or "cuda".
        standardize_input: Fit feature means/scales on training rows only.

    Examples:
        >>> adapter = DeepSurvAdapter(hidden=(16,), epochs=5)
        >>> adapter.config["hidden"]
        [16]
    """

    def __init__(
        self,
        hidden: Sequence[int] = (32, 32),
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 10,
        seed: int = 1,
        device: str = "cpu",
        standardize_input: bool = True,
    ) -> None:
        if not hidden or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or v < 1
            for v in hidden
        ):
            raise ValueError("hidden widths must be positive integers")
        if any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or v < 1
            for v in (epochs, patience)
        ):
            raise ValueError("epochs and patience must be positive integers")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("learning rate, epochs and patience must be positive")
        self.config = {
            "hidden": list(hidden),
            "lr": float(lr),
            "epochs": int(epochs),
            "patience": int(patience),
            "seed": int(seed),
            "device": str(device),
            "standardize_input": bool(standardize_input),
        }

    def _build(self, width: int) -> None:
        import torchtuples as tt
        from pycox.models import CoxPH

        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        net = tt.practical.MLPVanilla(
            width,
            self.config["hidden"],
            1,
            batch_norm=False,
            dropout=0,
            output_bias=False,
        )
        self.model = CoxPH(
            net,
            tt.optim.Adam(lr=self.config["lr"]),
            device=torch.device(self.config["device"]),
        )
        self.width = width

    def fit(
        self,
        features: ArrayLike,
        outcomes: ArrayLike,
        validation: tuple[ArrayLike, ArrayLike] | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> Self:
        """Fit the baseline using training patients and separate validation data.

        Args:
            features: Finite training covariates with shape [N, D].
            outcomes: Positive observed times and binary events, shape [N, 2].
            validation: Separate features/outcomes for neural early stopping.
                Cox accepts this argument for the shared interface and ignores it.
            checkpoint_dir: Directory for DeepSurv's best-validation weights.
                Required for DeepSurv; unused by Cox.

        Returns:
            This fitted adapter. Baseline hazards use training outcomes only.

        Raises:
            ValueError: If shapes, values or required validation inputs are invalid.
        """
        import torchtuples as tt

        x, y = _data(features, outcomes)
        if validation is None:
            raise ValueError("DeepSurv requires a separate validation split")
        xv, yv = _data(*validation)
        if x.shape[1] != xv.shape[1]:
            raise ValueError("training and validation feature widths differ")
        if checkpoint_dir is None:
            raise ValueError("provide checkpoint_dir for best-validation weights")
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.mean, self.scale = _scaler(x, self.config["standardize_input"])
        x, xv = _scaled(x, self.mean, self.scale), _scaled(xv, self.mean, self.scale)
        self._build(x.shape[1])
        stop = tt.callbacks.EarlyStopping(
            patience=self.config["patience"],
            file_path=str(path / "best_net.pt"),
            load_best=True,
            rm_file=False,
        )
        log = self.model.fit(
            x,
            (y[:, 0], y[:, 1]),
            batch_size=len(x),
            epochs=self.config["epochs"],
            callbacks=[stop],
            verbose=False,
            shuffle=False,
            val_data=(xv, (yv[:, 0], yv[:, 1])),
            val_batch_size=len(xv),
        )
        frame = log.to_pandas()
        if not np.isfinite(frame[["train_loss", "val_loss"]].to_numpy()).all():
            raise ValueError("DeepSurv training produced nonfinite losses")
        self.model.compute_baseline_hazards()
        self.fit_info = {
            "backend": "pycox.CoxPH",
            "ties": "pycox ordered risk-set approximation",
            "training_rows": len(x),
            "training_batch_size": len(x),
            "validation_batch_size": len(xv),
            "epochs_run": len(frame),
            "best_epoch": int(np.argmin(frame["val_loss"].to_numpy())),
            "best_validation_partial_nll": float(frame["val_loss"].min()),
            "training_history": frame.reset_index().to_dict(orient="records"),
        }
        return self

    def predict_survival(self, features: ArrayLike, grid: ArrayLike) -> np.ndarray:
        """Predict survival curves in the same order as the feature rows.

        Args:
            features: Finite covariates of the fitted width, shape [N, D].
            grid: Finite, nonnegative, strictly increasing times, shape [T].

        Returns:
            NumPy probabilities of shape [N, T]. Curves equal one before the
            first training knot and stay constant beyond the last training knot.
        """
        x, t = _prediction_input(features, grid, self.width)
        curves = self.model.predict_surv_df(_scaled(x, self.mean, self.scale))
        return _at_grid(curves.index.to_numpy(), curves.to_numpy().T, t)

    def save(self, directory: str | Path) -> None:
        """Save the model, training-derived feature transform and configuration.

        Args:
            directory: Checkpoint destination, created if needed.

        Notes:
            Cox uses joblib. Load only checkpoints from a trusted source.
            DeepSurv stores tensor weights and NumPy baseline-hazard arrays.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.net.state_dict(), path / "weights.pt")
        hazards = self.model.baseline_hazards_
        np.savez(
            path / "baseline_hazards.npz",
            time=hazards.index.to_numpy(),
            hazard=hazards.to_numpy(),
        )
        (path / "adapter.json").write_text(
            json.dumps(
                {
                    "kind": "deepsurv",
                    "config": self.config,
                    "width": self.width,
                    "feature_mean": self.mean.tolist(),
                    "feature_scale": self.scale.tolist(),
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> Self:
        """Restore a trusted checkpoint with its fitted feature transform.

        Args:
            directory: Directory previously written by ``save``.
            device: CPU/CUDA override supported by DeepSurv only. If omitted,
                DeepSurv preserves the original device.

        Returns:
            A fitted adapter ready for prediction.
        """
        path = Path(directory)
        saved = json.loads((path / "adapter.json").read_text())
        if device is not None:
            saved["config"]["device"] = device
        result = cls(**saved["config"])
        result._build(saved["width"])
        result.mean, result.scale = (
            np.asarray(saved["feature_mean"]),
            np.asarray(saved["feature_scale"]),
        )
        import pandas as pd

        result.model.net.load_state_dict(
            torch.load(
                path / "weights.pt",
                map_location=result.config["device"],
                weights_only=True,
            )
        )
        with np.load(
            path / "baseline_hazards.npz", allow_pickle=False
        ) as saved_hazards:
            hazards = pd.Series(
                saved_hazards["hazard"],
                index=saved_hazards["time"],
                name="baseline_hazards",
            )
        result.model.compute_baseline_cumulative_hazards(baseline_hazards_=hazards)
        return result
