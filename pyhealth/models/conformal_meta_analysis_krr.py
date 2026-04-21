"""Conformal Meta-Analysis Model for PyHealth.

Implements the Conformal Meta-Analysis (CMA) algorithm from:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    In Proceedings of Machine Learning Research, volume 259, 563-593.

CMA produces prediction intervals for treatment effects by combining:
    1. A learned prior (mu, kappa) from untrusted data
    2. Kernel ridge regression (KRR) on trusted trial data
    3. Full conformal prediction for distribution-free coverage

Unlike typical deep learning models, CMA does not train via gradient
descent. The "forward pass" runs the conformal prediction algorithm
using the provided batch as both training and test data (via
leave-one-out style prediction).

Each sample is expected to contain:
    - features (torch.Tensor): trial feature vector X
    - observed_effect (float): noisy observation Y
    - variance (float): within-trial variance V
    - prior_mean (float): untrusted prior M(X)
    - true_effect (float): target true effect U (label)

Example:
    >>> from pyhealth.datasets import PMLBMetaAnalysisDataset
    >>> from pyhealth.tasks import ConformalMetaAnalysisTask
    >>> from pyhealth.models import ConformalMetaAnalysisModel
    >>> dataset = PMLBMetaAnalysisDataset(
    ...     root="./data/pmlb",
    ...     pmlb_dataset_name="1196_BNG_pharynx",
    ...     synthesize_noise=True,
    ...     dev=True,
    ... )
    >>> samples = dataset.set_task(ConformalMetaAnalysisTask())
    >>> model = ConformalMetaAnalysisModel(dataset=samples, alpha=0.1)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models.base_model import BaseModel


class ConformalMetaAnalysisModel(BaseModel):
    """Conformal Meta-Analysis model using KRR + full conformal prediction.

    Produces prediction intervals (lower, upper) for the true effect
    u given trial features x, observed effect y, variance v, and a
    prior mean m. Under the paper's assumptions the interval covers
    the true effect u with probability at least 1 - alpha.

    This model does not have trainable parameters for KRR itself.
    The ``_dummy_param`` inherited from BaseModel lets PyHealth
    infer the model's device, but backpropagation is a no-op.
    To learn the prior from untrusted data, use ``PriorEncoder``
    separately and write its predictions into the dataset as
    ``prior_mean`` before instantiating this model.

    Args:
        dataset: SampleDataset produced by
            ``ConformalMetaAnalysisTask``.
        alpha: Significance level; coverage target is 1 - alpha.
            Defaults to 0.1 (90% coverage).
        eta: Noise correction parameter >= 0. 0 disables noise
            correction (appropriate for low-variance data).
            Defaults to 0.0.
        kernel_type: "gaussian" or "laplace". Defaults to "gaussian".
        kernel_bandwidth: Kernel bandwidth ``gamma``. If None
            (default), uses ``0.2`` to match the PMLB data
            generator in Appendix B.6 of the paper. Override with
            a positive float for other datasets; the value is used
            directly (not through a median heuristic).

    Attributes:
        alpha: Significance level.
        eta: Noise correction parameter.
        kernel_type: Kernel function type.
        kernel_bandwidth: Kernel bandwidth (may be None until fit).
        mode: Always "regression" for CMA.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        alpha: float = 0.1,
        eta: float = 0.0,
        kernel_type: str = "gaussian",
        kernel_bandwidth: Optional[float] = None,
    ) -> None:
        super().__init__(dataset=dataset)

        if kernel_type not in ("gaussian", "laplace"):
            raise ValueError(
                f"kernel_type must be 'gaussian' or 'laplace', "
                f"got '{kernel_type}'"
            )
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if kernel_bandwidth is not None and kernel_bandwidth <= 0:
            raise ValueError(
                f"kernel_bandwidth must be > 0 when set, "
                f"got {kernel_bandwidth}"
            )

        self.alpha = alpha
        self.eta = eta
        self.kernel_type = kernel_type
        self.kernel_bandwidth = kernel_bandwidth
        self.mode = "regression"

    # ------------------------------------------------------------------
    # Core CMA algorithm
    # ------------------------------------------------------------------
    def _compute_kernel_matrix(
            self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute the pairwise kernel matrix between two feature batches.

        The kernel is ``exp(-gamma * dist(X1, X2))`` where ``dist``
        is the mean-over-features squared distance for the Gaussian
        kernel and mean-over-features absolute distance for the
        Laplace kernel. Using mean (rather than sum) matches the
        authors' reference implementation for PMLB; using sum would
        change the effective bandwidth by a factor of d and break
        the RKHS assumption baked into the synthetic prior.

        The bandwidth ``gamma`` is taken from ``self.kernel_bandwidth``
        if set, otherwise defaults to ``0.2`` — the exact value used
        by the PMLB data generator in Appendix B.6 of Kaul & Gordon
        (2024). Mismatching the bandwidth between data generation
        and model fit breaks the conformal coverage guarantee on
        the synthetic benchmarks.

        Args:
            X1: First feature matrix, shape ``(n1, d)``.
            X2: Second feature matrix, shape ``(n2, d)``.

        Returns:
            Kernel matrix of shape ``(n1, n2)``.
        """
        # Pairwise differences over the feature axis
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]

        # Default to 0.2 to match the PMLB data generator exactly.
        # Exposed via kernel_bandwidth for non-PMLB datasets.
        gamma = (
            self.kernel_bandwidth
            if self.kernel_bandwidth is not None
            else 0.2
        )

        if self.kernel_type == "gaussian":
            # Authors use mean squared distance, not sum squared distance
            sq_dists = np.mean(diff ** 2, axis=2)
            return np.exp(-sq_dists * gamma)
        else:  # laplace
            abs_dists = np.mean(np.abs(diff), axis=2)
            return np.exp(-abs_dists * gamma)

    def _theorem4(
        self,
        Y: np.ndarray,
        V_bar: np.ndarray,
        M_bar: np.ndarray,
        K_bar: np.ndarray,
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute interval centers and half-widths (Theorem 4).

        Args:
            Y: Training observed effects, shape (n,).
            V_bar: All variances (train + test), shape (n+1,).
            M_bar: All prior means (train + test), shape (n+1,).
            K_bar: Full kernel matrix, shape (n+1, n+1).
            lam: Ridge parameter.

        Returns:
            Tuple (G, H) of arrays, each shape (n,), where the
            conformal interval for training trial i is G[i] +/- H[i].
        """
        # Variable names track the notation in Theorem 4 of the
        # paper rather than snake_case, so the linear-algebra steps
        # below are comparable line-by-line with the derivation:
        #   Y    = observed effects (training)
        #   V    = variances; v, V_bar include the test trial
        #   M    = prior means; m, M_bar include the test trial
        #   K    = kernel matrix; K_bar includes the test trial
        #   I    = identity; lam = ridge parameter
        #   Q, t = intermediate KRR-style projections
        #   A, B = coefficients of the affine score a - A^T r
        #   S2, D, rho = variance-correction terms
        #   G, H = interval center and half-width (the outputs)
        n_plus_1 = len(M_bar)
        I_bar = np.eye(n_plus_1)

        t_bar = np.linalg.solve(K_bar / lam + I_bar, M_bar)
        Q_bar = np.linalg.solve(K_bar + lam * I_bar, K_bar)

        Q = Q_bar[:-1, :-1]
        q = Q_bar[-1, :-1]
        q0 = Q_bar[-1, -1]

        V = V_bar[:-1]
        v = V_bar[-1]

        A = -q
        a = 1 - q0
        B = Y - Q @ Y - t_bar[:-1]
        b = -q @ Y - t_bar[-1]

        sign_A = np.sign(A)
        sign_A[A == 0] = 1.0
        B *= sign_A
        A *= sign_A

        S2 = lam * np.diag(Q)
        s2 = lam * q0

        I_mat = np.eye(len(Y))
        D = ((I_mat - Q) ** 2) @ V
        d = (q ** 2) @ V

        a2A2 = a ** 2 * S2 - A ** 2 * s2
        # Avoid division by zero
        a2A2_safe = np.where(np.abs(a2A2) < 1e-12, 1e-12, a2A2)

        rho = self.eta * (D * s2 - d * S2 - a2A2 * v)

        G = (A * B * s2 - a * b * S2) / a2A2_safe
        H = np.sqrt(
            np.maximum(0, s2 * S2 * (A * b - a * B) ** 2 - rho * a2A2)
        ) / a2A2_safe

        return G, H

    def _predict_one(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        V_train: np.ndarray,
        M_train: np.ndarray,
        x_test: np.ndarray,
        m_test: float,
    ) -> Tuple[float, float]:
        """Predict a prediction interval for one test trial.

        Args:
            X_train: Training feature matrix, shape (n, d).
            Y_train: Training observed effects, shape (n,).
            V_train: Training variances, shape (n,).
            M_train: Training prior means, shape (n,).
            x_test: Test feature vector, shape (d,).
            m_test: Test prior mean.

        Returns:
            Tuple (lower, upper) for the prediction interval.
        """
        n = len(Y_train)
        tau = int(np.ceil((1 - self.alpha) * (n + 1)))
        if tau > n:
            return -np.inf, np.inf

        # Augmented data (Algorithm 2: set test variance to 0)
        V_bar = np.append(V_train, 0.0)
        M_bar = np.append(M_train, m_test)
        X_all = np.vstack([X_train, x_test.reshape(1, -1)])
        K_bar = self._compute_kernel_matrix(X_all, X_all)

        # Ensure idiocentricity (Theorem 6)
        lam = float(np.max(np.diag(K_bar)))

        G, H = self._theorem4(Y_train, V_bar, M_bar, K_bar, lam)
        lower_endpoints = G - H
        upper_endpoints = G + H

        y_lower = float(np.flip(np.sort(lower_endpoints))[tau - 1])
        y_upper = float(np.sort(upper_endpoints)[tau - 1])
        return y_lower, y_upper

    # ------------------------------------------------------------------
    # PyHealth forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        observed_effect: torch.Tensor,
        variance: torch.Tensor,
        prior_mean: torch.Tensor,
        true_effect: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run CMA prediction on a batch of trials.

        The batch is used as both training and test data: each trial
        is held out in turn, predicted using the remaining trials
        as context, and its interval and point estimate are returned.

        Args:
            features: Feature tensor of shape (batch_size, d).
            observed_effect: Y values of shape (batch_size,) or
                (batch_size, 1).
            variance: V values of shape (batch_size,) or
                (batch_size, 1).
            prior_mean: M values of shape (batch_size,) or
                (batch_size, 1).
            true_effect: Optional target U values for loss/evaluation.

        Returns:
            Dictionary with keys:
                - ``y_pred``: point estimate per trial (midpoint of
                  the interval), shape (batch_size, 1)
                - ``y_prob``: alias for ``y_pred``, provided to
                  satisfy :class:`BaseModel`'s forward return
                  contract
                - ``interval_lower``: lower bounds, shape
                  (batch_size, 1)
                - ``interval_upper``: upper bounds, shape
                  (batch_size, 1)
                - ``loss``: MSE between midpoint and true_effect
                  (only present if ``true_effect`` is provided).
                  Note that this loss has ``requires_grad=True``
                  for API compatibility but no autograd graph —
                  CMA has no trainable parameters.
                - ``y_true``: the true labels, if provided
        """
        device = features.device

        X = features.detach().cpu().numpy().astype(np.float64)
        Y = observed_effect.detach().cpu().numpy().astype(np.float64).ravel()
        V = variance.detach().cpu().numpy().astype(np.float64).ravel()
        M = prior_mean.detach().cpu().numpy().astype(np.float64).ravel()

        n = len(Y)
        lowers = np.zeros(n, dtype=np.float64)
        uppers = np.zeros(n, dtype=np.float64)

        # Leave-one-out: predict each trial using the others
        for i in range(n):
            mask = np.arange(n) != i
            lo, hi = self._predict_one(
                X_train=X[mask],
                Y_train=Y[mask],
                V_train=V[mask],
                M_train=M[mask],
                x_test=X[i],
                m_test=float(M[i]),
            )
            lowers[i] = lo
            uppers[i] = hi

        midpoints = (lowers + uppers) / 2.0

        y_pred_tensor = torch.tensor(
            midpoints, dtype=torch.float32, device=device
        ).unsqueeze(-1)

        out: Dict[str, torch.Tensor] = {
            # y_pred is the CMA-specific name; y_prob is aliased to
            # the same tensor so code relying on the BaseModel.forward
            # return contract (which advertises "y_prob") keeps working.
            "y_pred": y_pred_tensor,
            "y_prob": y_pred_tensor,
            "interval_lower": torch.tensor(
                lowers, dtype=torch.float32, device=device
            ).unsqueeze(-1),
            "interval_upper": torch.tensor(
                uppers, dtype=torch.float32, device=device
            ).unsqueeze(-1),
        }

        if true_effect is not None:
            U = true_effect.detach().cpu().numpy().astype(np.float64).ravel()
            # Clip infinities so loss is finite when tau > n
            mid_finite = np.where(
                np.isfinite(midpoints), midpoints, U
            )
            mse = float(np.mean((mid_finite - U) ** 2))
            # requires_grad=True is cosmetic: the MSE is computed in
            # NumPy so no autograd graph connects it back to this
            # module's parameters. Calling .backward() on this loss is
            # a no-op — CMA has no trainable weights. Returning a
            # leaf-tensor with requires_grad is only done so PyHealth's
            # generic trainer loop doesn't reject the output.
            out["loss"] = torch.tensor(
                mse, dtype=torch.float32, device=device, requires_grad=True
            )
            out["y_true"] = torch.tensor(
                U, dtype=torch.float32, device=device
            ).unsqueeze(-1)

        return out