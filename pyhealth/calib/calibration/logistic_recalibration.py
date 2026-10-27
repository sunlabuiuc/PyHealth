"""
Logistic recalibration (intercept-only and intercept + slope).

Refits the calibration intercept, and optionally the calibration slope, of a
trained model on the logit scale using a calibration set.
"""

import warnings

import torch
from torch import optim
from torch.utils.data import Subset

from pyhealth.calib.base_classes import PostHocCalibrator
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["LogisticRecalibration"]


class LogisticRecalibration(PostHocCalibrator):
    """Logistic recalibration

    Updates the predicted log-odds of a trained model as
    ``a + b * logit``, where ``a`` is the calibration intercept and ``b`` is the
    calibration slope, both fitted by maximum likelihood on a calibration set.
    This is a common way to update a clinical prediction model for a new
    setting when there are too few local events to refit it.

    Two methods are supported:

    - ``"intercept"`` fits ``a`` with ``b`` fixed at 1. It corrects the
      overall event rate (calibration-in-the-large) and needs the fewest
      events.
    - ``"intercept_slope"`` fits both ``a`` and ``b``. A fitted slope with
      ``0 < b < 1`` means the original predictions were too extreme, and
      ``b > 1`` means they were too moderate.

    Temperature scaling for binary tasks fits ``b`` alone with ``a`` fixed at
    0, so it cannot fit an additive shift of the log-odds on its own.

    Only binary and multilabel tasks are supported. For multilabel tasks each
    label gets its own ``a`` and ``b``.

    The fit is plain maximum likelihood with no penalty. A label that takes
    a single value on the calibration set cannot be fitted, so it is left at
    ``a = 0`` and ``b = 1`` with a warning (if every label is like this, an
    error is raised). With ``"intercept_slope"``, a label whose positives and
    negatives are perfectly separated on the logit gets a warning, because
    its slope cannot be estimated. With very few events, prefer
    ``"intercept"``.

    Paper:

        [1] Steyerberg, Ewout W., Gerard J. J. M. Borsboom, Hans C. van
        Houwelingen, Marinus J. C. Eijkemans, and J. Dik F. Habbema.
        "Validation and updating of predictive logistic regression models:
        a study on sample size and shrinkage." Statistics in Medicine 23,
        no. 16 (2004): 2567-2586.

        [2] Vergouwe, Yvonne, Daan Nieboer, Rianne Oostenbrink, Thomas P. A.
        Debray, Gordon D. Murray, Michael W. Kattan, Hendrik Koffijberg,
        Karel G. M. Moons, and Ewout W. Steyerberg.
        "A closed testing procedure to select an appropriate method for
        updating prediction models." Statistics in Medicine 36, no. 28
        (2017): 4529-4539.

    :param model: A trained base model.
    :type model: BaseModel
    :param method: ``"intercept"`` or ``"intercept_slope"``,
        defaults to ``"intercept_slope"``.
    :type method: str

    Examples:
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.calib.calibration import LogisticRecalibration
        >>> # ... Train a binary model on data from the original setting ...
        >>> cal_model = LogisticRecalibration(model, method="intercept")
        >>> cal_model.calibrate(cal_dataset=new_site_data)
        >>> print(cal_model.intercept, cal_model.slope)
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> trainer = Trainer(model=cal_model, metrics=["ECE", "roc_auc"])
        >>> print(trainer.evaluate(test_dl))
    """

    METHODS = ("intercept", "intercept_slope")

    def __init__(
        self,
        model: BaseModel,
        method: str = "intercept_slope",
        debug=False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        self.mode = self.model.mode
        if self.mode not in ("binary", "multilabel"):
            raise ValueError(
                "LogisticRecalibration supports binary and multilabel tasks, "
                f"got mode={self.mode!r}"
            )
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.method = method
        for param in model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.device = model.device
        self.debug = debug

        num_outputs = self.model.get_output_size()
        self.intercept = torch.nn.Parameter(
            torch.zeros(num_outputs, dtype=torch.float32, device=self.device)
        )
        self.slope = torch.nn.Parameter(
            torch.ones(num_outputs, dtype=torch.float32, device=self.device),
            requires_grad=method == "intercept_slope",
        )

    def calibrate(self, cal_dataset: Subset, max_iter=100):
        """Fit the calibration intercept (and slope) on a calibration dataset.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        :param max_iter: maximum L-BFGS iterations, defaults to 100
        :type max_iter: int, optional
        :return: None
        :rtype: None
        """
        _cal_data = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_true", "logit"], debug=self.debug
        )
        logits = torch.tensor(_cal_data["logit"], dtype=torch.float, device=self.device)
        label = torch.tensor(_cal_data["y_true"], dtype=torch.float, device=self.device)
        self._fit(logits, label, max_iter=max_iter)

    def _fit(self, logits: torch.Tensor, label: torch.Tensor, max_iter=100):
        if logits.shape[1] != self.intercept.shape[0]:
            raise ValueError(
                f"model has {self.intercept.shape[0]} outputs but the calibration "
                f"logits have {logits.shape[1]} columns"
            )
        fit = label.min(dim=0).values != label.max(dim=0).values
        if not fit.any():
            raise ValueError(
                "every label takes a single value on the calibration set, so "
                "there is nothing to fit"
            )
        if not fit.all():
            warnings.warn(
                "labels with a single value on the calibration set are left "
                f"unchanged: outputs {(~fit).nonzero().flatten().tolist()}",
                RuntimeWarning,
                stacklevel=3,
            )
        if self.method == "intercept_slope":
            separated = self._separated(logits[:, fit], label[:, fit])
            if separated:
                cols = fit.nonzero().flatten()[separated].tolist()
                warnings.warn(
                    "positives and negatives are perfectly separated on the logit "
                    f"for outputs {cols}, so their slope cannot be estimated. "
                    "Consider method='intercept' or a larger calibration set",
                    RuntimeWarning,
                    stacklevel=3,
                )

        with torch.no_grad():
            self.intercept.zero_()
            self.slope.fill_(1.0)
        params = [self.intercept]
        if self.method == "intercept_slope":
            params.append(self.slope)
        optimizer = optim.LBFGS(
            params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        criterion = torch.nn.functional.binary_cross_entropy_with_logits

        def _eval():
            optimizer.zero_grad()
            recal = self.intercept[fit] + self.slope[fit] * logits[:, fit]
            loss = criterion(recal, label[:, fit])
            loss.backward()
            return loss

        self.train()
        optimizer.step(_eval)
        self.eval()

        if not (
            torch.isfinite(self.intercept).all() and torch.isfinite(self.slope).all()
        ):
            raise RuntimeError(
                "recalibration produced non-finite parameters. Check the "
                "calibration logits for NaN or infinite values"
            )

    @staticmethod
    def _separated(logits: torch.Tensor, label: torch.Tensor) -> list[int]:
        """Indices of columns whose classes do not overlap on the logit."""
        cols = []
        for j in range(logits.shape[1]):
            pos = logits[label[:, j] == 1, j]
            neg = logits[label[:, j] == 0, j]
            if neg.max() <= pos.min() or pos.max() <= neg.min():
                cols.append(j)
        return cols

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :param **kwargs: Additional arguments to the base model.

        :return:  A dictionary with all results from the base model, with the
            following modified:

            ``y_prob``: calibrated predicted probabilities.
            ``loss``: Binary cross entropy loss with the new logits.
            ``logit``: recalibrated logits, ``intercept + slope * logit``.
        :rtype: dict[str, torch.Tensor]
        """
        ret = self.model(**kwargs)
        ret["logit"] = self.intercept + self.slope * ret["logit"]
        ret["y_prob"] = self.model.prepare_y_prob(ret["logit"])
        criterion = self.model.get_loss_function()
        ret["loss"] = criterion(ret["logit"], ret["y_true"])
        return ret
