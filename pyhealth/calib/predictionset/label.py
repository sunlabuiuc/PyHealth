"""
LABEL: Least ambiguous set-valued classifiers with bounded error levels.

Paper:

    Sadinle, Mauricio, Jing Lei, and Larry Wasserman.
    "Least ambiguous set-valued classifiers with bounded error levels."
    Journal of the American Statistical Association 114, no. 525 (2019): 223-234.

"""

from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Subset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["LABEL"]


def _query_quantile(scores, alpha):
    scores = np.sort(scores)
    N = len(scores)
    loc = int(np.floor(alpha * (N + 1))) - 1
    return -np.inf if loc == -1 else scores[loc]


class LABEL(SetPredictor):
    """LABEL: Least ambiguous set-valued classifiers with bounded error levels.

    This is a prediction-set constructor for multi-class classification problems.
    It controls either :math:`\\mathbb{P}\\{Y \\not \\in C(X) | Y=k\\}\\leq \\alpha_k`
    (when ``alpha`` is an array), or :math:`\\mathbb{P}\\{Y \\not \\in C(X)\\}\\leq \\alpha` (when ``alpha`` is a float).
    Here, :math:`C(X)` denotes the final prediction set.
    This is essentially a split conformal prediction method using the predicted scores.

    Paper:

        Sadinle, Mauricio, Jing Lei, and Larry Wasserman.
        "Least ambiguous set-valued classifiers with bounded error levels."
        Journal of the American Statistical Association 114, no. 525 (2019): 223-234.


    :param model: A trained base model.
    :type model: BaseModel
    :param alpha: Target mis-coverage rate(s).
    :type alpha: Union[float, np.ndarray]

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset import LABEL
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate the set classifier, with different class-specific mis-coverage rates
        >>> cal_model = LABEL(model, [0.15, 0.3, 0.15, 0.15, 0.15])
        >>> # Note that we used the test set here because ISRUCDataset has relatively few
        >>> # patients, and calibration set should be different from the validation set
        >>> # if the latter is used to pick checkpoint. In general, the calibration set
        >>> # should be something exchangeable with the test set. Please refer to the paper.
        >>> cal_model.calibrate(cal_dataset=test_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(test_dl, additional_outputs=['y_predset'])
        >>> print(get_metrics_fn(cal_model.mode)(
        ... y_true_all, y_prob_all, metrics=['accuracy', 'miscoverage_ps'],
        ... y_predset=extra_output['y_predset'])
        ... )
        {'accuracy': 0.709843241966832, 'miscoverage_ps': array([0.1499847 , 0.29997638, 0.14993964, 0.14994704, 0.14999252])}
    """

    def __init__(
        self, model: BaseModel, alpha: Union[float, np.ndarray], debug=False, **kwargs
    ) -> None:
        super().__init__(model, **kwargs)
        if model.mode != "multiclass":
            raise NotImplementedError()
        self.mode = self.model.mode  # multiclass
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = model.device
        self.debug = debug

        if not isinstance(alpha, float):
            alpha = np.asarray(alpha)
        self.alpha = alpha

        self.t = None

    def calibrate(self, cal_dataset: Subset):
        """Calibrate the thresholds used to construct the prediction set.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        """
        cal_dataset = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_prob", "y_true"], debug=self.debug
        )
        y_prob = cal_dataset["y_prob"]
        y_true = cal_dataset["y_true"]

        N, K = cal_dataset["y_prob"].shape
        if isinstance(self.alpha, float):
            t = _query_quantile(y_prob[np.arange(N), y_true], self.alpha)
        else:
            t = [
                _query_quantile(y_prob[y_true == k, k], self.alpha[k]) for k in range(K)
            ]
        self.t = torch.tensor(t, device=self.device)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :return: A dictionary with all results from the base model, with the following updates:

                    y_predset: a bool tensor representing the prediction for each class.
        :rtype: Dict[str, torch.Tensor]
        """
        pred = self.model(**kwargs)
        pred["y_predset"] = pred["y_prob"] > self.t
        return pred


if __name__ == "__main__":

    from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn
    from pyhealth.calib.predictionset import LABEL

    sleep_ds = ISRUCDataset("/srv/local/data/trash", dev=True).set_task(
        sleep_staging_isruc_fn
    )
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
    model = SparcNet(
        dataset=sleep_ds, feature_keys=["signal"], label_key="label", mode="multiclass"
    )
    # ... Train the model here ...
    # Calibrate the set classifier, with different class-specific mis-coverage rates
    cal_model = LABEL(model, [0.15, 0.3, 0.15, 0.15, 0.15])
    # Note that I used the test set here because ISRUCDataset has relatively few
    # patients, and calibration set should be different from the validation set
    # if the latter is used to pick checkpoint. In general, the calibration set
    # should be something exchangeable with the test set. Please refer to the paper.
    cal_model.calibrate(cal_dataset=test_data)
    # Evaluate
    from pyhealth.trainer import Trainer, get_metrics_fn

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    print(
        get_metrics_fn(cal_model.mode)(
            y_true_all,
            y_prob_all,
            metrics=["accuracy", "miscoverage_ps"],
            y_predset=extra_output["y_predset"],
        )
    )
