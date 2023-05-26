"""KCal: Kernel-based Calibration

Implementation based on https://github.com/zlin7/KCal.

Paper:

    Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun.
    "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration
    for Deep Neural Networks." ICLR 2023.


"""
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from pyhealth.calib.base_classes import PostHocCalibrator
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.trainer import Trainer

from .bw import fit_bandwidth
from .embed_data import _EmbedData
from .kde import KDE_classification, KDECrossEntropyLoss, RBFKernelMean

__all__ = ["KCal"]


class ProjectionWrap(torch.nn.Module):
    """Base class for reprojections."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = KDECrossEntropyLoss()
        self.mode = "multiclass"

    def embed(self, x):
        """The actual projection"""
        raise NotImplementedError()

    def _forward(self, data, target=None, device=None):
        device = device or self.fc.weight.device

        data["supp_embed"] = self.embed(data["supp_embed"].to(device))
        data["supp_target"] = data["supp_target"].to(device)
        if target is None:
            # no supp vs pred - LOO prediction (for eval)
            assert "pred_embed" not in data
            data["pred_embed"] = None
            target = data["supp_target"]
            assert not self.training
        else:
            # used for train
            data["pred_embed"] = self.embed(data["pred_embed"].to(device))
            if "weights" in data and isinstance(data["weights"], torch.Tensor):
                data["weights"] = data["weights"].to(device)
        loss = self.criterion(
            data, target.to(device), eval_only=data["pred_embed"] is None
        )
        return {
            "loss": loss["loss"],
            "y_prob": loss["extra_output"]["prediction"],
            "y_true": target,
        }


class Identity(ProjectionWrap):
    """The identity reprojection (no reprojection)."""

    def embed(self, x):
        return x

    def forward(self, data, target=None):
        """Foward operations"""
        return self._forward(data, target, data["supp_embed"].device)


class SkipELU(ProjectionWrap):
    """The default reprojection module with 2 layers and a skip connection."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(input_features)
        self.mid = torch.nn.Linear(input_features, output_features)
        self.bn2 = torch.nn.BatchNorm1d(output_features)
        self.fc = torch.nn.Linear(output_features, output_features, bias=False)
        self.act = torch.nn.ELU()

    def embed(self, x):
        x = self.mid(self.bn(x))
        ret = self.fc(self.act(x))
        return ret + x

    def forward(self, data, target=None):
        """Foward operations"""
        return self._forward(data, target, self.fc.weight.device)


def _embed_dataset(model, dataset, record_id_name=None, debug=False, batch_size=32):

    ret = prepare_numpy_dataset(
        model,
        dataset,
        ["y_true", "embed"],
        incl_data_keys=["patient_id"]
        + ([] if record_id_name is None else [record_id_name]),
        forward_kwargs={"embed": True},
        debug=debug,
        batch_size=batch_size,
    )
    return {
        "labels": ret["y_true"],
        "indices": ret.get(record_id_name, None),
        "embed": ret["embed"],
        "group": ret["patient_id"],
    }


class KCal(PostHocCalibrator):
    """Kernel-based Calibration.
    This is a *full* calibration method for *multiclass* classification.
    It tries to calibrate the predicted probabilities for all classes,
    by using KDE classifiers estimated from the calibration set.

    Paper:

        Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun.
        "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration
        for Deep Neural Networks." ICLR 2023.

    Args:
        model (BaseModel): A trained model.

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.calibration import KCal
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate
        >>> cal_model = KCal(model)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Alternatively, you could re-fit the reprojection:
        >>> # cal_model.calibrate(cal_dataset=val_data, train_dataset=train_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> print(Trainer(model=cal_model, metrics=['cwECEt_adapt', 'accuracy']).evaluate(test_dl))
        {'accuracy': 0.7303689172252193, 'cwECEt_adapt': 0.03324275630220515}
    """

    def __init__(self, model: torch.nn.Module, debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        if model.mode != "multiclass":
            raise NotImplementedError()
        self.mode = self.model.mode  # multiclass
        self.model.eval()

        self.device = model.device
        self.debug = debug

        self.proj = Identity()
        self.kern = RBFKernelMean()
        self.record_id_name = None
        self.cal_data = {}
        self.num_classes = None

    def fit(
        self,
        train_dataset,
        val_dataset=None,
        split_by_patient=False,
        dim=32,
        bs_pred=64,
        bs_supp=20,
        epoch_len=5000,
        epochs=10,
        load_best_model_at_last=False,
        **train_kwargs
    ):
        """Fit the reprojection module.
        You don't need to call this function - it is called in :func:`KCal.calibrate`.
        For training details, please refer to the paper.

        Args:
            train_dataset (Dataset):
                The training dataset.
            val_dataset (Dataset, optional):
                The validation dataset. Defaults to None.
            split_by_patient (bool, optional):
                Whether to split the dataset by patient during training. Defaults to False.
            dim (int, optional):
                The dimension of the embedding. Defaults to 32.
            bs_pred (int, optional):
                The batch size for the prediction set. Defaults to 64.
            bs_supp (int, optional):
                The batch size for the support set. Defaults to 20.
            epoch_len (int, optional):
                The number of batches in an epoch. Defaults to 5000.
            epochs (int, optional):
                The number of epochs. Defaults to 10.
            load_best_model_at_last (bool, optional):
                Whether to load the best model (or the last model). Defaults to False.
            **train_kwargs:
                Other keyword arguments for :func:`pyhealth.trainer.Trainer.train`.
        """

        _train_data = _embed_dataset(
            self.model, train_dataset, self.record_id_name, self.debug
        )

        self.num_classes = max(_train_data["labels"]) + 1
        if not split_by_patient:
            # Allow using other samples from the same patient to make the prediction
            _train_data.pop("group")
        _train_data = _EmbedData(
            bs_pred=bs_pred, bs_supp=bs_supp, epoch_len=epoch_len, **_train_data
        )
        train_loader = DataLoader(
            _train_data, batch_size=1, collate_fn=_EmbedData._collate_func
        )

        val_loader = None
        if val_dataset is not None:
            _val_data = _embed_dataset(
                self.model, val_dataset, self.record_id_name, self.debug
            )
            _val_data = _EmbedData(epoch_len=1, **_val_data)
            val_loader = DataLoader(
                _val_data, batch_size=1, collate_fn=_EmbedData._collate_func
            )

        self.proj = SkipELU(len(_train_data.embed[0]), dim).to(self.device)
        trainer = Trainer(model=self.proj)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            monitor="loss",
            monitor_criterion="min",
            load_best_model_at_last=load_best_model_at_last,
            **train_kwargs
        )
        self.proj.eval()

    def calibrate(
        self,
        cal_dataset: Subset,
        num_fold=20,
        record_id_name=None,
        train_dataset: Subset = None,
        train_split_by_patient=False,
        load_best_model_at_last=True,
        **train_kwargs
    ):
        """Calibrate using a calibration dataset. If ``train_dataset`` is not None,
        it will be used to fit a re-projection from the base model embeddings.
        In either case, the calibration set will be used to construct the KDE classifier.

        Args:
            cal_dataset (Subset): Calibration set.
            record_id_name (str, optional): the key/name of the unique index for records.
                Defaults to None.
            train_dataset (Subset, optional): Dataset to train the reprojection.
                Defaults to None (no training).
            train_split_by_patient (bool, optional):
                Whether to split by patient when training the embeddings.
                That is, do we use samples from the same patient in KDE during *training*.
                Defaults to False.
            load_best_model_at_last (bool, optional):
                Whether to load the best reprojection basing on the calibration set.
                Defaults to True.
            train_kwargs (dict, optional): Additional arguments for training the reprojection.
                Passed to  :func:`KCal.fit`
        """

        self.record_id_name = record_id_name
        if train_dataset is not None:
            self.fit(
                train_dataset,
                val_dataset=cal_dataset,
                split_by_patient=train_split_by_patient,
                load_best_model_at_last=load_best_model_at_last,
                **train_kwargs
            )
        else:
            print(
                "No `train_dataset` - using the raw embeddings from the base classifier."
            )

        _cal_data = _embed_dataset(
            self.model, cal_dataset, self.record_id_name, self.debug
        )
        if self.num_classes is None:
            self.num_classes = max(_cal_data["labels"]) + 1
        assert (
            self.num_classes == max(_cal_data["labels"]) + 1
        ), "Train/Calibration data seem to have different classes"
        self.cal_data["Y"] = torch.tensor(
            _cal_data["labels"], dtype=torch.long, device=self.device
        )
        self.cal_data["Y"] = torch.nn.functional.one_hot(
            self.cal_data["Y"], self.num_classes
        ).float()
        with torch.no_grad():
            self.cal_data["X"] = self.proj.embed(
                torch.tensor(_cal_data["embed"], dtype=torch.float, device=self.device)
            )

        # Choose bandwidth
        self.kern.set_bandwidth(
            fit_bandwidth(group=_cal_data["group"], num_fold=num_fold, **self.cal_data)
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :param **kwargs: Additional arguments to the base model.

        :return:  A dictionary with all results from the base model, with the following modified:

            ``y_prob``: calibrated predicted probabilities.
            ``loss``: Cross entropy loss  with the new y_prob.
        :rtype: Dict[str, torch.Tensor]
        """
        ret = self.model(embed=True, **kwargs)
        X_pred = self.proj.embed(ret.pop("embed"))
        ret["y_prob"] = KDE_classification(
            kern=self.kern, X_pred=X_pred, **self.cal_data
        )
        ret["loss"] = self.proj.criterion.log_loss(ret["y_prob"], ret["y_true"])
        return ret


if __name__ == "__main__":

    from pyhealth.calib.calibration import KCal
    from pyhealth.datasets import (ISRUCDataset, get_dataloader,
                                   split_by_patient)
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn

    sleep_ds = ISRUCDataset(
        root="/srv/local/data/trash/",
        dev=True,
    ).set_task(sleep_staging_isruc_fn)
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
    model = SparcNet(
        dataset=sleep_ds, feature_keys=["signal"], label_key="label", mode="multiclass"
    )
    # ... Train the model here ...
    # Calibrate
    cal_model = KCal(model)
    cal_model.calibrate(cal_dataset=val_data)
    # Evaluate
    from pyhealth.trainer import Trainer

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    print(
        Trainer(model=cal_model, metrics=["cwECEt_adapt", "accuracy"]).evaluate(test_dl)
    )
