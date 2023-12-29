import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange

from pyhealth.metrics import (binary_metrics_fn, multiclass_metrics_fn,
                              multilabel_metrics_fn, regression_metrics_fn)
from pyhealth.utils import create_directory

logger = logging.getLogger(__name__)


def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")


def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return


def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    elif mode == "regression":
        return regression_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


class Trainer:
    """Trainer for PyTorch models.

    Args:
        model: PyTorch model.
        checkpoint_path: Path to the checkpoint. Default is None, which means
            the model will be randomly initialized.
        metrics: List of metric names to be calculated. Default is None, which
            means the default metrics in each metrics_fn will be used.
        device: Device to be used for training. Default is None, which means
            the device will be GPU if available, otherwise CPU.
        enable_logging: Whether to enable logging. Default is True.
        output_path: Path to save the output. Default is "./output".
        exp_name: Name of the experiment. Default is current datetime.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.metrics = metrics
        self.device = device

        # set logger
        if enable_logging:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_path = os.path.join(output_path, exp_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        # set device
        self.model.to(self.device)

        # logging
        logger.info(self.model)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Device: {self.device}")

        # load checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)

        logger.info("")
        return

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict[str, object]] = None,
        steps_per_epoch: int = None,
        evaluation_steps: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: float = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
    ):
        """Trains the model.

        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            steps_per_epoch: Number of steps per epoch. Default is None.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch == None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                output = self.model(**data)
                loss = output["loss"]
                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1
            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))

        return

    def inference(self, dataloader, additional_outputs=None,
                  return_patient_ids=False) -> Dict[str, float]:
        """Model inference.

        Args:
            dataloader: Dataloader for evaluation.
            additional_outputs: List of additional output to collect.
                Defaults to None ([]).

        Returns:
            y_true_all: List of true labels.
            y_prob_all: List of predicted probabilities.
            loss_mean: Mean loss over batches.
            additional_outputs (only if requested): Dict of additional results.
            patient_ids (only if requested): List of patient ids in the same order as y_true_all/y_prob_all.
        """
        loss_all = []
        y_true_all = []
        y_prob_all = []
        patient_ids = []
        if additional_outputs is not None:
            additional_outputs = {k: [] for k in additional_outputs}
        for data in tqdm(dataloader, desc="Evaluation"):
            self.model.eval()
            with torch.no_grad():
                output = self.model(**data)
                loss = output["loss"]
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()
                loss_all.append(loss.item())
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
                if additional_outputs is not None:
                    for key in additional_outputs.keys():
                        additional_outputs[key].append(output[key].cpu().numpy())
            if return_patient_ids:
                patient_ids.extend(data["patient_id"])
        loss_mean = sum(loss_all) / len(loss_all)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        outputs = [y_true_all, y_prob_all, loss_mean]
        if additional_outputs is not None:
            additional_outputs = {key: np.concatenate(val)
                                  for key, val in additional_outputs.items()}
            outputs.append(additional_outputs)
        if return_patient_ids:
            outputs.append(patient_ids)
        return outputs

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluates the model.

        Args:
            dataloader: Dataloader for evaluation.

        Returns:
            scores: a dictionary of scores.
        """
        if self.model.mode is not None:
            y_true_all, y_prob_all, loss_mean = self.inference(dataloader)
            mode = self.model.mode
            metrics_fn = get_metrics_fn(mode)
            scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics)
            scores["loss"] = loss_mean
        else:
            loss_all = []
            for data in tqdm(dataloader, desc="Evaluation"):
                self.model.eval()
                with torch.no_grad():
                    output = self.model(**data)
                    loss = output["loss"]
                    loss_all.append(loss.item())
            loss_mean = sum(loss_all) / len(loss_all)
            scores = {"loss": loss_mean}
        return scores

    def save_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = self.model.state_dict()
        torch.save(state_dict, ckpt_path)
        return

    def load_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms

    from pyhealth.datasets.utils import collate_fn_dict


    class MNISTDataset(Dataset):
        def __init__(self, train=True):
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            self.dataset = datasets.MNIST(
                "../data", train=train, download=True, transform=transform
            )

        def __getitem__(self, index):
            x, y = self.dataset[index]
            return {"x": x, "y": y}

        def __len__(self):
            return len(self.dataset)


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.mode = "multiclass"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x, y, **kwargs):
            x = torch.stack(x, dim=0).to(self.device)
            y = torch.tensor(y).to(self.device)
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            loss = self.loss(x, y)
            y_prob = torch.softmax(x, dim=1)
            return {"loss": loss, "y_prob": y_prob, "y_true": y}


    train_dataset = MNISTDataset(train=True)
    val_dataset = MNISTDataset(train=False)

    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn_dict, batch_size=64, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn_dict, batch_size=64, shuffle=False
    )

    model = Model()

    trainer = Trainer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        monitor="accuracy",
        epochs=5,
        test_dataloader=val_dataloader,
    )
