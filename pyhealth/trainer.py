import json
import logging
import os
import time
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


def _vram_stats(device: str) -> Dict[str, float]:
    """Returns current and peak VRAM usage in MB for a CUDA device."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {}
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return {"vram_allocated_mb": allocated, "vram_peak_mb": peak}


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
        patience=None,
        accumulation_steps: int = 1,
        use_amp: bool = False,
        amp_dtype: str = "bf16",
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
            patience: Number of epochs to wait for improvement before early stopping.
                Default is None, which means no early stopping.
            accumulation_steps: Gradient accumulation steps to simulate a larger
                effective batch size. Default is 1 (no accumulation).
            use_amp: Whether to use automatic mixed precision. Default is False.
            amp_dtype: AMP dtype — "bf16" (stable, recommended) or "fp16".
                Default is "bf16".
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        _amp_dtype = (
            torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        )
        # GradScaler only needed for fp16; bf16 has fp32 dynamic range
        scaler = (
            torch.cuda.amp.GradScaler()
            if (use_amp and _amp_dtype == torch.float16)
            else None
        )

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
        logger.info(f"Patience: {patience}")
        logger.info(f"Accumulation steps: {accumulation_steps}")
        logger.info(f"AMP: {use_amp} (dtype={amp_dtype})")

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
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0
        patience_counter = 0
        metrics_history: List[Dict] = []
        train_start = time.perf_counter()

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_start = time.perf_counter()
            # batch training loop
            logger.info("")
            for step_idx in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} / {epochs}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward (with optional AMP)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=_amp_dtype):
                        output = self.model(**data)
                        loss = output["loss"] / accumulation_steps
                else:
                    output = self.model(**data)
                    loss = output["loss"] / accumulation_steps
                # backward
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                training_loss.append(loss.item() * accumulation_steps)
                # optimizer step every accumulation_steps batches or epoch end
                is_update_step = (
                    (step_idx + 1) % accumulation_steps == 0
                    or (step_idx + 1) == steps_per_epoch
                )
                if is_update_step:
                    if max_grad_norm is not None:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            epoch_time = time.perf_counter() - epoch_start
            vram = _vram_stats(self.device)

            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            logger.info(f"epoch_time: {epoch_time:.2f}s")
            if vram:
                logger.info(
                    f"vram_peak: {vram['vram_peak_mb']:.1f} MB  "
                    f"vram_current: {vram['vram_allocated_mb']:.1f} MB"
                )
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            epoch_record: Dict = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": sum(training_loss) / len(training_loss),
                "epoch_time_s": round(epoch_time, 3),
                **{f"train_{k}": v for k, v in vram.items()},
            }

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                epoch_record.update({f"val_{k}": v for k, v in scores.items()})
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        patience_counter = 0
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))
                    else:
                        patience_counter += 1
                        # early stopping
                        if patience is not None and patience_counter >= patience:
                            logger.info(
                                f"Early stopping at epoch-{epoch}, step-{global_step}"
                            )
                            metrics_history.append(epoch_record)
                            break

            metrics_history.append(epoch_record)

        total_time = time.perf_counter() - train_start
        logger.info(f"--- Training complete: {total_time:.2f}s total ---")

        # persist metrics history
        if self.exp_path is not None:
            history_path = os.path.join(self.exp_path, "metrics_history.json")
            with open(history_path, "w") as f:
                json.dump(metrics_history, f, indent=2)
            logger.info(f"Metrics history saved to {history_path}")

        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
            os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info("--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))

        return metrics_history

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
        self.model.eval()
        for data in tqdm(dataloader, desc="Evaluation"):
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
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
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
