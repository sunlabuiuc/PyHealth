import logging
import os
from typing import Dict, Type, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

from pyhealth.utils import get_device, create_directory, set_logger
from pyhealth.evaluator import evaluate


def is_best(best_score: float, score: float, mode: str) -> bool:
    if mode == "max":
        return score > best_score
    elif mode == "min":
        return score < best_score
    else:
        raise ValueError(f"mode {mode} is not supported")


class Trainer:
    def __init__(
        self,
        device: Optional[str] = None,
        enable_cuda: bool = True,
        enable_logging: bool = True,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
    ):
        # self.device = get_device(enable_cuda=enable_cuda)
        self.device = device
        if enable_logging:
            self.exp_path = set_logger(output_path, exp_name)
        else:
            self.exp_path = None

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Dict[str, object] = {"lr": 1e-3},
        val_loader: DataLoader = None,
        val_metric=None,
        mode: str = "max",
        epochs: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        show_progress_bar: bool = True,
    ):
        if self.exp_path is not None:
            create_directory(os.path.join(self.exp_path))

        steps_per_epoch = len(train_loader)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        data_iterator = iter(train_loader)
        best_score = -1 * float("inf") if mode == "max" else float("inf")
        global_step = 0

        # for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        for epoch in range(epochs):
            training_loss = []

            model.zero_grad()
            model.train()

            for _ in trange(
                steps_per_epoch,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_loader)
                    data = next(data_iterator)

                output = model(**data, device=self.device, training=True)
                loss = output["loss"]
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                optimizer.zero_grad()

                training_loss.append(loss.item())
                global_step += 1

            # logging.info(f'--- Train epoch-{epoch}, step-{global_step} ---')
            # logging.info(f'loss: {sum(training_loss) / len(training_loss):.4f}')
            if self.exp_path is not None:
                self._save_ckpt(model, os.path.join(self.exp_path, "last.ckpt"))

            if val_metric is not None:
                y_gt, y_prod, y_pred = evaluate(model, val_loader, self.device)
                try:  # not sure the metric work for probability or predicted label
                    score = val_metric(y_gt, y_prod)
                except ValueError:
                    score = val_metric(y_gt, y_pred)
                print(f"{val_metric.__name__}: {score:.4f}")
                if is_best(best_score, score, mode):
                    best_score = score
                    if self.exp_path is not None:
                        self._save_ckpt(model, os.path.join(self.exp_path, "best.ckpt"))
        print("best_model_path:", os.path.join(self.exp_path, "best.ckpt"))

    def _save_ckpt(self, model, save_path):
        state_dict = model.state_dict()
        torch.save(state_dict, save_path)

    def load_best_model(self, model):
        path = os.path.join(self.exp_path, "best.ckpt")
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model
