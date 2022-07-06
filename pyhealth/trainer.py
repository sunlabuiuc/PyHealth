import logging
from typing import Dict, Type, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

import pyhealth.optimization as optimization


class Trainer:

    def __init__(
            self,
            device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))

        self.device = torch.device(device)

    def fit(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            evaluator: nn.Module = None,
            eval_dataloader: DataLoader = None,
            epochs: int = 1,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 1000,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 1000,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
    ):
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        model.to(self.device)
        evaluator.to(self.device)

        self.best_score = -9999999

        steps_per_epoch = len(train_dataloader)
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                        t_total=num_train_steps)

        global_step = 0
        data_iterator = iter(train_dataloader)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            model.zero_grad()
            model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                features, labels = data
                labels = labels.to(self.device)
                features = features.to(self.device)

                if use_amp:
                    with autocast():
                        loss_value = model(features, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value = model(features, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    evaluator.evaluate(eval_dataloader, device=self.device)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler.
        Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return optimization.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return optimization.get_constant_schedule_with_warmup(optimizer,
                                                                  num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return optimization.get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return optimization.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                   num_warmup_steps=warmup_steps,
                                                                                   num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))
