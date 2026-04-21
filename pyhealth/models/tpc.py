"""
Temporal Pointwise Convolution (TPC) Model for ICU Length-of-Stay Prediction

Contributors:
    - [TODO: Pankaj Meghani, Tarak Jha, Pranash Krishnan]
    - [TODO: meghani3tarakj2, pranash2]

Paper:
    Title: Temporal Pointwise Convolutional Networks for Length of Stay 
           Prediction in the Intensive Care Unit
    Authors: Emma Rocheteau, Pietro Liò, Stephanie Hyland
    Conference: CHIL 2021 (Conference on Health, Inference, and Learning)
    Link: https://arxiv.org/abs/2007.09483
    
Description:
    Implementation of the TPC model which combines grouped temporal convolutions
    with pointwise (1x1) convolutions for irregularly sampled multivariate time
    series in ICU settings. The model predicts remaining length of stay at hourly
    intervals throughout ICU admission.
    
    Novel Extension: Monte Carlo Dropout uncertainty estimation for predictive
    confidence intervals (not in original paper).

Usage:
    >>> from pyhealth.models import TPC
    >>> from pyhealth.datasets import MIMIC4EHRDataset
    >>> from pyhealth.tasks import RemainingLOSMIMIC4
    >>> 
    >>> dataset = mimic4.set_task(RemainingLOSMIMIC4())
    >>> model = TPC(dataset=dataset, n_layers=3, use_msle=True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel


class MSLELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.squared_error = nn.MSELoss(reduction="none")

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, seq_length: torch.Tensor, sum_losses: bool = False) -> torch.Tensor:
        mask = mask.bool()
        eps = 1e-8
        log_y_hat = torch.where(mask, torch.log(y_hat.clamp_min(eps)), torch.zeros_like(y_hat))
        log_y = torch.where(mask, torch.log(y.clamp_min(eps)), torch.zeros_like(y))
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1).float()
        return loss.mean()


class MaskedMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.squared_error = nn.MSELoss(reduction="none")

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, seq_length: torch.Tensor, sum_losses: bool = False) -> torch.Tensor:
        mask = mask.bool()
        y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y = torch.where(mask, y, torch.zeros_like(y))
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1).float()
        return loss.mean()


class TPC(BaseModel):
    def __init__(
        self,
        dataset: SampleDataset,
        timeseries_key: str = "timeseries",
        static_key: Optional[str] = "static",
        conditions_key: Optional[str] = "conditions",
        n_layers: int = 3,
        kernel_size: int = 4,
        temp_kernels: Optional[Sequence[int]] = None,
        point_sizes: Optional[Sequence[int]] = None,
        diagnosis_size: int = 64,
        last_linear_size: int = 64,
        main_dropout_rate: float = 0.3,
        temp_dropout_rate: float = 0.3,
        time_before_pred: int = 5,
        use_msle: bool = True,
        sum_losses: bool = False,
        apply_exp: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)
        self.mode = "regression"

        if len(self.label_keys) != 1:
            raise ValueError("tpc supports exactly one label key")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.label_key = self.label_keys[0]
        self.timeseries_key = timeseries_key
        self.static_key = static_key if static_key in self.feature_keys else None
        self.conditions_key = (
            conditions_key if conditions_key in self.feature_keys else None
        )

        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.temp_kernels: List[int] = (
            list(temp_kernels) if temp_kernels is not None else [8] * n_layers
        )
        self.point_sizes: List[int] = (
            list(point_sizes) if point_sizes is not None else [14] * n_layers
        )
        if len(self.temp_kernels) != n_layers:
            raise ValueError("temp_kernels must have exactly n_layers entries")
        if len(self.point_sizes) != n_layers:
            raise ValueError("point_sizes must have exactly n_layers entries")

        self.diagnosis_size = diagnosis_size
        self.last_linear_size = last_linear_size
        self.main_dropout_rate = main_dropout_rate
        self.temp_dropout_rate = temp_dropout_rate
        self.time_before_pred = time_before_pred
        self.use_msle = use_msle
        self.sum_losses = sum_losses
        self.apply_exp = apply_exp

        self.relu = nn.ReLU()

        self.hardtanh = nn.Hardtanh(min_val=1.0 / 48.0, max_val=100.0)
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)

        self.loss_fn: nn.Module = MSLELoss() if use_msle else MaskedMSELoss()

        sample = dataset[0]
        if self.timeseries_key not in sample:
            raise KeyError(
                f"timeseries_key '{self.timeseries_key}' not found in dataset sample"
                f"available keys: {list(sample.keys())}"
            )
        ts_sample: torch.Tensor = sample[self.timeseries_key]
        if ts_sample.dim() != 2:
            raise ValueError(
                f"Each timeseries sample must be 2-D (channels, time) or (time, channels), "
                f"got shape {tuple(ts_sample.shape)}"
            )

        num_channels = min(ts_sample.shape)
        if (num_channels - 2) % 2 != 0 or num_channels < 4:
            raise ValueError(
                "timeseries channel dimension must equal 2F+2 with F >= 1"
                f"Detected smallest dim = {num_channels}."
            )
        self.F: int = (num_channels - 2) // 2

        self.no_flat_features: int = 0
        if self.static_key is not None:
            static_sample: torch.Tensor = sample[self.static_key]
            self.no_flat_features = (
                1 if static_sample.dim() == 0 else int(static_sample.shape[-1])
            )

        self.D: int = 0
        self.diagnosis_encoder: Optional[nn.Linear] = None
        self.bn_diagnosis_encoder: Optional[nn.BatchNorm1d] = None

        if self.conditions_key is not None:
            self.D = self.dataset.input_processors[self.conditions_key].size()
            self.diagnosis_encoder = nn.Linear(self.D, self.diagnosis_size)
            self.bn_diagnosis_encoder = nn.BatchNorm1d(self.diagnosis_size)

      
        self.bn_point_last_los = nn.BatchNorm1d(self.last_linear_size)
        self._init_tpc()
        self.point_final_los = nn.Linear(self.last_linear_size, 1)

    def _init_tpc(self) -> None:
        self._layer_info: List[Dict[str, Any]] = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1
            padding = [(self.kernel_size - 1) * dilation, 0]
            self._layer_info.append(
                {
                    "temp_kernels": self.temp_kernels[i],
                    "point_size": self.point_sizes[i],
                    "dilation": dilation,
                    "padding": padding,
                    "stride": 1,
                }
            )

        self._create_temp_pointwise_layers()

        input_size = (
            (self.F + self._Zt) * (1 + self._Y)
            + self.diagnosis_size
            + self.no_flat_features
        )
        self.point_last_los = nn.Linear(input_size, self.last_linear_size)

    def _create_temp_pointwise_layers(self) -> None:
        self.layer_modules = nn.ModuleDict()
        Y = 0   
        Z = 0   
        Zt = 0 

        for i in range(self.n_layers):
            temp_in = (self.F + Zt) * (1 + Y) if i > 0 else 2 * self.F

            temp_out = (self.F + Zt) * self.temp_kernels[i]

            point_in = (
                (self.F + Zt - Z) * Y
                + Z
                + 2 * self.F
                + 2
                + self.no_flat_features
            )
            point_out = self.point_sizes[i]

            self.layer_modules[str(i)] = nn.ModuleDict(
                {
                    "temp": nn.Conv1d(
                        in_channels=temp_in,
                        out_channels=temp_out,
                        kernel_size=self.kernel_size,
                        stride=self._layer_info[i]["stride"],
                        dilation=self._layer_info[i]["dilation"],
                        groups=self.F + Zt,
                    ),
                    "bn_temp": nn.BatchNorm1d(temp_out),
                    "point": nn.Linear(point_in, point_out),
                    "bn_point": nn.BatchNorm1d(point_out),
                }
            )

            Y = self.temp_kernels[i]
            Z = point_out
            Zt += Z

        self._Y = Y
        self._Zt = Zt


    def _normalize_timeseries(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        if x.dim() != 3:
            raise ValueError(
                f"expected a 3-D batched timeseries (B, C, T) or (B, T, C), "
                f"got shape {tuple(x.shape)}."
            )
        expected_c = 2 * self.F + 2
        if x.shape[1] == expected_c:
            return x  
        if x.shape[2] == expected_c:
            return x.transpose(1, 2) 
        raise ValueError(
            f"cannot identify channel dimension of size {expected_c} in "
            f"timeseries shape {tuple(x.shape)}."
        )

    def _prepare_static(self, batch_size: int, kwargs: Dict[str, Any]) -> torch.Tensor:
        if self.static_key is None or self.no_flat_features == 0:
            return torch.zeros(batch_size, 0, device=self.device)
        flat = kwargs[self.static_key].to(self.device, dtype=torch.float32)
        if flat.dim() == 1:
            flat = flat.unsqueeze(-1)
        return flat

    def _prepare_diagnoses(self, batch_size: int, kwargs: Dict[str, Any]) -> torch.Tensor:
        if self.conditions_key is None or self.D == 0 or self.diagnosis_encoder is None:
            return torch.zeros(batch_size, self.diagnosis_size, device=self.device)

        codes = kwargs[self.conditions_key].to(self.device)  
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)

        multi_hot = torch.zeros(batch_size, self.D, device=self.device)
        valid = codes >= 0  
        safe_codes = codes.masked_fill(~valid, 0)
        multi_hot.scatter_add_(1, safe_codes, valid.float())
        multi_hot[:, 0] = 0.0 

        diag_enc = self.relu(
            self.main_dropout(
                self.bn_diagnosis_encoder(self.diagnosis_encoder((multi_hot > 0).float()))
            )
        )
        return diag_enc

    def _temp_pointwise(
        self,
        B: int,
        T: int,
        X: torch.Tensor,
        X_orig: torch.Tensor,
        repeat_flat: torch.Tensor,
        temp: nn.Conv1d,
        bn_temp: nn.BatchNorm1d,
        point: nn.Linear,
        bn_point: nn.BatchNorm1d,
        temp_kernels: int,
        padding: List[int],
        prev_temp: Optional[torch.Tensor],
        prev_point: Optional[torch.Tensor],
        point_skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_padded = F.pad(X, padding, "constant", 0)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  

        C_feat_groups = X_temp.shape[1] // temp_kernels  

        concat_parts: List[torch.Tensor] = []
        if prev_temp is not None:
            concat_parts.append(prev_temp)
        if prev_point is not None:
            concat_parts.append(prev_point)
        concat_parts.append(X_orig)
        if repeat_flat.shape[1] > 0:
            concat_parts.append(repeat_flat)
        X_concat = torch.cat(concat_parts, dim=1)  
        point_out = self.main_dropout(bn_point(point(X_concat)))

        if prev_point is not None:
            Z_prev = prev_point.shape[1]
            point_skip = torch.cat(
                [point_skip, prev_point.view(B, T, Z_prev).permute(0, 2, 1)],
                dim=1,
            )  

        temp_4d = X_temp.view(B, C_feat_groups, temp_kernels, T)  
        skip_4d = point_skip.unsqueeze(2)                          
        temp_stack = torch.cat([skip_4d, temp_4d], dim=2) 

        point_size = point_out.shape[1]
        point_4d = (
            point_out.view(B, T, point_size)
            .permute(0, 2, 1)          
            .unsqueeze(2)              
            .expand(-1, -1, 1 + temp_kernels, -1)  
        )

        combined = self.relu(
            torch.cat([temp_stack, point_4d], dim=1)
        )  

        next_X = combined.view(B, (C_feat_groups + point_size) * (1 + temp_kernels), T)

        temp_out = (
            X_temp.permute(0, 2, 1)                   
            .contiguous()
            .view(B * T, C_feat_groups * temp_kernels)
        )

        return temp_out, point_out, next_X, point_skip


    def forward(self, return_full_sequence: bool = False, **kwargs: Any) -> Dict[str, torch.Tensor]:
        X = self._normalize_timeseries(kwargs[self.timeseries_key]) 
        B, _, T = X.shape

        if T <= self.time_before_pred:
            raise ValueError(
                f"Sequence length T={T} must be greater than "
                f"time_before_pred={self.time_before_pred}."
            )

        flat = self._prepare_static(B, kwargs)            
        diagnoses_enc = self._prepare_diagnoses(B, kwargs) 

        X_orig = X.permute(0, 2, 1).contiguous().view(B * T, 2 * self.F + 2)
        repeat_flat = flat.repeat_interleave(T, dim=0)   

        values = X[:, 1 : self.F + 1, :]      
        decay = X[:, self.F + 1 : 2 * self.F + 1, :]  

        next_X = torch.stack([values, decay], dim=2).reshape(B, 2 * self.F, T)

        point_skip = values  

        prev_temp: Optional[torch.Tensor] = None
        prev_point: Optional[torch.Tensor] = None

        for i in range(self.n_layers):
            mods = self.layer_modules[str(i)]
            prev_temp, prev_point, next_X, point_skip = self._temp_pointwise(
                B=B,
                T=T,
                X=next_X,
                X_orig=X_orig,
                repeat_flat=repeat_flat,
                temp=mods["temp"],
                bn_temp=mods["bn_temp"],
                point=mods["point"],
                bn_point=mods["bn_point"],
                temp_kernels=self._layer_info[i]["temp_kernels"],
                padding=self._layer_info[i]["padding"],
                prev_temp=prev_temp,
                prev_point=prev_point,
                point_skip=point_skip,
            )

        post_hist = T - self.time_before_pred  

        ts_features = (
            next_X[:, :, self.time_before_pred :]   
            .permute(0, 2, 1)                       
            .contiguous()
            .view(B * post_hist, -1)               
        )

        combined_features = torch.cat(
            [
                flat.repeat_interleave(post_hist, dim=0),             
                diagnoses_enc.repeat_interleave(post_hist, dim=0),    
                ts_features,                                           
            ],
            dim=1,
        )

        last_hidden = self.relu(
            self.main_dropout(
                self.bn_point_last_los(self.point_last_los(combined_features))
            )
        )  

        raw_pred = self.point_final_los(last_hidden).view(B, post_hist)  
        if self.apply_exp:
            raw_pred = torch.exp(raw_pred)
        los_pred = self.hardtanh(raw_pred)  

        output: Dict[str, torch.Tensor] = {
            "logit": los_pred if return_full_sequence else los_pred.reshape(-1),
            "y_prob": los_pred if return_full_sequence else los_pred.reshape(-1),
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device, dtype=torch.float32)
            if y_true.dim() == 3 and y_true.shape[-1] == 1:
                y_true = y_true.squeeze(-1)
            y_true_post = y_true[:, self.time_before_pred :]  

            mask = y_true_post > 0
            seq_lengths = mask.sum(dim=1)
            loss = self.loss_fn(los_pred, y_true_post, mask, seq_lengths, self.sum_losses)

            output["loss"] = loss
            if return_full_sequence:
                output["y_true"] = y_true_post
                output["mask"] = mask
            else:
                flat_mask = mask.reshape(-1)
                output["y_true"] = los_pred.reshape(-1)[flat_mask]
                output["y_prob"] = los_pred.reshape(-1)[flat_mask]
                output["logit"] = los_pred.reshape(-1)[flat_mask]
                output["y_true"] = y_true_post.reshape(-1)[flat_mask]

        if kwargs.get("embed", False):
            output["embed"] = combined_features.view(B, post_hist, -1).mean(dim=1)

        return output

    def predict_with_uncertainty(self, mc_samples: int = 30, **kwargs: Any) -> Dict[str, torch.Tensor]:
    
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1.")

        was_training = self.training
        self.train()  

        samples: List[torch.Tensor] = []
        mask: Optional[torch.Tensor] = None
        y_true: Optional[torch.Tensor] = None

        with torch.no_grad():
            for _ in range(mc_samples):
                out = self.forward(return_full_sequence=True, **kwargs)
                samples.append(out["y_prob"])
                if mask is None:
                    mask = out.get("mask")
                if y_true is None:
                    y_true = out.get("y_true")

        if not was_training:
            self.eval()

        stacked = torch.stack(samples, dim=0) 
        result: Dict[str, torch.Tensor] = {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "samples": stacked,
        }
        if mask is not None:
            result["mask"] = mask
        if y_true is not None:
            result["y_true"] = y_true
        return result