"""
Author: Yunpeng Zhao    
NetID: yz101 (if applicable)
Paper Title: Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit
Paper Link: https://arxiv.org/pdf/2007.09483.pdf

Description:
This module implements a PyHealth-compatible version of the Temporal Pointwise
Convolutional Networks (TPCN) model. It integrates the original model logic for
predicting ICU length of stay (LOS) directly inside this class. TPCN combines temporal
convolutions, pointwise operations, and skip connections as proposed in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel
from typing import Dict, Any


class TPCN(BaseModel):
    """
    Temporal Pointwise Convolutional Network for Length of Stay prediction in ICU settings.
    Combines dilated temporal convolutions with per-timestep pointwise MLP layers,
    and integrates static (flat) and diagnosis features at each step.

    Args:
        dataset (SampleEHRDataset): The dataset used in PyHealth.
        config (dict): Configuration with keys:
            - F: number of temporal features (value + decay)
            - D: diagnosis feature size
            - S: static flat feature size
            - Z: pointwise output dimension
            - Y: temporal convolution output dimension
            - kernel_size: convolutional kernel size (default: 3)
            - n_layers: number of stacked TPC blocks (default: 2)
    """
    def __init__(self, dataset, config):
        super().__init__(dataset, feature_keys=["X", "diagnoses", "flat"], label_key="los")

        # Configuration
        self.F = config["F"]
        self.D = config["D"]
        self.S = config["S"]
        self.Z = config["Z"]
        self.Y = config["Y"]
        self.K = config.get("kernel_size", 3)
        self.L = config.get("n_layers", 2)

        # Model layers
        self.temporal_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()

        for l in range(self.L):
            dilation = l + 1
            padding = dilation * (self.K - 1)
            in_channels = self.F if l == 0 else self.Y + self.Z

            self.temporal_layers.append(nn.Sequential(
                nn.ConstantPad1d((padding, 0), 0),
                nn.Conv1d(in_channels, self.Y, kernel_size=self.K, dilation=dilation),
                nn.BatchNorm1d(self.Y),
                nn.ReLU()
            ))

            self.pointwise_layers.append(nn.Sequential(
                nn.Linear(self.F + self.D + self.S, self.Z),
                nn.ReLU()
            ))

        self.output_head = nn.Sequential(
            nn.Linear(self.Y + self.Z + self.D + self.S, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Hardtanh(min_val=1 / 48, max_val=100)
        )

    def forward(self, **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            kwargs: Dictionary containing the following keys:
                - X: temporal input, [B, F, T] as list[list[float]]
                - diagnoses: diagnosis feature, [B, D]
                - flat: static feature, [B, S]
                - los: target length of stay, [B, T]
                - mask: mask over valid positions, [B, T]
                - seq_length: sequence lengths, list[int]

        Returns:
            dict with keys:
                - loss: computed MSE loss
                - y_prob: predicted LoS values
                - y_true: ground truth
                - logit: raw outputs
        """
        X = torch.stack([torch.tensor(x, dtype=torch.float32) for x in kwargs["X"]])
        diagnoses = torch.tensor(kwargs["diagnoses"], dtype=torch.float32)
        flat = torch.tensor(kwargs["flat"], dtype=torch.float32)
        y_true = torch.tensor(kwargs["los"], dtype=torch.float32)
        mask = torch.tensor(kwargs["mask"], dtype=torch.bool)

        seq_len = kwargs["seq_length"]
        if isinstance(seq_len, list):
            seq_len = torch.tensor(seq_len, dtype=torch.long)
        elif isinstance(seq_len, int):
            seq_len = torch.tensor([seq_len], dtype=torch.long)

        B, F, T = X.shape
        h = X

        for l in range(self.L):
            temp_out = self.temporal_layers[l](h)

            diag_rep = diagnoses.unsqueeze(2).expand(-1, -1, T)
            flat_rep = flat.unsqueeze(2).expand(-1, -1, T)
            x_orig = X if l == 0 else h[:, :self.F, :]

            point_input = torch.cat([x_orig, diag_rep, flat_rep], dim=1)
            point_input = point_input.permute(0, 2, 1).reshape(B * T, -1)
            point_out = self.pointwise_layers[l](point_input).view(B, T, -1).permute(0, 2, 1)

            h = torch.cat([temp_out, point_out], dim=1)

        h_final = h.permute(0, 2, 1).reshape(B * T, -1)
        diag_flat = diagnoses.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)
        flat_flat = flat.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)
        final_input = torch.cat([h_final, diag_flat, flat_flat], dim=1)

        los = self.output_head(final_input).view(B, T)
        loss = self._compute_loss(los, y_true, mask, seq_len)

        return {"loss": loss, "y_prob": los, "y_true": y_true, "logit": los}

    def _compute_loss(self, y_hat, y, mask, seq_length):
        """
        Computes masked MSE loss on log-transformed values.

        Args:
            y_hat: predicted LoS [B, T]
            y: true LoS [B, T]
            mask: validity mask [B, T]
            seq_length: sequence lengths [B]

        Returns:
            scalar tensor loss
        """
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        y = y.where(mask, torch.zeros_like(y))
        loss = F.mse_loss(torch.log1p(y_hat), torch.log1p(y), reduction='none')
        loss = torch.sum(loss, dim=1) / seq_length.clamp(min=1)
        return loss.mean()


if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset, get_dataloader

    # Create a minimal mock dataset
    samples = []
    for i in range(1):
        T = 48
        sample = {
            "patient_id": str(i),
            "visit_id": str(i),
            "X": torch.rand(2, T).tolist(),
            "diagnoses": torch.rand(64).tolist(),
            "flat": torch.rand(32).tolist(),
            "los": (torch.rand(T) * 10 + 1).tolist(),
            "mask": [1] * T,
            "seq_length": T
        }
        samples.append(sample)

    dataset = SampleEHRDataset(samples=samples, dataset_name="demo")

    config = {
        "F": 2,
        "D": 64,
        "S": 32,
        "Z": 16,
        "Y": 16,
        "kernel_size": 3,
        "n_layers": 2
    }

    model = TPCN(dataset, config)
    model.eval()

    # Run a test batch through the model
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    output = model(**batch)

    print("✅ Loss:", output["loss"].item())
    print("✅ Output shape:", output["y_prob"].shape)
