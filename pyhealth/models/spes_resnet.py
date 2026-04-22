"""
SPES CNN baseline model (MSResNet).

Contributor: Sebastian Ho   
NetID: sho28
Paper Title: Localising the Seizure Onset Zone from Single-Pulse Electrical \
    Stimulation Responses with a CNN Transformer
Paper Link: https://proceedings.mlr.press/v252/norris24a.html
Description: Baseline MSResNet CNN model for SPES seizure-onset-zone \
    localisation.

Original Code: https://github.com/norrisjamie23/Localising_SOZ_from_SPES/
Baseline method adapted from https://github.com/geekfeiw/Multi-Scale-1D-ResNet
"""

from typing import Dict

import torch
import torch.nn as nn
# from torcheeg.transforms import RandomNoise
class RandomNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, eeg, **kwargs):
        if self.std > 0:
            return {'eeg': eeg + torch.randn_like(eeg) * self.std}
        return {'eeg': eeg}

from pyhealth.models.base_model import BaseModel


# -----------------------------------------------------------------------------
# MSResNet Helpers
# -----------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10, dropout_rate=0.2):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        self.drop = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256*3, num_classes)

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool7(z)

        # Align temporal length across branches for robust concatenation.
        # This avoids odd-length edge cases (e.g., 513 with distance included)
        # where one branch can retain length 2 while others collapse to 1.
        if x.shape[-1] != y.shape[-1] or y.shape[-1] != z.shape[-1]:
            x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
            y = torch.nn.functional.adaptive_avg_pool1d(y, 1)
            z = torch.nn.functional.adaptive_avg_pool1d(z, 1)

        out = torch.cat([x, y, z], dim=1)

        out = out[:, :, 0]#.squeeze()
        out = self.drop(out)
        # out1 = self.fc(out)

        return out


# -----------------------------------------------------------------------------
# SPESResNet Wrapper
# -----------------------------------------------------------------------------

class SPESResNet(BaseModel):
    """Multi-scale ResNet baseline for SPES-based SOZ localisation.

    Wraps the MSResNet (1D multi-branch ResNet) baseline from Norris et al. (ML4H
    2024) for binary seizure-onset-zone (SOZ) classification from cortico-cortical
    evoked potential (CCEP / SPES-style) tensors from PyHealth tasks.

    Expected batch keys (from SeizureOnsetZoneLocalisation or compatible tasks):

        * spes_responses: float tensor shaped
          (batch, max_channels, 2, timesteps + 1). The length-2 axis holds mean and
          std response modes; time index 0 may store per-channel distance (or padding)
          when enabled.
        * soz_label (or label_key): binary labels for the batch.

    Args:
        dataset: Task-processed SampleDataset; passed to BaseModel for loss
            and label schema.
        feature_keys: Keys to read from each sample. Default: ["spes_responses"].
        label_key: Supervision key. Default: "soz_label".
        mode: Optional mode override; if omitted, BaseModel infers from the
            dataset schema.
        input_channels: Channels sampled per example for the backbone. Default: 40.
        noise_std: Std of additive Gaussian noise on responses while training; 0
            disables. Default: 0.1.
        include_distance: If True, keep the distance column in the 1D MSResNet input;
            if False, use only the response time series. Default: False.
        **kwargs: Forwarded to MSResNet (e.g. layers, dropout_rate).

    Examples:
        >>> from pyhealth.datasets.respectccep import RESPectCCEPDataset
        >>> from pyhealth.tasks.ccep_detect_soz import SeizureOnsetZoneLocalisation
        >>> from pyhealth.models import SPESResNet
        >>> base = RESPectCCEPDataset(root="/path/to/respect_ccep")
        >>> sample_dataset = base.set_task(
        ...     SeizureOnsetZoneLocalisation(spes_mode="convergent")
        ... )
        >>> model = SPESResNet(dataset=sample_dataset, input_channels=40)
    """

    def __init__(
        self,
        dataset,
        feature_keys=None,
        label_key=None,
        mode=None,
        input_channels=40,
        noise_std=0.1,
        include_distance=False,
        **kwargs
    ):
        """Build MSResNet trunk and linear head for SOZ logits.

        Args:
            dataset: SampleDataset with input/output schema for the task.
            feature_keys: Feature keys; default ["spes_responses"].
            label_key: Label key; default "soz_label".
            mode: Optional mode override (otherwise inferred by BaseModel).
            input_channels: Subsampled channel count per forward pass. Default: 40.
            noise_std: Gaussian noise std on responses during training. Default: 0.1.
            include_distance: Include distance in the 1D input. Default: False.
            **kwargs: Extra args for MSResNet.
        """
        super(SPESResNet, self).__init__(
            dataset=dataset,
        )
        self.feature_keys = feature_keys or ["spes_responses"]
        self.label_key = label_key or "soz_label"
        if mode is not None:
            self.mode = mode
            
        self.input_channels = input_channels
        self.noise_std = noise_std
        self.include_distance = include_distance
        self.noise = RandomNoise(std=self.noise_std)  # Random noise transformer
        
        # Binary classification -> num_classes = 1 (outputting a single logit per sample)
        # The logits are passed to BCEWithLogitsLoss by BaseModel when mode="binary".
        num_classes = 1
        
        self.msresnets = MSResNet(input_channel=input_channels, num_classes=num_classes, **kwargs)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Run one forward pass (Trainer / inference).

        Args:
            **kwargs: At minimum spes_responses of shape
                (batch, max_C, 2, T + 1). For loss, also pass the label under
                label_key (default soz_label) as (batch,) or (batch, 1).

        Returns:
            Dict[str, torch.Tensor]: logit, y_prob, and optionally loss and
            y_true when labels are provided.
        """
        # Ensure batch tensors are on the same device as model weights.
        input_x = kwargs["spes_responses"].to(self.device)
        # [batch_size, max_C, 2, T+1] -> [batch_size, 2, max_C, T+1]
        x = input_x.transpose(1, 2)
        
        # Extract distances from dim 0
        distances = x[:, 0, :, 0]

        # Apply random noise to the non-distance channels during training.
        if self.training and self.noise_std > 0:
            x[:, :, :, 1:] = self.noise(eeg=x[:, :, :, 1:])['eeg']
        
        all_x = []

        # Process each sample in the batch individually to handle variable valid channels.
        for single_sample, distance in zip(x, distances):
            valid_rows = torch.where(distance != 0)[0]
            
            # If all are zero (e.g. dataset without coordinates fell back to 0),
            # use all possible non-zero rows based on time series.
            if len(valid_rows) == 0:
                # use std > 0 to find non-pad channels
                ts_std = single_sample[1, :, 1:]
                valid_rows = torch.where(ts_std.sum(dim=-1) != 0)[0]

            # Still blank
            if len(valid_rows) == 0:
                valid_rows = torch.arange(single_sample.shape[1], device=x.device)

            p = torch.ones(len(valid_rows), device=x.device) / len(valid_rows)
            idx = p.multinomial(
                num_samples=self.input_channels, 
                replacement=len(valid_rows) < self.input_channels
            )
            random_channels = valid_rows[idx]
            random_channels = random_channels.sort()[0]
            
            if self.include_distance:
                # Include distance (index 0) and the rest of the time series
                all_x.append(single_sample[0, random_channels, :])
            else:
                # Original logic: exclude the spatial distance entirely
                all_x.append(single_sample[0, random_channels, 1:])

        # Stack processed samples and pass them through the MSResNet and the final FC layer.
        processed_x = torch.stack(all_x, dim=0)
        
        # processed_x is [batch_size, input_channels, T]
        ms_out = self.msresnets(processed_x) # [batch_size, 768]
        logit = self.fc(ms_out)              # [batch_size, 1]

        # Squeeze for binary classification consistency
        if self.mode == "binary" and logit.shape[-1] == 1:
            logit = logit.squeeze(-1)
            
        y_true = kwargs.get(self.label_key)
        if y_true is not None:
            y_true = y_true.to(self.device)
        if self.mode == "binary" and y_true is not None and y_true.ndim > 1:
            y_true = y_true.squeeze(-1)
        loss_fn = self.get_loss_function()
        loss = loss_fn(logit, y_true.float() if self.mode == "binary" else y_true)
        y_prob = self.prepare_y_prob(logit)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }
