"""
SPES CNN-Transformer model.

Contributor: Sebastian Ho
NetID: sho28
Paper Title: Localising the Seizure Onset Zone from Single-Pulse Electrical \
    Stimulation Responses with a CNN Transformer
Paper Link: https://proceedings.mlr.press/v252/norris24a.html
Description: Convolutional-Transformer encoder for SPES seizure-onset-zone \
    localisation (Norris et al. 2024).

Original Code: https://github.com/norrisjamie23/Localising_SOZ_from_SPES/
"""

import random
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
from pyhealth.models.spes_resnet import MSResNet


class SPESResponseEncoder(nn.Module):
    """Convolutional and transformer encoder for SPES / CCEP SOZ classification.

    Embeds each recording channel with an optional MSResNet branch and/or flattened
    statistics, projects to embedding_dim, then applies nn.TransformerEncoder
    with a learned class token.

    Inputs match the layout used inside SPESTransformer.forward after the batch
    transpose: tensor x shaped (batch, 2, n_channels, timesteps + 1) (mean and
    std modes on the length-2 axis; time index 0 may hold distance).

    Args:
        mean: If True, use the mean response mode in embeddings.
        std: If True, use the std response mode in embeddings.
        conv_embedding: Enable MSResNet per channel. Default: True.
        mlp_embedding: Concatenate early time slices with conv features. Default: True.
        dropout_rate: Dropout on projection and transformer. Default: 0.5.
        num_layers: Transformer encoder depth. Default: 2.
        embedding_dim: Hidden size / patch width. Default: 64.
        random_channels: If set, subsample exactly this many channels. Default: None.
        noise_std: Gaussian noise on responses while training. Default: 0.1.
        include_distance: Keep distance at time index 0. Default: True.

    Raises:
        AssertionError: If mean and std are both False.
    """

    def __init__(
        self,
        mean: bool,
        std: bool,
        conv_embedding: bool = True,
        mlp_embedding: bool = True,
        dropout_rate: float = 0.5,
        num_layers: int = 2,
        embedding_dim: int = 64,
        random_channels=None,
        noise_std: float = 0.1,
        include_distance: bool = True,
    ):
        """Create MSResNet (optional), linear patch embedding, and transformer stack.

        Args:
            mean: Use mean mode in the embedding path.
            std: Use std mode in the embedding path.
            conv_embedding: Run MSResNet branch. Default: True.
            mlp_embedding: Add MLP-style time features when conv is on. Default: True.
            dropout_rate: Dropout probability. Default: 0.5.
            num_layers: Number of encoder layers. Default: 2.
            embedding_dim: Model width. Default: 64.
            random_channels: Fixed channel count per sample, or None. Default: None.
            noise_std: Noise std on responses in training. Default: 0.1.
            include_distance: Keep distance column. Default: True.
        """
        super(SPESResponseEncoder, self).__init__()

        assert mean or std, "Either mean or std (or both) must be True for embedding."

        self.mean = mean
        self.std = std
        self.conv_embedding = conv_embedding
        self.mlp_embedding = mlp_embedding
        self.random_channels = random_channels
        self.noise_std = noise_std
        self.include_distance = include_distance
        
        self.noise = RandomNoise(std=self.noise_std)

        # Distances are optionally stripped (1 padding unit reduction)
        offset = 0 if self.include_distance else 1

        if conv_embedding:
            input_channels = self.mean + self.std
            self.msresnet = MSResNet(input_channel=input_channels, num_classes=1)
            # MSResNet output size is 768
            embedding_in = 768 + (self.mean + self.std) * (155 - offset) * mlp_embedding
        else:
            embedding_in = (self.mean + self.std) * (509 - offset)

        self.patch_to_embedding = nn.Linear(embedding_in, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.class_token = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1, embedding_dim)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=embedding_dim // 8, 
            dim_feedforward=embedding_dim * 2, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """Encode SPES tensors to one vector per batch element.

        Args:
            x: Tensor (batch, 2, n_channels, timesteps + 1) (see class docstring).

        Returns:
            Class-token outputs, shape (batch, embedding_dim).
        """
        if self.training:
            x = self.apply_noise_and_zero_channels(x)
        
        if self.random_channels:
            distances = x[:, 0, :, 0]
            all_x = []

            for sample_idx, (single_sample, distance) in enumerate(zip(x, distances)):
                valid_rows = torch.where(distance != 0)[0]
                if len(valid_rows) == 0:
                    ts_std = single_sample[1, :, 1:]
                    valid_rows = torch.where(ts_std.sum(dim=-1) != 0)[0]

                if len(valid_rows) == 0:
                    valid_rows = torch.arange(single_sample.shape[1], device=x.device)

                if len(valid_rows) < self.random_channels:
                    idx = torch.randint(0, len(valid_rows), (self.random_channels,), device=x.device)
                
                else:
                    idx = torch.randperm(len(valid_rows), device=x.device)[:self.random_channels]
                random_channels_idx = valid_rows[idx].sort()[0]
                all_x.append(single_sample[:, random_channels_idx, :])
            # Stack processed samples and pass them through the MSResNet and the final layer.    
            x = torch.stack(all_x, dim=0)

        # Distances from the first mode 
        distances = x[:, 0, :, 0]
        key_padding_mask = self.create_key_padding_mask(distances)

        all_output = self.prepare_channels(x)
        x = self.dropout(self.patch_to_embedding(all_output))

        weight = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((weight, x), dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Return the class token output
        return x[:, 0]

    def apply_noise_and_zero_channels(self, x):
        """Augment training inputs with noise and random channel dropout.

        Called from forward when self.training is True.

        Args:
            x: Tensor (batch, modes, channels, timesteps).

        Returns:
            Augmented tensor with the same shape as x.
        """
        # Implementation of noise application and zeroing random channels
        non_zero_indices = torch.nonzero(x[:, 0, :, 0].sum(axis=0), as_tuple=False).squeeze(-1)

        if len(non_zero_indices) > 0:
            # Step 1: Uniformly sample a number from 0 to the length of non_zero_indices
            sample_size = random.randint(0, len(non_zero_indices) // 2)
            if sample_size > 0:
                # Step 2: Select a random sample of this number from non_zero_indices without replacement
                random_indices = torch.randperm(len(non_zero_indices))[:sample_size]
                random_sample = non_zero_indices[random_indices]
                # Set these to zero
                x[:, :, random_sample] = 0

        if self.noise_std > 0:
            x[:, :, :, 1:] = self.noise(eeg=x[:, :, :, 1:])['eeg']

        return x

    def create_key_padding_mask(self, distances):
        """Build a boolean padding mask for TransformerEncoder.

        Prepends False for the class token column. Called from forward.

        Args:
            distances: Tensor (batch, n_channels); zero marks padding.

        Returns:
            Boolean mask shaped (batch, 1 + n_channels) suitable for
            src_key_padding_mask.
        """
        key_padding_mask = (distances == 0)

        # Prepend a false column for the class token
        false_column = torch.zeros(distances.size(0), 1, dtype=torch.bool, device=distances.device)
        key_padding_mask = torch.cat([false_column, key_padding_mask], dim=1)

        return key_padding_mask

    def prepare_channels(self, x):
        """Fuse per-channel features before patch_to_embedding.

        Called from forward after optional subsampling.

        Args:
            x: Tensor (batch, modes, channels, timesteps).

        Returns:
            Tensor (batch, channels, feature_dim) fed into the linear projector.
        """
        start_idx = 0 if self.include_distance else 1
        if self.conv_embedding:
            if self.mean:
                if self.std:
                    conv_input = x[:, :, :, 1:]
                else:
                    conv_input = x[:, :1, :, 1:]
            else:
                conv_input = x[:, 1:, :, 1:]

            batch_size, modes, chans, timesteps = conv_input.shape
            conv_input = conv_input.swapaxes(1, 2).reshape(-1, modes, timesteps)

            late_output = self.msresnet(conv_input)
            late_output = late_output.reshape(batch_size, chans, -1)

            if self.mlp_embedding:
                if self.mean:
                    if self.std:
                        all_output = torch.cat([x[:, 0, :, start_idx:155], x[:, 1, :, start_idx:155], late_output], dim=-1)
                    else:
                        all_output = torch.cat([x[:, 0, :, start_idx:155], late_output], dim=-1)
                else:
                    all_output = torch.cat([x[:, 1, :, start_idx:155], late_output], dim=-1)
            else:
                all_output = late_output
        elif self.mlp_embedding:
            if self.mean:
                if self.std:
                    all_output = torch.cat([x[:, 0, :, start_idx:], x[:, 1, :, start_idx:]], dim=-1)
                else:
                    all_output = x[:, 0, :, start_idx:]
            else:
                all_output = x[:, 1, :, start_idx:]
        
        return all_output


class SPESTransformer(BaseModel):
    """CNN-transformer for SPES / CCEP seizure-onset-zone localisation.

    Norris et al. (ML4H 2024) style model: multi-scale 1D convolutions plus a
    transformer over per-channel tokens, exposed through BaseModel / Trainer.

    Expected batch keys (e.g. from SeizureOnsetZoneLocalisation):

        * spes_responses: (batch, max_channels, 2, timesteps + 1) before the
          internal transpose to (batch, 2, max_channels, timesteps + 1).
        * soz_label (or label_key): binary labels.

    Args:
        dataset: Task SampleDataset for BaseModel.
        feature_keys: Input keys. Default: ["spes_responses"].
        label_key: Label key. Default: "soz_label".
        mode: Optional mode override (else inferred from schema).
        mean: Forwarded to SPESResponseEncoder. Default: True.
        std: Forwarded to SPESResponseEncoder. Default: True.
        conv_embedding: Forwarded to SPESResponseEncoder. Default: True.
        mlp_embedding: Forwarded to SPESResponseEncoder. Default: True.
        dropout_rate: Dropout on encoder and head. Default: 0.5.
        num_layers: Transformer depth. Default: 2.
        embedding_dim: Encoder width. Default: 64.
        random_channels: Channel subsample count, or None. Default: None.
        noise_std: Encoder training noise. Default: 0.0.
        include_distance: Keep distance at time index 0. Default: True.
        **kwargs: Unused; accepted for API compatibility.

    Examples:
        >>> from pyhealth.datasets.respectccep import RESPectCCEPDataset
        >>> from pyhealth.tasks.ccep_detect_soz import SeizureOnsetZoneLocalisation
        >>> from pyhealth.models import SPESTransformer
        >>> base = RESPectCCEPDataset(root="/path/to/respect_ccep")
        >>> sample_dataset = base.set_task(
        ...     SeizureOnsetZoneLocalisation(spes_mode="convergent")
        ... )
        >>> model = SPESTransformer(
        ...     dataset=sample_dataset,
        ...     embedding_dim=64,
        ...     num_layers=2,
        ...     random_channels=16,
        ... )
    """

    def __init__(
        self,
        dataset,
        feature_keys=None,
        label_key=None,
        mode=None,
        mean=True,
        std=True,
        conv_embedding=True,
        mlp_embedding=True,
        dropout_rate=0.5,
        num_layers=2,
        embedding_dim=64,
        random_channels=None,
        noise_std=0.0,
        include_distance=True,
        **kwargs
    ):
        """Build SPESResponseEncoder and the classifier MLP head.

        Args:
            dataset: Task dataset for schema and loss.
            feature_keys: Feature keys; default ["spes_responses"].
            label_key: Label key; default "soz_label".
            mode: Optional mode override.
            mean: Encoder mean flag. Default: True.
            std: Encoder std flag. Default: True.
            conv_embedding: Encoder conv path. Default: True.
            mlp_embedding: Encoder MLP path. Default: True.
            dropout_rate: Dropout. Default: 0.5.
            num_layers: Encoder depth. Default: 2.
            embedding_dim: Hidden size. Default: 64.
            random_channels: Channel subsample. Default: None.
            noise_std: Encoder noise. Default: 0.0.
            include_distance: Keep distance. Default: True.
            **kwargs: Reserved.
        """
        super(SPESTransformer, self).__init__(
            dataset=dataset,
        )
        self.feature_keys = feature_keys or ["spes_responses"]
        self.label_key = label_key or "soz_label"
        if mode is not None:
            self.mode = mode
        
        num_classes = 1

        self.encoder = SPESResponseEncoder(
            mean=mean,
            std=std,
            conv_embedding=conv_embedding,
            mlp_embedding=mlp_embedding,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            random_channels=random_channels,
            noise_std=noise_std,
            include_distance=include_distance
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_classes)
        )

        nn.init.xavier_uniform_(self.fc[1].weight)
        if self.fc[1].bias is not None:
            nn.init.zeros_(self.fc[1].bias)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits and optional loss.

        Args:
            **kwargs: Requires spes_responses; pass labels under label_key for
                loss (default key soz_label).

        Returns:
            Dict[str, torch.Tensor]: logit, y_prob, y_true, loss per
            BaseModel conventions.
        """
        # [batch_size, max_C, 2, T+1] -> [batch_size, 2, max_C, T+1]
        input_x = kwargs["spes_responses"].to(self.device)
        x = input_x.transpose(1, 2)

        features = self.encoder(x)
        logit = self.fc(features)

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
