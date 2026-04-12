import torch
import torch.nn as nn
from typing import Dict, List
from pyhealth.models import BaseModel


class ResBlock(nn.Module):
    """Residual Block used in Signal Encoders."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
        )
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.pool = nn.MaxPool1d(2)
        self.gelu = nn.GELU()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv(x)
        x = self.gelu(x + res)
        return self.pool(x)


class Wav2Sleep(BaseModel):
    """Wav2Sleep: A Unified Multi-Modal Approach to Sleep Stage Classification.

    Paper: Carter, J. F.; and Tarassenko, L. 2024. wav2sleep: A Unified
    Multi-Modal Approach to Sleep Stage Classification from Physiological Signals.

    The model consists of modality-specific CNN encoders, a transformer-based
    epoch mixer with a [CLS] token, and a dilated CNN sequence mixer.
    """

    def __init__(
            self,
            dataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int = 128,
            nhead: int = 8,
            num_layers: int = 2,
            mask_prob: Dict[str, float] = None,
            **kwargs,
    ):
        super(Wav2Sleep, self).__init__(
            dataset=dataset,
            **kwargs
        )

        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode
        self.embedding_dim = embedding_dim

        if dataset is not None and hasattr(dataset, "label_schema"):
            self.total_num_classes = 5
        else:
            self.total_num_classes = 5

        # [span_2](start_span)Default masking probabilities from paper[span_2]
        # (end_span)
        self.mask_probs = mask_prob or {
            "ecg": 0.5, "ppg": 0.1, "abd": 0.7, "thx": 0.7
        }

        # 1. [span_3](start_span)[span_4](start_span)Signal Encoders: Modality
        # specific CNNs[span_3](end_span)[span_4](end_span)
        self.feature_encoders = nn.ModuleDict()
        for key in feature_keys:
            # [span_5](start_span)[span_6](start_span)Paper uses 6-8 layers depending
            # on sampling rate k[span_5](end_span)[span_6](end_span)
            layers = [ResBlock(1, 16)]
            layers += [ResBlock(16 * (2 ** i), 16 * (2 ** (i + 1))) for i in range(3)]
            layers.append(nn.AdaptiveAvgPool1d(1))
            self.feature_encoders[key] = nn.Sequential(*layers)

        # 2. [span_7](start_span)Epoch Mixer: Transformer with [CLS] token[span_7]
        # (end_span)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=512,
            batch_first=True, activation="gelu"
        )
        self.epoch_mixer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. [span_8](start_span)[span_9](start_span)Sequence Mixer: Dilated
        # Convolutions[span_8](end_span)[span_9](end_span)
        # [span_10](start_span)Two blocks with dilations (1, 2, 4, 8, 16, 32)[span_10]
        # (end_span)
        self.sequence_mixer = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, 7, padding=6, dilation=2),
            nn.GELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 7, padding=12, dilation=4),
            nn.GELU(),
        )
        self.fc = nn.Linear(embedding_dim, self.total_num_classes)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with stochastic masking and multi-modal fusion."""
        batch_size = kwargs[self.feature_keys[0]].shape[0]
        seq_len = kwargs[self.feature_keys[0]].shape[1]  # T=1200

        # List to store features [batch*seq_len, 1, embedding_dim]
        all_modality_features = []

        for key in self.feature_keys:
            x = kwargs[key].view(-1, 1, kwargs[key].shape[-1])  # [B*T, 1, L]
            feat = self.feature_encoders[key](x).view(batch_size, seq_len, -1)

            # [span_11](start_span)Stochastic Masking during training[span_11]
            # (end_span)
            if self.training:
                p = self.mask_probs.get(key.lower(), 0.5)
                mask = (torch.rand(batch_size, 1, 1, device=feat.device) > p).float()
                feat = feat * mask

            all_modality_features.append(feat.unsqueeze(2))  # [B, T, 1, D]

        # Combine modalities for Epoch Mixer
        # x: [B*T, num_modalities, D]
        x = torch.cat(all_modality_features, dim=2).view(-1, len(self.feature_keys)
                                                         , 128)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B*T, M+1, D]

        # Epoch Fusion
        x = self.epoch_mixer(x)
        z_t = x[:, 0, :].view(batch_size, seq_len, -1)  # Extract CLS [B, T, D]

        # [span_12](start_span)Sequence Mixing: Capture temporal dependencies[span_12]
        # (end_span)
        z_t = z_t.transpose(1, 2)  # [B, D, T]
        z_seq = self.sequence_mixer(z_t).transpose(1, 2)  # [B, T, D]

        logits = self.fc(z_seq)

        # PyHealth expectation: return loss and probabilities
        return {
            "y_prob": torch.softmax(logits, dim=-1),
            "y_true": kwargs[self.label_key],
            "loss": nn.CrossEntropyLoss()(logits.view(-1, self.total_num_classes),
                                          kwargs[self.label_key].view(-1))
        }
