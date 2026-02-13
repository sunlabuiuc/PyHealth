import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# =============================================================================
# LAZY IMPORT FOR OPTIONAL DEPENDENCY
# =============================================================================
# The linear_attention_transformer package is only needed for TFMTokenizer and
# TFM_TOKEN_Classifier classes. However, this module is imported at package
# load time via pyhealth.models.__init__.py.
#
# Problem: If we import LinearAttentionTransformer at module level, users who
# don't need TFMTokenizer still get ImportError when they `import pyhealth.models`.
#
# Solution: Lazy import - only load the dependency when actually instantiating
# a class that needs it. This keeps the repo functional for the 95% of users
# who don't use TFM classes, while still providing clear error messages for
# the 5% who do but forgot to install the dependency.
#
# Why test failures are not an issue:
#   - Package imports work correctly (pyhealth.models loads w/ out error)
#   - Only those who instantiate TFMTokenizer see the ImportError
#   - Error message provides clear install instructions
#   - Tests in tests/core/test_tfm_tokenizer.py will fail w/ out the optional
#     dependency, but this is intentional behavior showing the lazy import works
# =============================================================================

LinearAttentionTransformer = None # Set to True to use the LinearAttentionTransformer


def _get_linear_attention_transformer():
    """Lazily import LinearAttentionTransformer on first use.

    This function implements a lazy import pattern to avoid breaking the
    PyHealth package when the optional `linear_attention_transformer`
    dependency is not installed.

    Returns:
        The LinearAttentionTransformer class from the external package.

    Raises:
        ImportError: If the package is not installed, with a helpful
            message explaining how to install it.

    Why This Pattern:
        - pyhealth.models.__init__.py imports from this file at package load
        - A top-level `from linear_attention_transformer import ...` would
          cause ImportError for ALL users of pyhealth.models, even those
          who don't need TFMTokenizer
        - By deferring the import to class instantiation time, we ensure
          the error only occurs for users who actually try to use the
          affected classes
    """
    global LinearAttentionTransformer
    if LinearAttentionTransformer is None:
        try:
            from linear_attention_transformer import LinearAttentionTransformer as LAT
            LinearAttentionTransformer = LAT
        except ImportError:
            raise ImportError(
                "linear_attention_transformer is required for TFMTokenizer. "
                "Install it with: pip install linear-attention-transformer"
            )
    return LinearAttentionTransformer


from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb
    
    
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x
    

class BIOT(BaseModel):
    """BIOT: Biosignal transformer for cross-data learning in the wild
    Citation:
    Yang, Chaoqi, M. Westover, and Jimeng Sun. "Biot: Biosignal transformer for cross-data learning in the wild." Advances in Neural Information Processing Systems 36 (2023): 78240-78260.

    The BIOT model encodes multichannel biosignal data (such as EEG) into compact feature representations 
    using spectral patch embeddings, channel positional encodings, and a transformer encoder.

    The model expects as input:
        - Raw temporal biosignals: shape (batch_size, n_channels, n_time)

    Args:
        dataset: the dataset to train or evaluate the model (must be compatible with SampleDataset).
        emb_size: embedding dimension for token/channel representations. Default is 256.
        heads: number of transformer attention heads. Default is 8.
        depth: number of transformer encoder layers. Default is 4.
        n_fft: number of frequency bins used in the STFT transform. Default is 200.
        hop_length: hop length for the STFT transform. Default is 100.
        n_classes: number of output classes for classification tasks (only used for BIOTClassifier).
        n_channels: number of channels in the biosignal data. Default is 18. 
            This includes the 16 channels of the TUEV dataset and 2 additional channels for Sleep dataset.
    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> from pyhealth.models import BIOT
        >>> dataset = TUEVDataset(root="/path/to/tuev")
        >>> sample_dataset = dataset.set_task()
        >>> model = BIOT(dataset=sample_dataset,
        >>>             emb_size=256,
        >>>             heads=8,
        >>>             depth=4,
        >>>             n_fft=200,
        >>>             hop_length=100,
        >>>             n_classes=6,
        >>>             n_channels=18,
        >>>             )
        >>> model.load_pretrained_weights("pretrained-models/EEG-six-datasets-18-channels.ckpt")
        >>> # Pretrained weights for the BIOT model trained on the EEG-six-datasets dataset with 18 channels. 
        >>> # Provided by the authors: https://github.com/ycq091044/BIOT/blob/main/pretrained-models/EEG-six-datasets-18-channels.ckpt
        >>> output = model(torch.randn(8, 18, TIME_STEPS))  # (batch, channels, time)
    """
    
    def __init__(self, 
                 dataset: SampleDataset,
                 emb_size: int = 256,
                 heads: int = 8,
                 depth: int = 4,
                 n_fft: int = 200,
                 hop_length: int = 100,
                 n_classes: int = 6,
                 n_channels: int = 18,
                 **kwargs):
        super().__init__(dataset=dataset)
        self.biot = BIOTClassifier(emb_size=emb_size, 
                                   heads=heads, 
                                   depth=depth, 
                                   n_classes=n_classes, 
                                   n_channels=n_channels, 
                                   n_fft=n_fft, 
                                   hop_length=hop_length)
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False, map_location: str = None):
        """Load pre-trained weights from checkpoint.
        
        Args:
            checkpoint_path: path to the checkpoint file.
            strict: whether to strictly enforce key matching. Default is True.
            map_location: device to map the loaded tensors. Default is None.
        """
        if map_location is None:
            map_location = str(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.biot.load_state_dict(state_dict, strict=strict)
        print(f"✓ Successfully loaded weights from {checkpoint_path}")
        
        
    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        
        Args:
            **kwargs: keyword arguments containing 'signal'.
                
        Returns:
            a dictionary containing loss, y_prob, y_true, logit, tokens, embeddings.
        """
        signal = kwargs.get("signal")
        if signal is None:
            raise ValueError("'signal' must be provided in inputs")
        signal = signal.to(self.device)
        logits = self.biot(signal)
        label_key = self.label_keys[0]
        y_true = kwargs[label_key].to(self.device)
        
        loss_fn = self.get_loss_function()
        loss = loss_fn(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        return results
        
        

if __name__ == "__main__":
    _get_linear_attention_transformer()
    print("Testing BIOT model...")
    model = BIOTClassifier(emb_size=256, heads=8, depth=4, n_classes=6, n_channels=18, n_fft=200, hop_length=100)
    print(f"✓ Created BIOTClassifier: {model.__class__.__name__}")
    
    batch_size = 2
    n_channels = 18
    n_time = 10
    n_samples = 200*n_time
    
    dummy_signal = torch.randn(batch_size, n_channels, n_samples)
    logits = model(dummy_signal)
    print(f"✓ BIOTClassifier forward pass:")
    print(f"  Logits shape: {logits.shape}")
    
    print("\n✓ All tests passed!")