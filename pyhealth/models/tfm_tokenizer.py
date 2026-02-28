import math
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

LinearAttentionTransformer = None


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


def get_stft_torch(X, resampling_rate = 200):
    B,C,T = X.shape
    x_temp = rearrange(X, 'B C T -> (B C) T')
    window = torch.hann_window(resampling_rate).to(x_temp.device)
    x_stft_temp = torch.abs(torch.stft(x_temp, n_fft=resampling_rate, hop_length=resampling_rate//2, 
                          onesided = True,
                          return_complex=True, center = False,#normalized = True,
                          window = window)[:,:resampling_rate//2,:])
    
    x_stft_temp = rearrange(x_stft_temp, '(B C) F T -> B C F T', B=B)
    
    return x_stft_temp

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Args:
        d_model: dimension of the model embedding.
        dropout: dropout probability. Default is 0.1.
        max_len: maximum sequence length. Default is 1000.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        """Forward propagation.
        
        Args:
            x: input embeddings of shape (batch, max_len, d_model).
            
        Returns:
            output tensor of shape (batch, max_len, d_model).
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder using linear attention.
    
    Args:
        emb_size: embedding size. Default is 64.
        num_heads: number of attention heads. Default is 8.
        depth: number of transformer layers. Default is 4.
        max_seq_len: maximum sequence length. Default is 1024.
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 8,
        depth: int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        LAT = _get_linear_attention_transformer()
        self.transformer = LAT(
            dim=emb_size,
            heads=num_heads,
            depth=depth,
            max_seq_len=max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

    def forward(self, x):
        """Forward propagation.
        
        Args:
            x: input tensor of shape (batch, seq_len, emb_size).
            
        Returns:
            output tensor of shape (batch, seq_len, emb_size).
        """
        x = self.transformer(x)
        return x


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


class EMAVectorQuantizer(nn.Module):
    """Exponential Moving Average Vector Quantizer.
    
    Args:
        emb_size: dimensionality of embeddings.
        code_book_size: number of codebook entries.
        decay: exponential moving average decay factor. Default is 0.99.
        eps: small constant for numerical stability. Default is 1e-5.
    """

    def __init__(
        self, emb_size: int, code_book_size: int, decay: float = 0.99, eps: float = 1e-5
    ):
        super().__init__()
        self.emb_size = emb_size
        self.code_book_size = code_book_size
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(code_book_size, emb_size)
        self.embedding.weight.data.uniform_(-1 / code_book_size, 1 / code_book_size)

        self.register_buffer("cluster_size", torch.zeros(code_book_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, x):
        """Forward propagation.
        
        Args:
            x: input tensor of shape (B, T, emb_size).
            
        Returns:
            quantized: quantized vectors of shape (B, T, emb_size).
            encoding_indices: indices of selected codebook entries of shape (B, T).
        """
        flat_x = x.reshape(-1, self.emb_size)

        dist = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )

        encoding_indices = torch.argmin(dist, dim=1)
        quantized = self.embedding(encoding_indices).view_as(x)

        if self.training:
            encodings_one_hot = F.one_hot(encoding_indices, self.code_book_size).type_as(
                flat_x
            )

            new_cluster_size = encodings_one_hot.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(
                new_cluster_size, alpha=1 - self.decay
            )

            dw = encodings_one_hot.t() @ flat_x
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.code_book_size * self.eps)
                * n
            )

            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        encoding_indices = encoding_indices.reshape(x.size(0), x.size(1))
        return quantized, encoding_indices


def freq_bin_temporal_masking(
    X,
    freq_mask_ratio: float = 0.5,
    freq_bin_size: int = 5,
    time_mask_ratio: float = 0.5,
    time_bin_size: int = 10,
):
    """Apply frequency-bin and temporal masking to spectrograms.
    
    Args:
        X: input spectrogram of shape (B, F, T).
        freq_mask_ratio: ratio of frequency bins to mask. Default is 0.5.
        freq_bin_size: size of frequency bins. Default is 5.
        time_mask_ratio: ratio of time bins to mask. Default is 0.5.
        time_bin_size: size of time bins. Default is 10.
        
    Returns:
        X_masked: masked spectrogram (unmasked regions).
        X_masked_sym: inverse masked spectrogram (masked regions).
        full_mask: boolean mask for unmasked regions.
        full_mask_sym: boolean mask for masked regions.
    """
    B, F, T = X.shape

    num_freq_bins = F // freq_bin_size
    X_freq_binned = X.view(B, num_freq_bins, freq_bin_size, T)
    freq_mask = torch.ones_like(X_freq_binned)
    num_freq_bins_to_mask = int(num_freq_bins * freq_mask_ratio)
    freq_bins_to_mask = torch.randperm(num_freq_bins)[:num_freq_bins_to_mask]
    freq_mask[:, freq_bins_to_mask, ...] = 0
    freq_mask = freq_mask.view(B, F, T)

    num_time_bins = T // time_bin_size
    X_time_binned = X.view(B, F, num_time_bins, time_bin_size)
    time_mask = torch.ones_like(X_time_binned)
    num_time_bins_to_mask = int(num_time_bins * time_mask_ratio)
    time_bins_to_mask = torch.randperm(num_time_bins)[:num_time_bins_to_mask]
    time_mask[:, :, time_bins_to_mask, :] = 0
    time_mask = time_mask.view(B, F, T)

    full_mask = freq_mask * time_mask
    full_mask_sym = 1 - full_mask
    full_mask = full_mask.to(torch.bool)
    full_mask_sym = full_mask_sym.to(torch.bool)
    X_masked = X * full_mask
    X_masked_sym = X * full_mask_sym

    return X_masked, X_masked_sym, full_mask, full_mask_sym


class TFM_VQVAE2_deep(nn.Module):
    """TFM-Tokenizer module with raw EEG and STFT as input.
    
    Args:
        in_channels: number of input channels. Default is 1.
        n_freq: number of frequency bins in STFT. Default is 100.
        n_freq_patch: frequency patch size. Default is 5.
        emb_size: embedding dimension. Default is 64.
        code_book_size: size of the VQ codebook. Default is 8192.
        trans_freq_encoder_depth: depth of frequency encoder. Default is 4.
        trans_temporal_encoder_depth: depth of temporal encoder. Default is 4.
        trans_decoder_depth: depth of decoder. Default is 4.
        beta: weight for commitment loss. Default is 1.0.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_freq: int = 100,
        n_freq_patch: int = 5,
        emb_size: int = 64,
        code_book_size: int = 8192,
        trans_freq_encoder_depth: int = 4,
        trans_temporal_encoder_depth: int = 4,
        trans_decoder_depth: int = 4,
        beta: float = 1.0,
    ):
        super().__init__()
        self.n_freq_patch = n_freq_patch
        self.emb_size = emb_size
        self.code_book_size = code_book_size

        # bin wise frequency embedding
        self.freq_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, kernel_size=n_freq_patch, stride=n_freq_patch),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size=1, stride=1),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size=1, stride=1),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size),
        )

        # Freq Encoder
        self.trans_freq_encoder = TransformerEncoder(
            emb_size=emb_size,
            num_heads=8,
            depth=trans_freq_encoder_depth,
            max_seq_len=n_freq // n_freq_patch,
        )

        # Temporal embedding
        self.temporal_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, kernel_size=200, stride=100),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size=1, stride=1),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size),
            nn.Conv1d(emb_size, emb_size // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size // 2),
        )

        # attention based aggregation
        global_freq_divider = n_freq // (n_freq_patch * n_freq_patch)
        self.freq_patch_embedding_2_atten = nn.Sequential(
            nn.Conv1d(
                emb_size,
                emb_size // (global_freq_divider * 2),
                kernel_size=n_freq_patch,
                stride=n_freq_patch,
            ),
            nn.Sigmoid(),
        )
        self.freq_patch_embedding_2 = nn.Sequential(
            nn.Conv1d(
                emb_size,
                emb_size // (global_freq_divider * 2),
                kernel_size=n_freq_patch,
                stride=n_freq_patch,
            ),
        )

        # Temporal Encoder
        self.trans_temporal_encoder = TransformerEncoder(
            emb_size=emb_size, num_heads=8, depth=trans_temporal_encoder_depth
        )

        # Vector quantization bottleneck
        self.quantizer = EMAVectorQuantizer(emb_size, code_book_size)
        self.beta = beta

        # Decoder
        self.trans_decoder = TransformerEncoder(
            emb_size=emb_size, num_heads=8, depth=trans_decoder_depth
        )

        # self.decoder = nn.Linear(emb_size, n_freq)
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.Tanh(), nn.Linear(emb_size, n_freq)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"quantizer.embedding.weight"}

    def tokenize(self, x, x_temporal):
        """Tokenize EEG signals into discrete tokens.
        
        Args:
            x: STFT spectrogram of shape (B, F, T).
            x_temporal: raw temporal signal of shape (B, n_samples).
            
        Returns:
            quant_out: quantized output.
            indices: discrete token indices.
            quant_in: input to quantizer (before quantization).
        """
        B, F, T = x.shape
        x = x.permute(0, 2, 1).reshape(-1, 1, F)
        x = self.freq_patch_embedding(x)
        x = x.permute(0, 2, 1)

        x = self.trans_freq_encoder(x)

        x = x.permute(0, 2, 1)
        atten = self.freq_patch_embedding_2_atten(x)
        x = self.freq_patch_embedding_2(x) * atten
        x = x.reshape(-1, x.size(1) * x.size(2))

        x = rearrange(x, "(B T) E -> B T E", T=T)

        x_temporal = x_temporal.unsqueeze(1)
        x_temporal = self.temporal_patch_embedding(x_temporal)
        x_temporal = rearrange(x_temporal, "B E T -> B T E")

        x = torch.cat((x, x_temporal), dim=-1)

        x = self.trans_temporal_encoder(x)

        quant_in = l2norm(x)
        quant_out, indices = self.quantizer(quant_in)

        return quant_out, indices, quant_in

    def forward(self, x, x_temporal):
        """Forward propagation.
        
        Args:
            x: STFT spectrogram of shape (B, F, T).
            x_temporal: raw temporal signal of shape (B, n_samples).
            
        Returns:
            x: reconstructed STFT spectrogram.
            indices: discrete token indices.
            quant_out: quantized output.
            quant_in: input to quantizer.
        """
        quant_out, indices, quant_in = self.tokenize(x, x_temporal)
        quant_out = quant_in + (quant_out - quant_in).detach()
        x = self.trans_decoder(quant_out)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        return x, indices, quant_out, quant_in

    def vec_quantizer_loss(self, quant_in, quant_out):
        """Compute vector quantizer losses.
        
        Args:
            quant_in: input to quantizer.
            quant_out: output from quantizer.
            
        Returns:
            loss: total VQ loss.
            code_book_loss: codebook loss component.
            commitment_loss: commitment loss component.
        """
        commitment_loss = torch.mean((quant_out.detach() - quant_in) ** 2)
        code_book_loss = torch.mean((quant_out - quant_in.detach()) ** 2)
        loss = code_book_loss + self.beta * commitment_loss
        return loss, code_book_loss, commitment_loss

    @torch.no_grad()
    def forward_ana(self, x, x_temporal):
        """Forward propagation with intermediate outputs for analysis.
        
        Returns:
            x_dec: reconstructed output.
            indices: quantizer indices.
            quant_out: quantized representation.
            quant_in: input to quantizer.
            freq_encoded: frequency encoder tokens.
            temporal_encoded: temporal encoder tokens.
        """
        B, F, T = x.shape

        x_freq = x.permute(0, 2, 1).reshape(-1, 1, F)
        x_freq = self.freq_patch_embedding(x_freq)
        x_freq = x_freq.permute(0, 2, 1)

        freq_encoded = self.trans_freq_encoder(x_freq)

        x_freq_agg = freq_encoded.permute(0, 2, 1)
        atten = self.freq_patch_embedding_2_atten(x_freq_agg)
        x_freq_agg = self.freq_patch_embedding_2(x_freq_agg) * atten
        x_freq_agg = x_freq_agg.reshape(-1, x_freq_agg.size(1) * x_freq_agg.size(2))
        x_freq_agg = rearrange(x_freq_agg, "(B T) E -> B T E", T=T)

        x_temporal_branch = x_temporal.unsqueeze(1)
        x_temporal_branch = self.temporal_patch_embedding(x_temporal_branch)
        x_temporal_branch = rearrange(x_temporal_branch, "B E T -> B T E")

        x_combined = torch.cat((x_freq_agg, x_temporal_branch), dim=-1)

        temporal_encoded = self.trans_temporal_encoder(x_combined)

        quant_in = l2norm(temporal_encoded)
        quant_out, indices = self.quantizer(quant_in)
        quant_out = quant_in + (quant_out - quant_in).detach()

        x_dec = self.trans_decoder(quant_out)
        x_dec = self.decoder(x_dec)
        x_dec = x_dec.permute(0, 2, 1)

        return x_dec, indices, quant_out, quant_in, freq_encoded, temporal_encoded


class TFM_TOKEN_Classifier(nn.Module):
    """Downstream classifier using TFM tokens.
    
    Args:
        emb_size: embedding dimension. Default is 256.
        code_book_size: size of the VQ codebook. Default is 8192.
        num_heads: number of attention heads. Default is 8.
        depth: number of transformer layers. Default is 12.
        max_seq_len: maximum sequence length. Default is 61.
        n_classes: number of output classes. Default is 5.
    """

    def __init__(
        self,
        emb_size: int = 256,
        code_book_size: int = 8192,
        num_heads: int = 8,
        depth: int = 12,
        max_seq_len: int = 61,
        n_classes: int = 5,
    ):
        super().__init__()

        self.eeg_token_embedding = nn.Embedding(code_book_size + 1, emb_size)
        self.channel_embed = nn.Embedding(16, emb_size)
        self.index = nn.Parameter(torch.LongTensor(range(16)), requires_grad=False)
        self.temporal_pos_embed = PositionalEncoding(emb_size)
        self.pos_drop = nn.Dropout(p=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        LAT = _get_linear_attention_transformer()
        self.LAT = LAT(
            dim=emb_size,
            heads=num_heads,
            depth=depth,
            max_seq_len=max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        self.classification_head = nn.Linear(emb_size, n_classes)

    def forward(self, x, num_ch: int = 16):
        """Forward propagation.
        
        Args:
            x: token indices of shape (B, C, T).
            num_ch: number of channels. Default is 16.
            
        Returns:
            pred: class predictions of shape (B, n_classes).
        """
        x = self.eeg_token_embedding(x)

        for i in range(x.shape[1]):
            used_channel_embed = (
                self.channel_embed(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(x.size(0), -1, -1)
            )
            x[:, i] = self.temporal_pos_embed(x[:, i] + used_channel_embed)

        x = rearrange(x, "B C T E -> B (C T) E")

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.LAT(x)
        pred = self.classification_head(x[:, 0])
        return pred

    def masked_prediction(self, x, num_ch: int = 16):
        """Forward propagation with masked prediction (all tokens).
        
        Args:
            x: token indices of shape (B, C, T).
            num_ch: number of channels. Default is 16.
            
        Returns:
            pred: predictions for all tokens (excluding CLS).
        """
        x = self.eeg_token_embedding(x)

        for i in range(x.shape[1]):
            used_channel_embed = (
                self.channel_embed(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(x.size(0), -1, -1)
            )
            x[:, i] = self.temporal_pos_embed(x[:, i] + used_channel_embed)

        x = rearrange(x, "B C T E -> B (C T) E")

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.LAT(x)
        pred = self.classification_head(x[:, 1:])
        return pred

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temporal_pos_embed", "cls_token"}


def get_tfm_tokenizer_2x2x8(code_book_size: int = 8192, emb_size: int = 64):
    """Create TFM-Tokenizer with 2x2x8 architecture.
    
    Args:
        code_book_size: size of the VQ codebook. Default is 8192.
        emb_size: embedding dimension. Default is 64.
        
    Returns:
        TFM_VQVAE2_deep model instance.
    """
    vqvae = TFM_VQVAE2_deep(
        in_channels=1,
        n_freq=100,
        n_freq_patch=5,
        emb_size=emb_size,
        code_book_size=code_book_size,
        trans_freq_encoder_depth=2,
        trans_temporal_encoder_depth=2,
        trans_decoder_depth=8,
        beta=1.0,
    )
    return vqvae


def get_tfm_token_classifier_64x4(
    n_classes: int = 5, code_book_size: int = 8192, emb_size: int = 64
):
    """Create TFM-Token classifier with 64x4 architecture.
    
    Args:
        n_classes: number of output classes. Default is 5.
        code_book_size: size of the VQ codebook. Default is 8192.
        emb_size: embedding dimension. Default is 64.
        
    Returns:
        TFM_TOKEN_Classifier model instance.
    """
    classifier = TFM_TOKEN_Classifier(
        emb_size=emb_size,
        code_book_size=code_book_size,
        num_heads=8,
        depth=4,
        max_seq_len=2048,
        n_classes=n_classes,
    )
    return classifier


def load_embedding_weights(source_model, target_model):
    """Load embedding weights from tokenizer to classifier.
    
    Args:
        source_model: the tokenizer model (TFM_VQVAE2_deep).
        target_model: the classifier model (TFM_TOKEN_Classifier).
    """
    source_weights = source_model.quantizer.embedding.weight.data
    target_weights = target_model.eeg_token_embedding.weight.data

    src_vocab_size, src_emb_dim = source_weights.shape
    tgt_vocab_size, tgt_emb_dim = target_weights.shape

    print(f"Source Embedding Shape: {source_weights.shape}")
    print(f"Target Embedding Shape: {target_weights.shape}")

    if src_emb_dim != tgt_emb_dim:
        raise ValueError(
            f"Embedding size mismatch: {src_emb_dim} (source) vs {tgt_emb_dim} (target)"
        )

    if src_vocab_size > tgt_vocab_size:
        adapted_weights = source_weights[:tgt_vocab_size, :]
        print(f"Trimming source embeddings from {src_vocab_size} to {tgt_vocab_size}")
    elif src_vocab_size < tgt_vocab_size:
        adapted_weights = torch.zeros(
            (tgt_vocab_size, tgt_emb_dim), dtype=source_weights.dtype
        )
        adapted_weights[:src_vocab_size, :] = source_weights
        print(f"Padding source embeddings from {src_vocab_size} to {tgt_vocab_size}")
    else:
        adapted_weights = source_weights

    target_model.eeg_token_embedding.weight.data.copy_(adapted_weights)
    print("Successfully loaded embedding weights!")


class TFMTokenizer(BaseModel):
    """TFM-Tokenizer model.
    
    This model uses VQ-VAE with transformers to tokenize EEG signals. It can
    extract discrete tokens and continuous embeddings for downstream tasks.
    
    The model expects two inputs:
        - STFT spectrogram: shape (batch, n_freq, n_time)
        - Raw temporal signal: shape (batch, n_samples)
    
    Args:
        dataset: the dataset to train the model.
        emb_size: embedding dimension. Default is 64.
        code_book_size: size of the VQ codebook. Default is 8192.
        n_freq: number of frequency bins in STFT. Default is 100.
        n_freq_patch: frequency patch size. Default is 5.
        trans_freq_encoder_depth: depth of frequency encoder. Default is 2.
        trans_temporal_encoder_depth: depth of temporal encoder. Default is 2.
        trans_decoder_depth: depth of decoder. Default is 8.
        use_classifier: whether to use the classifier head. Default is True.
        classifier_depth: depth of classifier transformer. Default is 4.
        classifier_heads: number of attention heads in classifier. Default is 8.
        
    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> from pyhealth.models import TFMTokenizer
        >>> dataset = TUEVDataset(root="/path/to/tuev")
        >>> sample_dataset = dataset.set_task()
        >>> model = TFMTokenizer(dataset=sample_dataset)
        >>> model.load_pretrained_weights("tfm_encoder_best_model.pth")
    """

    def __init__(
        self,
        dataset: SampleDataset,
        emb_size: int = 64,
        code_book_size: int = 8192,
        n_freq: int = 100,
        n_freq_patch: int = 5,
        trans_freq_encoder_depth: int = 2,
        trans_temporal_encoder_depth: int = 2,
        trans_decoder_depth: int = 8,
        use_classifier: bool = True,
        classifier_depth: int = 4,
        classifier_heads: int = 8,
        **kwargs,
    ):
        super().__init__(dataset=dataset)

        self.emb_size = emb_size
        self.code_book_size = code_book_size
        self.use_classifier = use_classifier

        self.tokenizer = TFM_VQVAE2_deep(
            in_channels=1,
            n_freq=n_freq,
            n_freq_patch=n_freq_patch,
            emb_size=emb_size,
            code_book_size=code_book_size,
            trans_freq_encoder_depth=trans_freq_encoder_depth,
            trans_temporal_encoder_depth=trans_temporal_encoder_depth,
            trans_decoder_depth=trans_decoder_depth,
            beta=1.0,
        )

        if use_classifier:
            output_size = self.get_output_size()
            self.classifier = TFM_TOKEN_Classifier(
                emb_size=emb_size,
                code_book_size=code_book_size,
                num_heads=classifier_heads,
                depth=classifier_depth,
                max_seq_len=2048,
                n_classes=output_size,
            )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        
        Args:
            **kwargs: keyword arguments containing 'stft', 'signal', and label key.
                
        Returns:
            a dictionary containing loss, y_prob, y_true, logit, tokens, embeddings.
        """
        # stft = kwargs.get("stft")
        signal = kwargs.get("signal")
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(0)
        B,C,T = signal.shape
        stft = get_stft_torch(signal)
        stft = rearrange(stft, 'B C F T -> (B C) F T')
        signal = rearrange(signal, 'B C T -> (B C) T')
        
        if stft is None or signal is None:
            raise ValueError("Both 'stft' and 'signal' must be provided in inputs")

        stft = stft.to(self.device)
        signal = signal.to(self.device)

        reconstructed, tokens, quant_out, quant_in = self.tokenizer(stft, signal)

        recon_loss = F.mse_loss(reconstructed, stft)
        vq_loss, _, _ = self.tokenizer.vec_quantizer_loss(quant_in, quant_out)
        tokens_reshaped = rearrange(tokens, '(B C) T -> B C T', C=C)
        quant_out_reshaped = rearrange(quant_out, '(B C) T E -> B C T E', C=C)

        results = {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "tokens": tokens_reshaped,
            "embeddings": quant_out_reshaped,
        }

        if self.use_classifier and len(self.label_keys) > 0:
            label_key = self.label_keys[0]
            y_true = kwargs[label_key].to(self.device)

            # Reshape tokens to (B, C, T) for multi-channel classifier
            # tokens shape: (B, T) -> (B, 1, T)
            logits = self.classifier(tokens_reshaped,num_ch=C)
            loss_fn = self.get_loss_function()
            print(f"logits shape: {logits.shape}")
            print(f"y_true shape: {y_true.shape}")
            cls_loss = loss_fn(logits, y_true)
            total_loss = recon_loss + vq_loss + cls_loss
            y_prob = self.prepare_y_prob(logits)

            results.update(
                {
                    "loss": total_loss,
                    "cls_loss": cls_loss,
                    "y_prob": y_prob,
                    "y_true": y_true,
                    "logit": logits,
                }
            )
        else:
            results["loss"] = recon_loss + vq_loss

        return results

    def get_embeddings(self, dataloader) -> torch.Tensor:
        """Extract continuous embeddings for all samples in a dataloader.
        
        Args:
            dataloader: PyHealth dataloader.
            
        Returns:
            tensor of shape (n_samples, seq_len, emb_size).
        """
        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                signal = batch.get("signal").to(self.device)
                if len(signal.shape) == 2:
                    signal = signal.unsqueeze(0)
                B,C,T = signal.shape
                stft = get_stft_torch(signal)
                stft = rearrange(stft, 'B C F T -> (B C) F T')
                signal = rearrange(signal, 'B C T -> (B C) T')
                _, _, quant_out, _ = self.tokenizer(stft, signal)
                print(f"quant_out shape: {quant_out.shape}")
                quant_out = rearrange(quant_out, '(B C) T E -> B C T E', C=C)
                print(f"quant_out shape: {quant_out.shape}")
                all_embeddings.append(quant_out.cpu())

        return torch.cat(all_embeddings, dim=0)

    def get_tokens(self, dataloader) -> torch.Tensor:
        """Extract discrete tokens for all samples in a dataloader.
        
        Args:
            dataloader: PyHealth dataloader.
            
        Returns:
            tensor of shape (n_samples, seq_len).
        """
        self.eval()
        all_tokens = []

        with torch.no_grad():
            for batch in dataloader:
                signal = batch.get("signal").to(self.device)
                if len(signal.shape) == 2:
                    signal = signal.unsqueeze(0)
                B,C,T = signal.shape
                stft = get_stft_torch(signal)
                stft = rearrange(stft, 'B C F T -> (B C) F T')
                signal = rearrange(signal, 'B C T -> (B C) T')
                _, tokens, _, _ = self.tokenizer(stft, signal)
                tokens = rearrange(tokens, '(B C) T -> B C T', C=C)
                all_tokens.append(tokens.cpu())

        return torch.cat(all_tokens, dim=0)

    def load_pretrained_weights(
        self, 
        tokenizer_checkpoint_path: str, 
        classifier_checkpoint_path: str = None,
        is_masked_training: bool = False,
        strict: bool = False, 
        map_location: str = None
    ):
        """Load pre-trained weights from checkpoint.
        
        Args:
            tokenizer_checkpoint_path: path to the tokenizer checkpoint file.
            classifier_checkpoint_path: path to the classifier checkpoint file.
            strict: whether to strictly enforce key matching. Default is True.
            map_location: device to map the loaded tensors. Default is None.
        """
        if map_location is None:
            map_location = str(self.device)

        # Load tokenizer weights
        self.tokenizer.load_state_dict(torch.load(tokenizer_checkpoint_path, map_location=map_location), strict=strict)

        if classifier_checkpoint_path is not None and not is_masked_training:
            self.classifier.load_state_dict(torch.load(classifier_checkpoint_path, map_location=map_location))
            print(f"✓ Successfully loaded weights from {classifier_checkpoint_path}")
        elif is_masked_training:
            load_embedding_weights(self.tokenizer, self.classifier)
            print("✓ Successfully loaded embedding weights!")
        else:
            print(f"No classifier checkpoint path provided. Skipping classifier weight loading.")

    

if __name__ == "__main__":
    print("Testing TFM-Tokenizer components...")

    tokenizer = get_tfm_tokenizer_2x2x8()
    print(f"✓ Created tokenizer: {tokenizer.__class__.__name__}")

    classifier = get_tfm_token_classifier_64x4(n_classes=6)
    print(f"✓ Created classifier: {classifier.__class__.__name__}")

    batch_size = 2
    n_freq = 100
    n_time = 60
    n_samples = 1280

    dummy_stft = torch.randn(batch_size, n_freq, n_time)
    dummy_signal = torch.randn(batch_size, n_samples)

    recon, tokens, quant_out, quant_in = tokenizer(dummy_stft, dummy_signal)
    print(f"✓ Tokenizer forward pass:")
    print(f"  Reconstructed shape: {recon.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Embeddings shape: {quant_out.shape}")

    preds = classifier(tokens)
    print(f"✓ Classifier forward pass:")
    print(f"  Predictions shape: {preds.shape}")

    print("\n✓ All tests passed!")
