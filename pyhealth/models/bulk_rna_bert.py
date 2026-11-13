"""BulkRNABert survival model for PyHealth.

Reproduces the downstream survival prediction pipeline of Gélard et al. (2025),
"BulkRNABert: Cancer prognosis from bulk RNA-seq based language models"
(ML4H 2025), using the authors' released frozen encoder and a newly initialized
MLP survival head trained with Cox partial likelihood.


Reference implementation:
    https://github.com/instadeepai/multiomics-open-research

Two operating modes:

- **Cached embeddings** (``use_cached_embeddings=True``): the input ``embedding``
  field is a precomputed 256-dimensional patient representation, typically the
  mean over genes of the frozen encoder's last transformer layer. This is the
  fast path used by the example script and the ablation.

- **Raw expression** (``use_cached_embeddings=False``): the model accepts
  ``(batch, 19_062)`` TPM vectors in the model gene order, tokenizes them with
  :func:`tokenize`, and runs the frozen encoder internally.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.trajsurv import cox_partial_likelihood_loss

N_GENES = 19_062
N_BINS = 64
EMBED_DIM = 256
GENE_EMBED_DIM = 200
NUM_HEADS = 8
FFN_DIM = 512
NUM_LAYERS = 4
LAYER_NORM_EPS = 1e-5
TCGA_NORMALIZATION_FACTOR = 5.547176906585117
HF_REPO_ID = "InstaDeepAI/BulkRNABert"
# The checkpoint our parity test validated against (encoder max abs diff
# 3.71e-05, tokenizer bit-for-bit identical). Pinned so future users
# reproduce the same behavior.
HF_DEFAULT_REVISION = "7bcae6ac4f68fe1c781e8c1a626d8ee023747228"
def _unpack(v):
    if isinstance(v, (tuple, list)):
        return v[0]
    return v

def tokenize(tpm: np.ndarray, normalization_factor: float = TCGA_NORMALIZATION_FACTOR) -> np.ndarray:
    """Discretize TPM into 64 expression bins (matches the paper's tokenizer).
    Args:
        tpm: Array of shape ``(n_samples, N_GENES)`` in model gene order.
        normalization_factor: Max ``log10(1 + TPM)`` value from pre-training.

    Returns:
        int64 array of shape ``(n_samples, N_GENES)`` with bin ids in ``[0, 63]``.

    Raises:
        ValueError: If ``tpm`` is not 2D with ``N_GENES`` columns, contains
            non-finite values, or contains negative values.
    """
    if not np.isfinite(normalization_factor) or normalization_factor <= 0:
        raise ValueError(
            f"normalization_factor must be finite and positive, "
            f"got {normalization_factor}"
        )
    tpm = np.asarray(tpm, dtype=np.float64)
    if tpm.ndim != 2:
        raise ValueError(f"tpm must be 2D, got shape {tpm.shape}")
    if tpm.shape[1] != N_GENES:
        raise ValueError(
            f"tpm must have {N_GENES} genes (model gene order), got {tpm.shape[1]}"
        )
    if not np.isfinite(tpm).all():
        raise ValueError("tpm contains non-finite values (NaN or inf)")
    if (tpm < 0).any():
        raise ValueError("tpm contains negative values")

    x = np.log10(1.0 + tpm) / normalization_factor
    edges = np.linspace(0.0, 1.0, N_BINS)
    ids = np.digitize(x, edges)
    ids[x == 0.0] = 0
    return np.clip(ids, 0, N_BINS - 1).astype(np.int64)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.w_k = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.w_v = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.output = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, NUM_HEADS, EMBED_DIM // NUM_HEADS).transpose(1, 2)
        out = F.scaled_dot_product_attention(heads(self.w_q(x)), heads(self.w_k(x)), heads(self.w_v(x)))
        return self.output(out.transpose(1, 2).reshape(b, n, EMBED_DIM))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm_self_attention = nn.LayerNorm(EMBED_DIM, eps=LAYER_NORM_EPS)
        self.mha = Attention()
        self.layer_norm_mlp = nn.LayerNorm(EMBED_DIM, eps=LAYER_NORM_EPS)
        self.fc1 = nn.Linear(EMBED_DIM, FFN_DIM)
        self.fc2 = nn.Linear(FFN_DIM, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.layer_norm_self_attention(x))
        return x + self.fc2(F.gelu(self.fc1(self.layer_norm_mlp(x)), approximate="none"))


class BulkRNABertEncoder(nn.Module):
    """Frozen BulkRNABert encoder: token ids -> per-gene embeddings (B, N_GENES, 256)."""

    def __init__(self, n_genes: int = N_GENES) -> None:
        super().__init__()
        self.expression_embedding_layer = nn.Embedding(N_BINS, EMBED_DIM)
        self.gene_embedding_layer = nn.Embedding(n_genes, GENE_EMBED_DIM)
        self.fc_gene_embedding = nn.Linear(GENE_EMBED_DIM, EMBED_DIM)
        self.transformer_layers = nn.ModuleList(Block() for _ in range(NUM_LAYERS))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        genes = self.fc_gene_embedding(self.gene_embedding_layer.weight)
        x = self.expression_embedding_layer(tokens) + genes
        for layer in self.transformer_layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def mean_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self(tokens).mean(dim=1)


class SurvivalHead(nn.Module):
    """MLP head: 256 -> 512 (SELU) -> 256 (SELU) -> LayerNorm -> 1."""

    def __init__(self, in_dim: int = 256, hidden: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SELU()]
            width = size
        self.hidden = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.out = nn.Linear(width, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = module.in_features ** -0.5
                nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.hidden(x))).squeeze(-1)


def cox_loss(
    risk: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
    group: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean negative Cox partial log-likelihood (Breslow ties).

    Delegates to :func:`pyhealth.models.trajsurv.cox_partial_likelihood_loss`
    for the global case. With ``group``, one risk set is built per cohort
    and the per-cohort losses are combined by an event-count-weighted
    average, so the result is still a mean over all observed events rather
    than a mean of per-cohort means (which would weight small cohorts
    equally with large ones).

    Args:
        risk: Risk scores, shape ``(batch,)``.
        duration: Survival times, shape ``(batch,)``.
        event: Event indicators, shape ``(batch,)``.
        group: Optional cohort ids, shape ``(batch,)``.

    Returns:
        Scalar loss. If no events are present, returns a zero-valued tensor
        that still propagates gradients (safe for minibatches).
    """
    if group is None:
        return cox_partial_likelihood_loss(risk, duration, event)

    total_weighted_loss = 0.0
    total_events = 0.0
    for g in group.unique():
        mask = group == g
        n_events = float(event[mask].sum().item())
        if n_events == 0:
            continue
        per_group_loss = cox_partial_likelihood_loss(
            risk[mask], duration[mask], event[mask]
        )
        total_weighted_loss = total_weighted_loss + per_group_loss * n_events
        total_events += n_events
    if total_events == 0:
        return risk.sum() * 0.0
    return total_weighted_loss / total_events

# --- Weight loaders ---

def load_hf_safetensors(
    path_or_repo: str | Path = HF_REPO_ID,
    revision: str | None = None,
) -> BulkRNABertEncoder:
    """Load BulkRNABert weights from a safetensors file or the Hugging Face Hub.

    Args:
        path_or_repo: Either a path to a local ``.safetensors`` file (or a
            directory containing ``model.safetensors``), or a Hugging Face
            repository ID such as ``"InstaDeepAI/BulkRNABert"``. Defaults to
            the official repo.
        revision: Commit hash or branch to pin when downloading from HF Hub.
            Ignored when loading a local file.
    
    Returns:
        A ``BulkRNABertEncoder`` in eval mode. Parameters are **not** frozen
        here; the constructor freezes them (freeze here would break users who
        want to fine-tune the encoder).
    """
    from safetensors.torch import load_file

    p = Path(path_or_repo)
    if p.is_file():
        weights_file = str(p)
    elif p.is_dir():
        weights_file = str(p / "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        repo = str(path_or_repo) if path_or_repo else HF_REPO_ID
        weights_file = hf_hub_download(
            repo_id=repo, filename="model.safetensors", revision=revision,
        )

    state = load_file(weights_file)
    state = {k: v for k, v in state.items() if not k.startswith("lm_head.")}
    embedding_table = state["expression_embedding_layer.weight"]
    if embedding_table.shape[0] < N_BINS:
        raise ValueError(
            f"expression_embedding_layer has {embedding_table.shape[0]} rows; "
            f"expected at least {N_BINS} (64 expression bins)"
        )
    state["expression_embedding_layer.weight"] = embedding_table[:N_BINS]
    model = BulkRNABertEncoder()
    model.load_state_dict(state, strict=True)
    return model.eval()


def _rebuild_jax_array(fun, args, arr_state, aval_state):
    array = fun(*args)
    array.__setstate__(arr_state)
    return array


class _HaikuUnpickler(pickle.Unpickler):
    """Restricted unpickler for Haiku ``params.joblib`` (no JAX needed)."""
    ALLOWED = {
    ("jax._src.array", "_reconstruct_array"): _rebuild_jax_array,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}
    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")
        return self.ALLOWED[(module, name)]


def load_jax_checkpoint(path: str) -> BulkRNABertEncoder:
    """Load the authors' Haiku ``params.joblib`` encoder checkpoint."""
    with open(path, "rb") as fh:
        params = _HaikuUnpickler(fh).load()
    flat = {f"{m}|{n}": np.asarray(v) for m, vs in params.items() for n, v in vs.items()}
    state: dict[str, torch.Tensor] = {}
    def linear(prefix: str, target: str) -> None:
        state[f"{target}.weight"] = torch.from_numpy(flat[f"{prefix}|w"].T).float()
        state[f"{target}.bias"] = torch.from_numpy(flat[f"{prefix}|b"]).float()
    def norm(prefix: str, target: str) -> None:
        state[f"{target}.weight"] = torch.from_numpy(flat[f"{prefix}|scale"]).float()
        state[f"{target}.bias"] = torch.from_numpy(flat[f"{prefix}|offset"]).float()
    expr_table = torch.from_numpy(
        flat["bulk_bert/~/expression_embedding|embeddings"]).float()
    if expr_table.shape[0] < N_BINS:
        raise ValueError(
            f"JAX expression_embedding_layer has {expr_table.shape[0]} rows; "
            f"expected at least {N_BINS} (64 expression bins)"
        )
    state["expression_embedding_layer.weight"] = expr_table[:N_BINS]
    state["gene_embedding_layer.weight"] = torch.from_numpy(
        flat["bulk_bert/~/gene_embedding|embeddings"]).float()
    linear("bulk_bert/~/linear", "fc_gene_embedding")
    for i in range(NUM_LAYERS):
        src = f"bulk_bert/self_attention_block_{i}/~/"
        dst = f"transformer_layers.{i}."
        norm(src + "self_attention_layer_norm", dst + "layer_norm_self_attention")
        norm(src + "final_layer_norm", dst + "layer_norm_mlp")
        linear(src + "fc1", dst + "fc1")
        linear(src + "fc2", dst + "fc2")
        for hn, n in [("query", "w_q"), ("key", "w_k"), ("value", "w_v"), ("mha_output", "output")]:
            linear(src + "self_attention/" + hn, dst + "mha." + n)
    model = BulkRNABertEncoder()
    model.load_state_dict(state, strict=True)
    return model.eval()


# --- PyHealth model ---

class BulkRNABert(BaseModel):
    """BulkRNABert survival model (frozen encoder + trainable MLP head).

     Two modes for ``embedding`` inputs:

    - **Cached embeddings** (``use_cached_embeddings=True``): ``embedding`` is
      a precomputed ``(batch, 256)`` tensor (mean over genes of the frozen
      encoder's last transformer layer). Skips the encoder entirely.
    - **Raw expression** (``use_cached_embeddings=False``): ``embedding`` is
      a ``(batch, 19_062)`` TPM tensor in the model gene order. The model
      tokenizes and runs the frozen encoder to produce 256-d embeddings.

    The Cox partial likelihood is computed inside ``forward()`` and returned
    as ``out["loss"]``, following the ``MetabricSurvivalPrediction`` and
    ``InHospitalSurvivalMIMIC4`` task conventions in this repo.

    Args:
        dataset: PyHealth ``SampleDataset`` (optional when use_cached_embeddings
            is True and the caller provides ``embedding`` in kwargs).
        encoder: ``"hf"`` (safetensors) or ``"repo"`` (JAX ``params.joblib``).
        encoder_source: HF repository ID, local ``.safetensors`` file/directory,
            or path to a JAX ``params.joblib`` (depending on ``encoder``).
            Defaults to the official ``InstaDeepAI/BulkRNABert`` repository.
        revision: HF Hub commit hash or branch to pin when downloading.
            Ignored for local files and for ``encoder="repo"``.
        use_cached_embeddings: Skip the encoder and treat inputs as 256-d.
        feature_key: Fallback input field name when ``dataset`` is None.
        cox_loss_mode: ``"global"`` (paper default) or ``"cohort"``
            (risk sets restricted to the same cancer cohort).

    Example:
        >>> from pyhealth.models import BulkRNABert
        >>> # Downloaded directly from HF Hub, no local path needed:
        >>> model = BulkRNABert(use_cached_embeddings=False)
        >>> # Or from a local checkpoint:
        >>> model = BulkRNABert(
        ...     encoder_source="/path/to/model.safetensors",
        ...     use_cached_embeddings=False,
        ... )
    """
    def __init__(
        self,
        dataset: SampleDataset | None = None,
        encoder: str = "hf",
        encoder_source: str | Path = HF_REPO_ID,
        revision: str | None = HF_DEFAULT_REVISION,
        use_cached_embeddings: bool = False,
        feature_key: str = "embedding",
        cox_loss_mode: str = "global"
    ) -> None:
        super().__init__(dataset=dataset)
        self._feature_key = feature_key
        if cox_loss_mode not in {"global", "cohort"}:
            raise ValueError(
                f"cox_loss_mode must be 'global' or 'cohort', got {cox_loss_mode!r}"
            )
        self.cox_loss_mode = cox_loss_mode
        self.use_cached_embeddings = use_cached_embeddings
        self.encoder: Optional[BulkRNABertEncoder] = None
        if not use_cached_embeddings:
            if encoder == "hf":
                self.encoder = load_hf_safetensors(
                    encoder_source, revision=revision
                )
            elif encoder == "repo":
                source_path = Path(encoder_source)
                if str(encoder_source) == HF_REPO_ID or not source_path.exists():
                    raise ValueError(
                        "encoder='repo' requires encoder_source to be a local "
                        "path to a JAX params.joblib checkpoint, not the default "
                        "HF repository ID. Example: "
                        "encoder_source='data/models/repo/bulk_rna_bert_tcga/params.joblib'"
                    )
                self.encoder = load_jax_checkpoint(str(source_path))
            else:
                raise ValueError(f"Unknown encoder: {encoder!r}")
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.head = SurvivalHead()

    def train(self, mode: bool = True) -> "BulkRNABert":
        """Keep the frozen encoder in eval mode even when the model is training."""
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cached_embeddings:
            if x.shape[-1] != EMBED_DIM:
                raise ValueError(
                    f"Cached embedding must have last dim {EMBED_DIM}, got {x.shape[-1]}"
                )
            return x
        tokens = torch.from_numpy(tokenize(x.detach().cpu().numpy())).to(x.device)
        with torch.no_grad():
            return self.encoder.mean_embedding(tokens)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute risk scores (and loss if survival labels are present).

        Returns:
            logit: (B, 1) unbounded Cox risk scores.
            y_prob: same as logit (Cox scores are not calibrated probabilities).
            loss: scalar Cox loss (only when ``duration`` and ``event``
                are in ``kwargs``).
            y_true: (B, 1) durations (only when labels are present).
        """
        key = self.feature_keys[0] if self.feature_keys else self._feature_key
        raw = _unpack(kwargs[key])
        emb = self._encode(raw)
        logit = self.head(emb).unsqueeze(-1)

        out: dict[str, torch.Tensor] = {"logit": logit, "y_prob": logit, "risk_score": logit,}

        if "duration" in kwargs and "event" in kwargs:
            duration = _unpack(kwargs["duration"]).reshape(-1).float()
            event = _unpack(kwargs["event"]).reshape(-1).float()
            group = None
            if self.cox_loss_mode == "cohort":
                if "cohort" not in kwargs:
                    raise ValueError(
                        "cox_loss_mode='cohort' requires cohort labels in the batch"
                    )
                group = _unpack(kwargs["cohort"]).reshape(-1).long()
            out["loss"] = cox_loss(logit.squeeze(-1), duration, event, group)
            out["y_true"] = _unpack(kwargs["duration"])

        return out