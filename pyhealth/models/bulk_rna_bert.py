"""BulkRNABert survival model for PyHealth.

Reproduces the downstream survival prediction pipeline of Gélard et al. (2025),
"BulkRNABert: Cancer prognosis from bulk RNA-seq based language models"
(ML4H 2025), using the authors' released frozen encoder and a newly initialized
MLP survival head trained with Cox partial likelihood.

Reference implementation:
    https://github.com/instadeepai/multiomics-open-research
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

N_GENES = 19_062
N_BINS = 64
EMBED_DIM = 256
GENE_EMBED_DIM = 200
NUM_HEADS = 8
FFN_DIM = 512
NUM_LAYERS = 4
LAYER_NORM_EPS = 1e-5
TCGA_NORMALIZATION_FACTOR = 5.547176906585117


def tokenize(tpm: np.ndarray, normalization_factor: float = TCGA_NORMALIZATION_FACTOR) -> np.ndarray:
    """Discretize TPM into 64 expression bins (matches the paper's tokenizer)."""
    x = np.log10(1.0 + np.asarray(tpm, dtype=np.float64)) / normalization_factor
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


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor,
             group: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean negative Cox partial log-likelihood (Breslow ties).

    Args:
        risk: (B,) risk scores.
        time: (B,) survival times.
        event: (B,) event indicators.
        group: (B,) optional cohort ids. If set, risk sets are per-cohort.

    Returns:
        Scalar loss (zero-valued if no events in the batch, still differentiable).
    """
    if group is None:
        group = torch.zeros_like(time, dtype=torch.long)
    losses = []
    for g in group.unique():
        mask = group == g
        r, t, e = risk[mask], time[mask], event[mask]
        if e.sum() == 0:
            continue
        order = torch.argsort(t, descending=True)
        r, t, e = r[order], t[order], e[order]
        log_risk = torch.logcumsumexp(r, dim=0)
        last_tied = torch.searchsorted(-t, -t, right=True) - 1
        losses.append(-(r - log_risk[last_tied])[e.bool()])
    if not losses:
        return risk.sum() * 0.0
    return torch.cat(losses).mean()


# --- Weight loaders ---

def load_hf_safetensors(path: str) -> BulkRNABertEncoder:
    """Load the Hugging Face safetensors weights for BulkRNABert."""
    from safetensors.torch import load_file
    state = load_file(path)
    state = {k: v for k, v in state.items() if not k.startswith("lm_head.")}
    state["expression_embedding_layer.weight"] = state["expression_embedding_layer.weight"][:N_BINS]
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
    state["expression_embedding_layer.weight"] = torch.from_numpy(
        flat["bulk_bert/~/expression_embedding|embeddings"]).float()
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

    Follows the PyHealth 2.1 survival-task convention (see ``Metabric`` and
    ``InHospitalSurvivalMIMIC4``): the Cox loss is computed inside ``forward``
    and returned as ``output["loss"]``; no ``LabelProcessor`` is created.

    Args:
        dataset: PyHealth ``SampleDataset``. Required by ``BaseModel``.
        encoder: ``"hf"`` (safetensors) or ``"repo"`` (Haiku ``params.joblib``).
        models_dir: Directory containing the encoder weights.
        use_cached_embeddings: Skip the encoder and treat inputs as 256-d.

    Example:
        >>> # with cached embeddings:
        >>> import torch
        >>> from pyhealth.models import BulkRNABert
        >>> m = BulkRNABert(dataset=None, use_cached_embeddings=True)
        >>> out = m.forward(gene_expression=torch.randn(4, 256))
        >>> out["logit"].shape
        torch.Size([4, 1])
    """
    _feature_key: str = "gene_expression" 
    def __init__(
        self,
        dataset: SampleDataset | None = None,
        encoder: str = "hf",
        models_dir: str | None = None,
        use_cached_embeddings: bool = False,
        feature_key: str = "gene_expression",
    ) -> None:
        super().__init__(dataset=dataset)
        feature_key: str = "gene_expression",
        self.use_cached_embeddings = use_cached_embeddings
        self.encoder: Optional[BulkRNABertEncoder] = None
        if not use_cached_embeddings:
            if models_dir is None:
                raise ValueError("models_dir is required when use_cached_embeddings=False")
            if encoder == "hf":
                self.encoder = load_hf_safetensors(f"{models_dir}/hf_bulkrnabert/model.safetensors")
            elif encoder == "repo":
                self.encoder = load_jax_checkpoint(f"{models_dir}/repo/bulk_rna_bert_tcga/params.joblib")
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
            return x
        tokens = torch.from_numpy(tokenize(x.detach().cpu().numpy())).to(x.device)
        with torch.no_grad():
            return self.encoder.mean_embedding(tokens)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute risk scores (and loss if survival labels are present).

        Returns:
            logit: (B, 1) unbounded Cox risk scores.
            y_prob: same as logit (Cox scores are not calibrated probabilities).
            loss: scalar Cox loss (only when ``survival_time`` and ``event``
                are in ``kwargs``).
            y_true: (B, 1) survival times (only when labels are present).
        """
        key = self.feature_keys[0] if self.feature_keys else self._feature_key
        raw = kwargs[key]
        if isinstance(raw, torch.Tensor):
            x = raw
        elif isinstance(raw, (tuple, list)):
            x = raw[0]
        else:
            raise TypeError(f"Unexpected feature type: {type(raw)}")

        emb = self._encode(x)
        logit = self.head(emb).unsqueeze(-1)

        out: dict[str, torch.Tensor] = {"logit": logit, "y_prob": logit}

        if "survival_time" in kwargs and "event" in kwargs:
            time = kwargs["survival_time"].squeeze(-1).float()
            event = kwargs["event"].squeeze(-1).float()
            cohort = kwargs.get("cohort")
            if cohort is not None:
                cohort = cohort.squeeze(-1).long()
            out["loss"] = cox_loss(logit.squeeze(-1), time, event, cohort)
            out["y_true"] = kwargs["survival_time"]

        return out