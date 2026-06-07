"""HALO: Hierarchical Autoregressive Language mOdel for synthetic EHR generation.

This is a faithful port of the reference implementation
(https://github.com/Brandon-Theodorou/HALO_Inpatient) wrapped as a PyHealth
``BaseModel`` so it consumes the standard
``dataset -> set_task -> SampleDataset -> model`` pipeline.

HALO is a two-level model:

* a GPT-2-style **coarse** transformer operates over visit-level multi-hot
  vectors, and
* a **fine** autoregressive head predicts the (multi-label) set of codes within
  each visit.

The transformer/head classes below (``LayerNorm``, ``Conv1D``, ``Attention``,
``MLP``, ``Block``, ``CoarseTransformerModel``, ``AutoregressiveLinear``,
``FineAutoregressiveHead``, ``HALOModel``) are ported verbatim from the
reference ``model.py``. The only behavioural change is that PyHealth's HALO is
**unconditional** (``label_vocab_size = 0``): it generates visit-code sequences
without conditioning on CCS labels.
"""

import copy
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel


# ----------------------------------------------------------------------------
# Configuration (plain class, not a dataclass; mirrors reference config.py)
# ----------------------------------------------------------------------------
class HALOConfig:
    """Hyperparameter container for the HALO transformer.

    Kept as a plain class with explicit ``__init__`` assignments (matching the
    reference ``config.py``) so the low-level modules can read attributes such
    as ``config.n_embd``.
    """

    def __init__(
        self,
        total_vocab_size: int,
        code_vocab_size: int,
        label_vocab_size: int = 0,
        special_vocab_size: int = 3,
        n_positions: int = 56,
        n_ctx: int = 48,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        batch_size: int = 48,
        epoch: int = 50,
        pos_loss_weight: Optional[float] = None,
        lr: float = 1e-4,
    ) -> None:
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.epoch = epoch
        self.pos_loss_weight = pos_loss_weight
        self.lr = lr


# ----------------------------------------------------------------------------
# Transformer building blocks (ported verbatim from reference model.py)
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside sqrt)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=n_embd (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=4 * n_embd
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)

    def forward(self, x):
        # tanh-approximate GELU, matching the reference HALO implementation.
        h = F.gelu(self.c_fc(x), approximate="tanh")
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class CoarseTransformerModel(nn.Module):
    def __init__(self, config):
        super(CoarseTransformerModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(
            config.total_vocab_size, config.n_embd, bias=False
        )
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_visits, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_visits.size(1) + past_length,
                dtype=torch.long,
                device=input_visits.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(
                input_visits.size(0), input_visits.size(1)
            )

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for block, layer_past in zip(self.h, past):
            hidden_states, _ = block(hidden_states, layer_past)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class AutoregressiveLinear(nn.Linear):
    """Same as Linear except it has a configurable mask on the weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer(
            "mask", torch.tril(torch.ones(in_features, out_features)).int()
        )

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class FineAutoregressiveHead(nn.Module):
    def __init__(self, config):
        super(FineAutoregressiveHead, self).__init__()
        self.auto1 = AutoregressiveLinear(
            config.n_embd + config.total_vocab_size,
            config.n_embd + config.total_vocab_size,
        )
        self.auto2 = AutoregressiveLinear(
            config.n_embd + config.total_vocab_size,
            config.n_embd + config.total_vocab_size,
        )
        self.n_embd = config.n_embd
        self.tot_vocab = config.total_vocab_size

    def forward(self, history, input_visits):
        history = history[:, :-1, :]
        input_visits = input_visits[:, 1:, :]
        code_logits = self.auto2(
            torch.relu(self.auto1(torch.cat((history, input_visits), dim=2)))
        )[:, :, self.n_embd - 1:-1]
        return code_logits

    def sample(self, history, input_visits):
        history = history[:, :-1, :]
        input_visits = input_visits[:, 1:, :]
        currVisit = torch.cat((history, input_visits), dim=2)[:, -1, :].unsqueeze(1)
        code_logits = self.auto2(torch.relu(self.auto1(currVisit)))[
            :, :, self.n_embd - 1:-1
        ]
        return code_logits


class HALOModel(nn.Module):
    """Low-level HALO transformer + autoregressive head (ported verbatim)."""

    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)

    def forward(
        self,
        input_visits,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        past=None,
        pos_loss_weight=None,
    ):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, input_visits)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            loss_weights = None
            if pos_loss_weight is not None:
                loss_weights = torch.ones(
                    code_probs.shape, device=code_probs.device
                )
                loss_weights = loss_weights + (pos_loss_weight - 1) * shift_labels
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks
                if pos_loss_weight is not None:
                    loss_weights = loss_weights * ehr_masks

            bce = nn.BCELoss(weight=loss_weights)
            loss = bce(code_probs, shift_labels)
            return loss, code_probs, shift_labels

        return code_probs

    def sample(self, input_visits, random=True):
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits)
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, input_visits)
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)

            remaining_visit = visit[:, 0, i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:, -1, i + first_nonzero] = visit[:, 0, i + first_nonzero]
            i = i + first_nonzero + 1

        return input_visits


# ----------------------------------------------------------------------------
# PyHealth BaseModel wrapper
# ----------------------------------------------------------------------------
class HALO(BaseModel):
    """HALO synthetic-EHR generator, wrapped as a PyHealth ``BaseModel``.

    Trains a GPT-2-style transformer on patient visit-code sequences and
    generates synthetic patients by autoregressive sampling. Generation is
    **unconditional** (no label conditioning).

    The model infers its code vocabulary from the fitted ``SampleDataset``:
    ``code_vocab_size = dataset.input_processors["visits"].vocab_size()``
    (the ``NestedSequenceProcessor`` vocab, which already reserves index 0 for
    ``<pad>`` and index 1 for ``<unk>``). Three special tokens are appended for
    start-of-sequence, end-of-sequence, and pad-visit.

    Args:
        dataset: A fitted ``SampleDataset`` whose ``input_schema`` contains
            ``{"visits": NestedSequenceProcessor}`` and whose ``output_schema``
            is empty.
        embed_dim: Transformer embedding dimension (``n_embd``). Default: 768.
        n_heads: Number of attention heads. Must divide ``embed_dim``.
            Default: 12.
        n_layers: Number of transformer layers. Default: 12.
        n_ctx: Maximum number of visit positions (context length). Default: 48.
        batch_size: Training batch size. Default: 48.
        epochs: Number of training epochs. Default: 50.
        pos_loss_weight: Positive-class weight for the BCE loss. ``None`` means
            no weighting. Default: None.
        lr: Learning rate for the Adam optimizer. Default: 1e-4.
        save_dir: Directory for checkpoints written by ``train_model``.
            Default: ``"./save/"``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p1", "visits": [["A", "B"], ["C"]]},
        ...     {"patient_id": "p2", "visits": [["A"], ["B", "C"]]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"visits": "nested_sequence"},
        ...     output_schema={},
        ... )
        >>> model = HALO(dataset, embed_dim=16, n_heads=2, n_layers=2, n_ctx=8)
        >>> isinstance(model, HALO)
        True
    """

    def __init__(
        self,
        dataset,
        embed_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        n_ctx: int = 48,
        batch_size: int = 48,
        epochs: int = 50,
        pos_loss_weight: Optional[float] = None,
        lr: float = 1e-4,
        save_dir: str = "./save/",
    ) -> None:
        super(HALO, self).__init__(dataset)

        if "visits" not in dataset.input_processors:
            raise ValueError(
                "HALO expects an input feature named 'visits' backed by a "
                "NestedSequenceProcessor."
            )

        self.save_dir = save_dir
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr

        # Code vocab from the NestedSequenceProcessor (includes <pad>, <unk>).
        self.visits_processor = dataset.input_processors["visits"]
        code_vocab_size = self.visits_processor.vocab_size()
        label_vocab_size = 0  # unconditional generation -- no output labels
        # +3 special tokens: start-of-sequence, end-of-sequence, pad-visit.
        total_vocab_size = code_vocab_size + label_vocab_size + 3

        self.config = HALOConfig(
            total_vocab_size=total_vocab_size,
            code_vocab_size=code_vocab_size,
            label_vocab_size=label_vocab_size,
            special_vocab_size=3,
            n_positions=n_ctx + 8,  # position table needs a little slack
            n_ctx=n_ctx,
            n_embd=embed_dim,
            n_layer=n_layers,
            n_head=n_heads,
            batch_size=batch_size,
            epoch=epochs,
            pos_loss_weight=pos_loss_weight,
            lr=lr,
        )

        # Registered as a sub-module so .parameters()/.to() work.
        self.halo_model = HALOModel(self.config)

    # ------------------------------------------------------------------
    # Multi-hot encoding helper
    # ------------------------------------------------------------------
    def _encode_visits(self, visits: torch.Tensor):
        """Convert a padded index tensor to HALO multi-hot format.

        ``NestedSequenceProcessor`` returns code indices; the transformer
        expects multi-hot vectors of shape ``(batch, n_ctx, total_vocab_size)``
        with special tokens. Layout (mirrors the reference): position 0 is the
        start token, visits occupy positions 2+, the end token is placed on the
        last visit's row, and the pad token fills the remaining positions.

        Args:
            visits: LongTensor ``(batch, max_visits, max_codes_per_visit)``.
                Index 0 is ``<pad>`` and is skipped.

        Returns:
            batch_ehr: FloatTensor ``(batch, n_ctx, total_vocab_size)``.
            batch_mask: FloatTensor ``(batch, n_ctx - 1, 1)``, shifted to align
                with the autoregressive prediction targets.
        """
        cfg = self.config
        batch_size = visits.shape[0]

        batch_ehr = torch.zeros(
            batch_size, cfg.n_ctx, cfg.total_vocab_size, device=self.device
        )
        batch_mask = torch.zeros(batch_size, cfg.n_ctx, 1, device=self.device)

        start_idx = cfg.code_vocab_size + cfg.label_vocab_size
        end_idx = start_idx + 1
        pad_idx = start_idx + 2

        for i in range(batch_size):
            # Count actual (non-padding) visits for this patient.
            n_visits = int((visits[i].sum(dim=-1) > 0).sum().item())
            n_visits = min(n_visits, cfg.n_ctx - 2)
            for j in range(n_visits):
                for code_idx in visits[i, j]:
                    if code_idx > 0:  # skip <pad> (index 0)
                        batch_ehr[i, j + 2, code_idx] = 1
                batch_mask[i, j + 2] = 1

            batch_ehr[i, 0, start_idx] = 1            # start token
            batch_ehr[i, n_visits + 1, end_idx] = 1   # end token (on last visit)
            batch_ehr[i, n_visits + 2:, pad_idx] = 1  # pad visits

        batch_mask = batch_mask[:, 1:, :]  # shift to align with shifted targets
        return batch_ehr, batch_mask

    # ------------------------------------------------------------------
    # forward -- required by BaseModel
    # ------------------------------------------------------------------
    def forward(self, visits: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            visits: LongTensor ``(batch, max_visits, max_codes_per_visit)`` from
                the ``NestedSequenceProcessor``.
            **kwargs: Any other batch keys are ignored.

        Returns:
            Dict with ``loss`` (scalar BCE) and ``y_prob`` (code probabilities,
            shape ``(batch, n_ctx - 1, total_vocab_size)``).
        """
        visits = visits.to(self.device)
        batch_ehr, batch_mask = self._encode_visits(visits)

        loss, code_probs, _ = self.halo_model(
            batch_ehr,
            position_ids=None,
            ehr_labels=batch_ehr,
            ehr_masks=batch_mask,
            pos_loss_weight=self.config.pos_loss_weight,
        )
        return {"loss": loss, "y_prob": code_probs}

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device=None) -> torch.device:
        """Resolve a user-supplied device, defaulting to CUDA when available.

        Args:
            device: ``None``, a device string (e.g. ``"cuda"``, ``"cuda:1"``,
                ``"cpu"``), or a ``torch.device``. When ``None``, CUDA is used
                if available, otherwise CPU.
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def train_model(self, train_dataset, val_dataset=None, device=None) -> None:
        """Train the HALO model with a custom loop.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``. Uses the standard ``get_dataloader`` (which pads
        the variable visit dimension for us), an Adam optimizer, and BCE loss.
        When ``val_dataset`` is given, validation loss is computed after each
        epoch and the best checkpoint is saved to ``self.save_dir``.

        Args:
            train_dataset: ``SampleDataset`` for training.
            val_dataset: Optional ``SampleDataset`` for validation.
            device: Device to train on, e.g. ``"cuda"``, ``"cuda:1"``, or
                ``"cpu"``. If ``None`` (default), uses CUDA when available and
                falls back to CPU.
        """
        device = self._resolve_device(device)
        self.to(device)
        print(f"Training on: {device}")

        os.makedirs(self.save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.halo_model.parameters(), lr=self._lr)

        checkpoint_path = os.path.join(self.save_dir, "halo_model")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.halo_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        train_loader = get_dataloader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        global_loss = 1e10
        for epoch in tqdm(range(self._epochs), desc="Epochs"):
            self.halo_model.train()
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in batch_iter:
                visits = batch["visits"].to(self.device)
                batch_ehr, batch_mask = self._encode_visits(visits)

                optimizer.zero_grad()
                loss, _, _ = self.halo_model(
                    batch_ehr,
                    position_ids=None,
                    ehr_labels=batch_ehr,
                    ehr_masks=batch_mask,
                    pos_loss_weight=self.config.pos_loss_weight,
                )
                loss.backward()
                optimizer.step()
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            if val_dataset is not None:
                self.halo_model.eval()
                val_loader = get_dataloader(
                    val_dataset, batch_size=self._batch_size, shuffle=False
                )
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        visits = val_batch["visits"].to(self.device)
                        batch_ehr, batch_mask = self._encode_visits(visits)
                        val_loss, _, _ = self.halo_model(
                            batch_ehr,
                            position_ids=None,
                            ehr_labels=batch_ehr,
                            ehr_masks=batch_mask,
                            pos_loss_weight=self.config.pos_loss_weight,
                        )
                        val_losses.append(val_loss.item())

                cur_val_loss = float(np.mean(val_losses))
                print(f"Epoch {epoch} Validation Loss: {cur_val_loss:.7f}")
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                        "model": self.halo_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(state, checkpoint_path)
                    print("------------ Save best model ------------")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def generate(
        self, num_samples: int, random_sampling: bool = True, device=None
    ) -> List[Dict]:
        """Generate synthetic patients using the trained HALO model.

        Autoregressive sampling: feed a start token and repeatedly call
        ``halo_model.sample`` until an end token is produced or ``n_ctx`` steps
        are reached, then decode code indices back to code strings.

        Args:
            num_samples: Number of synthetic patients to generate.
            random_sampling: If True, Bernoulli sampling (stochastic). If False,
                rounding (deterministic). Default: True.
            device: Device to generate on, e.g. ``"cuda"``, ``"cuda:1"``, or
                ``"cpu"``. If ``None`` (default), uses CUDA when available and
                falls back to CPU.

        Returns:
            List of dicts, each ``{"patient_id": "synthetic_i",
            "visits": [[code, ...], ...]}`` with decoded code strings.
        """
        device = self._resolve_device(device)
        self.to(device)

        cfg = self.config
        index_to_code = {v: k for k, v in self.visits_processor.code_vocab.items()}
        end_token_idx = cfg.code_vocab_size + cfg.label_vocab_size + 1
        start_token_idx = cfg.code_vocab_size + cfg.label_vocab_size

        self.halo_model.eval()
        synthetic_dataset: List[Dict] = []
        sample_batch_size = min(num_samples, 256)
        generated = 0
        pbar = tqdm(total=num_samples, desc="Generating patients")

        with torch.no_grad():
            while generated < num_samples:
                bs = min(sample_batch_size, num_samples - generated)
                stoken = torch.zeros(
                    cfg.total_vocab_size, device=self.device, dtype=torch.float32
                )
                stoken[start_token_idx] = 1
                prev = stoken.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
                empty = torch.zeros(
                    bs, 1, cfg.total_vocab_size,
                    device=self.device, dtype=torch.float32,
                )

                for _ in range(cfg.n_ctx - 1):
                    prev = self.halo_model.sample(
                        torch.cat((prev, empty), dim=1), random_sampling
                    )
                    has_end = prev[:, :, end_token_idx].sum(dim=1).bool()
                    if has_end.all():
                        break

                batch_ehrs = prev.cpu().detach().numpy()
                for i in range(bs):
                    ehr = batch_ehrs[i]  # (seq_len, total_vocab_size)
                    visits_out: List[List[str]] = []
                    # Position 0 is the start token; visits occupy positions 1+.
                    for j in range(1, len(ehr)):
                        indices = np.nonzero(ehr[j])[0]
                        visit_codes: List[str] = []
                        hit_end = False
                        for idx in indices:
                            if idx < cfg.code_vocab_size:
                                code = index_to_code.get(int(idx))
                                if code not in (None, "<pad>", "<unk>"):
                                    visit_codes.append(code)
                            elif idx == end_token_idx:
                                hit_end = True
                        if visit_codes:
                            visits_out.append(visit_codes)
                        if hit_end:
                            break

                    synthetic_dataset.append(
                        {
                            "patient_id": f"synthetic_{generated + i}",
                            "visits": visits_out,
                        }
                    )
                generated += bs
                pbar.update(bs)
            pbar.close()

        return synthetic_dataset
