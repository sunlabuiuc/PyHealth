"""ARMR: Adaptively Responsive network for Medication Recommendation.

Paper: Feiyue Wu, Tianxing Wu, and Shenqi Jing. "ARMR: Adaptively Responsive
Network for Medication Recommendation." IJCAI 2025.
https://arxiv.org/abs/2507.04428

This is a reimplementation from the paper's equations (Section 4), for
PyHealth's drug-recommendation tasks. The authors' code
(https://github.com/seucoin/armr2) was used only to check details that the
paper leaves open; see the class docstring for each choice.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from pyhealth.datasets import SampleDataset
from pyhealth.medcode import ATC
from pyhealth.models.base_model import BaseModel
from pyhealth.models.embedding import EmbeddingModel


def _reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverses the first ``lengths[b]`` steps of each sequence, keeping padding last.

    Args:
        x: tensor of shape [batch, steps, dim].
        lengths: number of valid steps per sequence, shape [batch].

    Returns:
        Tensor of the same shape with each valid prefix reversed.
    """
    steps = x.size(1)
    pos = torch.arange(steps, device=x.device).unsqueeze(0)
    lengths = lengths.unsqueeze(1)
    idx = torch.where(pos < lengths, lengths - 1 - pos, pos)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class SelectiveSSM(nn.Module):
    """A Mamba block (selective state-space model) in pure PyTorch.

    Paper: Albert Gu and Tri Dao. "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces." 2023. The scan is a plain loop over time, which
    is fast enough for visit sequences (tens of steps).

    Args:
        dim: input and output size.
        state_size: SSM state size per channel. Default is 16.
        expand: inner width as a multiple of ``dim``. Default is 2.
        conv_size: width of the causal depthwise convolution. Default is 4.

    Examples:
        >>> block = SelectiveSSM(dim=8)
        >>> block(torch.randn(2, 5, 8)).shape
        torch.Size([2, 5, 8])
    """

    def __init__(
        self, dim: int, state_size: int = 16, expand: int = 2, conv_size: int = 4
    ):
        super().__init__()
        inner = expand * dim
        self.dt_rank = math.ceil(dim / 16)
        self.state_size = state_size
        self.in_proj = nn.Linear(dim, 2 * inner)
        self.conv = nn.Conv1d(
            inner, inner, conv_size, groups=inner, padding=conv_size - 1
        )
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        a = torch.arange(1, state_size + 1, dtype=torch.float).repeat(inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the block over ``x`` of shape [batch, steps, dim]."""
        steps = x.size(1)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :steps].transpose(1, 2)
        u = F.silu(u)
        dt, b, c = self.x_proj(u).split(
            [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # [batch, steps, inner]
        a = -torch.exp(self.a_log)  # [inner, state]
        da = torch.exp(dt.unsqueeze(-1) * a)  # [batch, steps, inner, state]
        dbu = dt.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
        h = torch.zeros_like(da[:, 0])
        ys = []
        for t in range(steps):
            h = da[:, t] * h + dbu[:, t]
            ys.append((h * c[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.d
        return self.out_proj(y * F.silu(gate))


class PiecewiseTemporalLearning(nn.Module):
    """Piecewise Temporal Learning (PTL), ARMR Eq. 1-4.

    The ``n_recent`` most recent visits go through a feedforward network
    (Eq. 1); the earlier, distant visits go through a Mamba block in
    chronological order (Eq. 2); the recent states then attend over the
    distant ones (Eq. 3), and both are concatenated (Eq. 4). Patients with
    fewer than ``n_recent`` visits are zero-padded, and those with no distant
    visit get no attention term.

    The paper writes the recent segment as "the first N visits"; like the
    authors' code, this orders visits from the most recent, so the segment
    is the N visits closest to the prediction.

    Args:
        dim: visit representation size.
        n_recent: number of recent visits N.

    Examples:
        >>> ptl = PiecewiseTemporalLearning(dim=8, n_recent=2)
        >>> out = ptl(torch.randn(3, 4, 8), torch.tensor([4, 2, 1]))
        >>> out.shape
        torch.Size([3, 4, 8])
    """

    def __init__(self, dim: int, n_recent: int):
        super().__init__()
        self.n_recent = n_recent
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm_near = nn.LayerNorm(dim)
        self.norm_far = nn.LayerNorm(dim)
        self.mamba = SelectiveSSM(dim)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encodes visits.

        Args:
            seq: visit representations in chronological order, padding last,
                shape [batch, visits, dim].
            lengths: number of valid visits per patient, shape [batch].

        Returns:
            ``h_S`` of shape [batch, 2 * n_recent, dim].
        """
        batch, visits, dim = seq.shape
        n = self.n_recent
        if visits < n + 1:  # room for n recent visits and one distant visit
            seq = torch.cat([seq, seq.new_zeros(batch, n + 1 - visits, dim)], 1)
        recent_first = _reverse_padded(seq, lengths)
        near = recent_first[:, :n]
        near_valid = (torch.arange(n, device=seq.device) < lengths.unsqueeze(1))
        near = near * near_valid.unsqueeze(-1)
        h_near = self.norm_near(near + self.ff(near))  # Eq. 1

        far_len = (lengths - n).clamp(min=0)
        far = _reverse_padded(recent_first[:, n:], far_len)  # oldest first
        h_far = self.mamba(far)  # Eq. 2
        far_valid = torch.arange(far.size(1), device=seq.device) < far_len.unsqueeze(1)
        scores = torch.bmm(h_near, h_far.transpose(1, 2))
        scores = scores.masked_fill(~far_valid.unsqueeze(1), float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1))  # no distant visit
        h_far = self.norm_far(h_near + torch.bmm(attn, h_far))  # Eq. 3
        return torch.cat([h_near, h_far], dim=1)  # Eq. 4


class ARMR(BaseModel):
    """ARMR medication recommendation model.

    Paper: Feiyue Wu, Tianxing Wu, and Shenqi Jing. "ARMR: Adaptively
    Responsive Network for Medication Recommendation." IJCAI 2025.

    ARMR encodes each code history with Piecewise Temporal Learning
    (:class:`PiecewiseTemporalLearning`): recent visits through a feedforward
    network and distant visits through a Mamba block. Its Adaptively
    Responsive Module splits the medication history into drugs that were
    reused and drugs that were new at each visit, encodes both, and weights
    every drug's embedding by whether the patient took it before, with a
    learned time decay for past use. The recommendation fuses a direct score
    from the patient representation with the cosine similarity between the
    patient and each weighted drug embedding (Eq. 19). The loss is
    ``alpha * BCE + (1 - alpha) * multi-label margin`` (Eq. 22, alpha 0.7).

    Choices where the paper is open:

    - Eq. 9 uses one learned decay rate and scale, ``alpha * exp(-beta *
      lag)`` summed over the visits in which drug ``j`` was given, where
      ``lag`` counts visits back from the prediction.
    - Eq. 11's attention maps the drug-history attention to a per-feature
      weight (``W_o`` and ``W_n`` are ``2N -> dim``), so the weighting changes
      the direction of a drug's embedding, which cosine similarity sees.
    - ``use_graph`` (default ``True``) adds two-layer GCN embeddings of the
      drug co-occurrence graph and the drug-drug interaction graph to the
      drug embeddings, with weights 1 and 0.5, as the authors' released code
      does; the paper's text does not describe this. Set it to ``False`` for
      the equations as written.

    Note:
        Takes ``conditions``, ``procedures`` and ``drugs_hist`` (nested
        per-visit codes, the current visit's drugs left out) and predicts the
        ``drugs`` label, as :class:`~pyhealth.tasks.DrugRecommendationMIMIC3`
        and :class:`~pyhealth.tasks.DrugRecommendationMIMIC4` produce. Drug
        codes are ATC level 3, which the DDI graph needs.

    Args:
        dataset: the training dataset, used for vocabularies and the graphs.
        embedding_dim: code and visit embedding size. Default is 64.
        n_recent: number of recent visits ``N`` handled by the feedforward
            branch. Default is 3.
        alpha: weight of the BCE term in the loss (Eq. 22). Default is 0.7.
        dropout: dropout on the patient representation. Default is 0.3.
        use_graph: add the co-occurrence and DDI graph embeddings to the drug
            embeddings, as the authors' code does. Default is True.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v1",
        ...      "conditions": [["d1", "d2"], ["d3"]],
        ...      "procedures": [["p1"], ["p2"]],
        ...      "drugs_hist": [["A01A"], []],
        ...      "drugs": ["A01A", "B01A"]},
        ...     {"patient_id": "p1", "visit_id": "v2",
        ...      "conditions": [["d2"], ["d1"], ["d3"]],
        ...      "procedures": [["p2"], ["p1"], ["p1"]],
        ...      "drugs_hist": [["B01A"], ["C03C"], []],
        ...      "drugs": ["C03C"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "nested_sequence",
        ...                   "procedures": "nested_sequence",
        ...                   "drugs_hist": "nested_sequence"},
        ...     output_schema={"drugs": "multilabel"},
        ...     dataset_name="demo",
        ... )
        >>> model = ARMR(dataset, embedding_dim=16, use_graph=False)
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> sorted(out)
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 64,
        n_recent: int = 3,
        alpha: float = 0.7,
        dropout: float = 0.3,
        use_graph: bool = True,
    ):
        super().__init__(dataset=dataset)
        for key in ("conditions", "procedures", "drugs_hist"):
            if key not in self.dataset.input_schema:
                raise ValueError(
                    f"ARMR needs '{key}' in the input schema, as "
                    "pyhealth.tasks.DrugRecommendationMIMIC3/4 produce"
                )
        if len(self.label_keys) != 1:
            raise ValueError("ARMR predicts exactly one label (the drugs)")
        self.label_key = self.label_keys[0]
        self.feature_keys = ["conditions", "procedures"]
        self.embedding_dim = embedding_dim
        self.n_recent = n_recent
        self.alpha = alpha
        self.use_graph = use_graph

        label_vocab = self.dataset.output_processors[self.label_key].label_vocab
        self.label_size = len(label_vocab)
        dim = embedding_dim

        # Patient representation (Eq. 12-15)
        self.embedding_model = EmbeddingModel(dataset, dim)
        self.ptl_d = PiecewiseTemporalLearning(dim, n_recent)
        self.ptl_p = PiecewiseTemporalLearning(dim, n_recent)

        # Adaptively Responsive Module (Eq. 5-11)
        self.register_buffer("_hist_to_label", self._hist_vocab_map(label_vocab))
        self.drug_emb = nn.Parameter(torch.empty(self.label_size, dim))
        nn.init.xavier_uniform_(self.drug_emb)
        self.ptl_reused = PiecewiseTemporalLearning(dim, n_recent)
        self.ptl_new = PiecewiseTemporalLearning(dim, n_recent)
        self.w_reused = nn.Linear(2 * n_recent, dim, bias=False)
        self.w_new = nn.Linear(2 * n_recent, dim, bias=False)
        self.decay_scale = nn.Parameter(torch.tensor(1.0))  # alpha in Eq. 9
        self.decay_rate = nn.Parameter(torch.tensor(0.0))  # softplus -> beta

        if use_graph:
            ddi_adj = self._ddi_adj(label_vocab)
            self.register_buffer("ehr_adj", _normalize(self._ehr_adj(label_vocab)))
            self.register_buffer("ddi_adj_norm", _normalize(ddi_adj))
            self.gcn_ehr = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
            self.gcn_ddi = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
            os.makedirs(CACHE_PATH, exist_ok=True)
            np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())

        # Joint recommendation (Eq. 16-19)
        patient_size = 4 * n_recent * dim
        self.dropout = nn.Dropout(dropout)
        self.ff_direct = nn.Sequential(
            nn.Linear(patient_size, dim), nn.ReLU(), nn.Linear(dim, self.label_size)
        )
        self.ff_query = nn.Sequential(
            nn.Linear(patient_size, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.w_direct = nn.Parameter(torch.tensor(1.0))
        self.w_sim = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------ vocabulary
    def _hist_vocab_map(self, label_vocab: dict) -> torch.Tensor:
        """Maps drugs_hist input indices to label indices; others to a trash bin."""
        hist_vocab = self.dataset.input_processors["drugs_hist"].code_vocab
        mapping = torch.full((len(hist_vocab),), self.label_size, dtype=torch.long)
        for code, idx in hist_vocab.items():
            if code in label_vocab:
                mapping[idx] = label_vocab[code]
        return mapping

    def _ehr_adj(self, label_vocab: dict) -> torch.Tensor:
        """Co-occurrence graph of the drugs given together in a training visit."""
        adj = torch.zeros(self.label_size, self.label_size)
        for sample in self.dataset:
            drugs = sample[self.label_key]
            if isinstance(drugs, torch.Tensor):  # processed: multi-hot
                ids = torch.nonzero(drugs.flatten()).flatten()
            else:
                ids = torch.tensor([label_vocab[c] for c in drugs if c in label_vocab])
            if ids.numel() > 1:
                adj[ids.unsqueeze(1), ids.unsqueeze(0)] = 1.0
        adj.fill_diagonal_(0.0)
        return adj

    def _ddi_adj(self, label_vocab: dict) -> torch.Tensor:
        """Drug-drug interaction graph at ATC level 3 (as GAMENet and SafeDrug)."""
        adj = torch.zeros(self.label_size, self.label_size)
        for a, b in ATC().get_ddi(gamenet_ddi=True):
            a, b = ATC.convert(a, level=3), ATC.convert(b, level=3)
            if a in label_vocab and b in label_vocab:
                adj[label_vocab[a], label_vocab[b]] = 1.0
                adj[label_vocab[b], label_vocab[a]] = 1.0
        return adj

    # ---------------------------------------------------------------- pieces
    def _visit_lengths(self, codes: torch.Tensor) -> torch.Tensor:
        """Number of visits with at least one code, per patient."""
        return (codes > 0).any(dim=-1).sum(dim=1)

    def _drug_history(self, drugs_hist: torch.Tensor, visits: int) -> torch.Tensor:
        """Multi-hot drugs of each visit in the label space, [batch, visits, M]."""
        if drugs_hist.size(1) < visits:
            pad = drugs_hist.new_zeros(
                drugs_hist.size(0), visits - drugs_hist.size(1), drugs_hist.size(2)
            )
            drugs_hist = torch.cat([drugs_hist, pad], dim=1)
        drugs_hist = drugs_hist[:, :visits]
        mapped = self._hist_to_label[drugs_hist.clamp(min=0)]
        multihot = torch.zeros(
            *drugs_hist.shape[:2], self.label_size + 1, device=drugs_hist.device
        )
        multihot.scatter_(2, mapped, 1.0)
        return multihot[..., : self.label_size]

    def _drug_embeddings(self) -> torch.Tensor:
        """Base drug embeddings ``E_m``, plus the graph embeddings if used."""
        emb = self.drug_emb
        if not self.use_graph:
            return emb
        out = emb
        for layers, adj, weight in (
            (self.gcn_ehr, self.ehr_adj, 1.0),
            (self.gcn_ddi, self.ddi_adj_norm, 0.5),
        ):
            h = F.relu(adj @ layers[0](emb))
            out = out + weight * (adj @ layers[1](h))
        return out

    def _responsive_drugs(
        self, history: torch.Tensor, prev_len: torch.Tensor
    ) -> torch.Tensor:
        """The patient's weighted drug embeddings ``E'_m`` (Eq. 5-11).

        Args:
            history: multi-hot drugs per previous visit, [batch, visits, M],
                zero after each patient's ``prev_len`` previous visits.
            prev_len: number of previous visits per patient, [batch].

        Returns:
            Tensor of shape [batch, M, dim].
        """
        visits = history.size(1)
        seen_before = torch.cumsum(history, dim=1) - history > 0
        reused = history * seen_before  # Eq. 5
        new = history - reused  # Eq. 6
        emb = self._drug_embeddings()
        h_reused = self.ptl_reused(reused @ emb, prev_len)  # Eq. 7
        h_new = self.ptl_new(new @ emb, prev_len)  # Eq. 8

        # Eq. 9: time-decayed past use; lag 1 is the visit before the prediction
        steps = torch.arange(visits, device=history.device).unsqueeze(0)
        lag = (prev_len.unsqueeze(1) - steps).clamp(min=1).float()
        decay = self.decay_scale * torch.exp(-F.softplus(self.decay_rate) * lag)
        used = (history * decay.unsqueeze(-1)).sum(dim=1)  # [batch, M]
        unused = 1.0 - (history.sum(dim=1) > 0).float()  # Eq. 10

        # Eq. 11
        scale = math.sqrt(self.embedding_dim)
        q_reused = torch.softmax(
            self.w_reused(torch.einsum("md,bkd->bmk", emb, h_reused)) / scale, dim=1
        )
        q_new = torch.softmax(
            self.w_new(torch.einsum("md,bkd->bmk", emb, h_new)) / scale, dim=1
        )
        weight = used.unsqueeze(-1) * q_reused + unused.unsqueeze(-1) * q_new
        return emb.unsqueeze(0) * weight * self.label_size

    def _loss(self, prob: torch.Tensor, logit: torch.Tensor, y: torch.Tensor):
        """``alpha * BCE + (1 - alpha) * multi-label margin`` (Eq. 20-22)."""
        bce = F.binary_cross_entropy_with_logits(logit, y)
        diff = 1.0 - (prob.unsqueeze(2) - prob.unsqueeze(1))  # p_i - p_j
        pairs = y.unsqueeze(2) * (1.0 - y).unsqueeze(1)  # i positive, j negative
        margin = (F.relu(diff) * pairs).sum(dim=(1, 2)) / self.label_size
        return self.alpha * bce + (1.0 - self.alpha) * margin.mean()

    # --------------------------------------------------------------- forward
    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Recommends drugs for the last visit of each patient.

        Args:
            **kwargs: the batch: ``conditions``, ``procedures`` and
                ``drugs_hist`` of shape [batch, visits, codes], and, for the
                loss, the ``drugs`` label of shape [batch, num_drugs].

        Returns:
            ``logit`` and ``y_prob`` of shape [batch, num_drugs]; ``loss`` and
            ``y_true`` when the label is in the batch.
        """
        conditions = kwargs["conditions"].to(self.device)
        procedures = kwargs["procedures"].to(self.device)
        embedded = self.embedding_model(
            {"conditions": conditions, "procedures": procedures}
        )
        d = embedded["conditions"].sum(dim=2)  # Eq. 12
        p = embedded["procedures"].sum(dim=2)  # Eq. 13
        lengths = torch.maximum(
            self._visit_lengths(conditions), self._visit_lengths(procedures)
        ).clamp(min=1)
        patient = torch.cat(
            [self.ptl_d(d, lengths), self.ptl_p(p, lengths)], dim=1
        )  # Eq. 14-15
        patient = self.dropout(patient.flatten(1))

        # previous visits: all but the last one, whose drugs are the target
        prev_len = lengths - 1
        history = self._drug_history(
            kwargs["drugs_hist"].to(self.device), d.size(1)
        )
        valid = torch.arange(history.size(1), device=self.device) < prev_len.unsqueeze(1)
        history = history * valid.unsqueeze(-1)
        drugs = self._responsive_drugs(history, prev_len)

        o_direct = self.ff_direct(patient)  # Eq. 16
        query = self.ff_query(patient)
        o_sim = F.cosine_similarity(query.unsqueeze(1), drugs, dim=-1)  # Eq. 17-18
        logit = self.w_direct * o_direct + self.w_sim * o_sim  # Eq. 19
        y_prob = torch.sigmoid(logit)
        out = {"logit": logit, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            out["loss"] = self._loss(y_prob, logit, y_true)
            out["y_true"] = y_true
        return out


def _normalize(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalizes ``adj + I``, as in Kipf and Welling's GCN."""
    adj = adj + torch.eye(adj.size(0))
    return adj / adj.sum(dim=1, keepdim=True)
