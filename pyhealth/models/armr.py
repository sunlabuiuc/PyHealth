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

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.selective_ssm import SelectiveSSM
from pyhealth.models.utils import (
    ddi_adjacency,
    ehr_cooccurrence_adjacency,
    gcn_normalize_adjacency,
    history_to_label_index,
    save_ddi_adj,
)


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
    every drug's embedding with the reused-drug attention if the patient took
    it before and the new-drug attention otherwise. The recommendation fuses a
    direct score from the patient representation with the cosine similarity
    between the patient and each weighted drug embedding (Eq. 19), plus a
    learned, time-decayed score for each drug the patient took before. The
    loss is
    ``alpha * BCE + (1 - alpha) * multi-label margin`` (Eq. 22, alpha 0.7).

    Choices where the paper is open:

    - Eq. 9 uses one learned decay rate and scale, ``alpha * exp(-beta *
      lag)`` summed over the visits in which drug ``j`` was given, where
      ``lag`` counts visits back from the prediction.
    - In Eq. 11 that decayed use multiplies the drug's whole embedding, and
      the cosine similarity of Eq. 17-18 cannot see a per-drug scale, so the
      decay would have no effect on the output. Here Eq. 11 uses the
      indicator of past use instead, and the decayed use is added to the
      drug's score (Eq. 19) as a recency term.
    - Eq. 11's attention maps the drug-history attention to a per-feature
      weight (``W_o`` and ``W_n`` are ``2N -> dim``), so the weighting changes
      the direction of a drug's embedding, which cosine similarity sees. The
      reused-drug softmax is taken over the drugs the patient took before
      and the new-drug softmax over the drugs never taken (the paper does not
      give the axis; the authors' code restricts each branch to its set).
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
        dataset: the dataset whose processors give the vocabularies. A
            split from :func:`~pyhealth.datasets.split_by_patient` works, as
            does the full dataset.
        embedding_dim: code and visit embedding size. Default is 64.
        n_recent: number of recent visits ``N`` handled by the feedforward
            branch. Default is 3.
        alpha: weight of the BCE term in the loss (Eq. 22). Default is 0.7.
        dropout: dropout on the patient representation. Default is 0.3.
        use_graph: add the co-occurrence and DDI graph embeddings to the drug
            embeddings, as the authors' code does. Default is True.
        graph_dataset: the samples the drug co-occurrence graph is built
            from; pass the training split, so that validation and test
            labels stay out of the model. Required when ``use_graph`` is
            True.

    Raises:
        ValueError: if ``use_graph`` is True and ``graph_dataset`` is None.

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
        >>> model = ARMR(dataset, embedding_dim=16, use_graph=False)  # doctest: +SKIP
        >>> # with the graphs (the default), pass the training split:
        >>> # ARMR(dataset, graph_dataset=train_split)
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)  # doctest: +SKIP
        >>> sorted(out)  # doctest: +SKIP
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
        graph_dataset: SampleDataset | None = None,
    ):
        if use_graph and graph_dataset is None:
            raise ValueError(
                "ARMR(use_graph=True) builds a drug co-occurrence graph from "
                "drug labels: pass the training split as graph_dataset (for "
                "example ARMR(dataset, graph_dataset=train)), or set "
                "use_graph=False."
            )
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

        ddi_adj = self._ddi_adj(label_vocab)
        if use_graph:
            ehr_adj = self._ehr_adj(label_vocab, graph_dataset)
            self.register_buffer("ehr_adj", gcn_normalize_adjacency(ehr_adj))
            self.register_buffer("ddi_adj_norm", gcn_normalize_adjacency(ddi_adj))
            self.gcn_ehr = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
            self.gcn_ddi = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        # Trainer.evaluate passes ddi_adj_matrix to the "ddi" metric; the file
        # is the fallback for callers of the metric that don't pass a matrix,
        # as for GAMENet, SafeDrug, MICRON and MoleRec
        self.ddi_adj_matrix = save_ddi_adj(
            ddi_adj, os.path.join(CACHE_PATH, "ddi_adj.npy")
        )

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
        return history_to_label_index(hist_vocab, label_vocab)

    def _ehr_adj(self, label_vocab: dict, samples) -> torch.Tensor:
        """Co-occurrence graph of the drugs given together in a visit.

        Args:
            label_vocab: drug code to label index.
            samples: the samples to read the drug labels from (the training
                split).
        """
        drug_sets = (sample[self.label_key] for sample in samples)
        return ehr_cooccurrence_adjacency(drug_sets, label_vocab)

    def _ddi_adj(self, label_vocab: dict) -> torch.Tensor:
        """Drug-drug interaction graph at ATC level 3 (as GAMENet and SafeDrug)."""
        return ddi_adjacency(label_vocab)

    # ---------------------------------------------------------------- pieces
    def _visit_lengths(self, codes: torch.Tensor) -> torch.Tensor:
        """Number of visits per patient: the last visit with a code, plus one.

        An empty visit is all padding, like the padded visits after the end,
        so counting non-empty visits would undercount a patient with an
        empty visit in the middle; the position of the last non-empty visit
        does not.
        """
        nonempty = (codes > 0).any(dim=-1)  # [batch, visits]
        position = torch.arange(1, codes.size(1) + 1, device=codes.device)
        return (nonempty * position).amax(dim=1)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The patient's weighted drug embeddings ``E'_m`` (Eq. 5-11).

        Args:
            history: multi-hot drugs per previous visit, [batch, visits, M],
                zero after each patient's ``prev_len`` previous visits.
            prev_len: number of previous visits per patient, [batch].

        Returns:
            The weighted drug embeddings, [batch, M, dim], and the time-decayed
            past use of each drug (Eq. 9), [batch, M], which is zero for drugs
            the patient never took.
        """
        visits = history.size(1)
        reused, new = _split_reused_new(history)  # Eq. 5-6
        emb = self._drug_embeddings()
        h_reused = self.ptl_reused(reused @ emb, prev_len)  # Eq. 7
        h_new = self.ptl_new(new @ emb, prev_len)  # Eq. 8

        # Eq. 9: time-decayed past use; lag 1 is the visit before the prediction
        steps = torch.arange(visits, device=history.device).unsqueeze(0)
        lag = (prev_len.unsqueeze(1) - steps).clamp(min=1).float()
        decay = self.decay_scale * torch.exp(-F.softplus(self.decay_rate) * lag)
        used = (history * decay.unsqueeze(-1)).sum(dim=1)  # [batch, M]
        taken = (history.sum(dim=1) > 0).float()
        unused = 1.0 - taken  # Eq. 10

        # Eq. 11: reused drugs attend over the drugs taken before, new drugs
        # over the drugs never taken; the decay goes into the score (forward)
        # instead, since the cosine similarity ignores a per-drug scale
        q_reused = self._set_attention(emb, h_reused, self.w_reused, taken)
        q_new = self._set_attention(emb, h_new, self.w_new, unused)
        return emb.unsqueeze(0) * (q_reused + q_new), used

    def _set_attention(
        self, emb: torch.Tensor, h: torch.Tensor, proj: nn.Linear, member: torch.Tensor
    ) -> torch.Tensor:
        """Eq. 11 attention ``q``, normalized over one set of drugs.

        Args:
            emb: drug embeddings, [M, dim].
            h: the set's time-sensitive representation, [batch, 2N, dim].
            proj: ``W_o`` or ``W_n``, mapping ``2N -> dim``.
            member: 1 for the drugs in the set, else 0, [batch, M].

        Returns:
            Per-feature weights, [batch, M, dim]: a softmax over the drugs of
            the set (each feature sums to 1 over them), zero outside the set
            and all zero when the set is empty.
        """
        logits = proj(torch.einsum("md,bkd->bmk", emb, h)) / math.sqrt(
            self.embedding_dim
        )
        inside = member.unsqueeze(-1) > 0
        logits = logits.masked_fill(~inside, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=1) * inside

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
        drugs, recency = self._responsive_drugs(history, prev_len)

        o_direct = self.ff_direct(patient)  # Eq. 16
        query = self.ff_query(patient)
        o_sim = F.cosine_similarity(query.unsqueeze(1), drugs, dim=-1)  # Eq. 17-18
        logit = self.w_direct * o_direct + self.w_sim * o_sim + recency  # Eq. 19
        y_prob = torch.sigmoid(logit)
        out = {"logit": logit, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            out["loss"] = self._loss(y_prob, logit, y_true)
            out["y_true"] = y_true
        return out


def _split_reused_new(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits multi-hot drugs per visit into reused and new drugs (Eq. 5-6).

    A drug is reused at a visit if the patient took it at an earlier visit,
    and new otherwise.

    Args:
        history: multi-hot drugs per visit, [batch, visits, M].

    Returns:
        ``(reused, new)``, each of the same shape as ``history``.
    """
    seen_before = torch.cumsum(history, dim=1) - history > 0
    reused = history * seen_before
    return reused, history - reused
