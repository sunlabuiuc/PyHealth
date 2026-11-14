"""SHy: Self-Explaining Hypergraph Neural Network for diagnosis prediction.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: Leisheng Yu, Yanxiao Cai, Minxing Zhang, Xia Hu. "Self-Explaining
    Hypergraph Neural Networks for Diagnosis Prediction." CHIL 2025 (PMLR 287).
Paper link: https://arxiv.org/abs/2502.10689

SHy models each patient as a hypergraph (diagnosis codes are nodes, visits are
hyperedges), refines code embeddings by hypergraph message passing, adds
likely-missing diagnoses to the record, and extracts K "temporal phenotypes"
(sub-hypergraphs) that it uses to predict the next visit's diagnoses. The
phenotypes and their weights are the model's explanation.

This is an independent pure-PyTorch implementation for PyHealth. It does not
need DHG, torch_scatter, torch_geometric or Pyro: hypergraphs are stored as
lists of (patient, visit, code) incidences and aggregated with ``index_add_``.
"""

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import NestedSequenceProcessor

AncestorFn = Callable[[str], list[str]]


def icd9_ancestor_fn() -> AncestorFn:
    """Returns a function mapping a dot-less ICD-9-CM code to its ancestors.

    Uses :class:`pyhealth.medcode.ICD9CM` (downloaded on first use). Ancestors
    are ordered from the closest (e.g. the 3-digit category) to the farthest
    (the chapter). The artificial root ``001-999.99`` above the numeric
    chapters is left out, so the farthest ancestor is always a chapter
    (``V01-V91.99`` and ``E000-E999.9`` for V and E codes). Unknown codes have
    no ancestors.

    Returns:
        A function ``code -> list of ancestor codes``.

    Examples:
        >>> fn = icd9_ancestor_fn()  # doctest: +SKIP
        >>> fn("4280")  # doctest: +SKIP
        ['428', '420-429.99', '390-459.99']
    """
    from pyhealth.medcode import InnerMap
    from pyhealth.medcode.codes.icd9cm import ICD9CM

    icd9 = InnerMap.load("ICD9CM")

    def ancestors(code: str) -> list[str]:
        std = ICD9CM.standardize(code)
        if std not in icd9:
            return []
        return [a for a in icd9.get_ancestors(std) if a != "001-999.99"]

    return ancestors


def build_code_levels(
    codes: Sequence[str],
    ancestor_fn: AncestorFn,
    num_levels: int,
) -> tuple[torch.Tensor, list[int]]:
    """Builds the per-level index matrix used by the hierarchical embedding.

    Each code's path is its ancestors from the top down, followed by the code
    itself. Paths shorter than ``num_levels`` repeat their last entry (as in
    CGL); longer paths keep the ``num_levels - 1`` top levels plus the code,
    so with ICD-9-CM ancestors from :func:`icd9_ancestor_fn` and ``num_levels
    = 4`` the levels are chapter, block, 3-digit category and code (a 5-digit
    code drops its 4-digit parent). Index 0 of every level is reserved for
    "no code".

    Args:
        codes: The code of each vocabulary index. ``<pad>`` gets index 0 at
            every level; ``<unk>`` gets index 0 at every level except the last.
        ancestor_fn: Maps a code to its ancestors, closest first.
        num_levels: Number of hierarchy levels L.

    Returns:
        A ``(len(codes), num_levels)`` long tensor of level indices and the
        number of entries of each level (including the reserved index 0).

    Examples:
        >>> parents = {"a1": ["a"], "a2": ["a"], "b1": ["b"]}
        >>> levels, sizes = build_code_levels(
        ...     ["<pad>", "a1", "a2", "b1"], lambda c: parents.get(c, []), 2
        ... )
        >>> levels.tolist(), sizes
        ([[0, 0], [1, 1], [1, 2], [2, 3]], [3, 4])
    """
    vocabs: list[dict[str, int]] = [{} for _ in range(num_levels)]
    rows = []
    for code in codes:
        if code == "<pad>":
            rows.append([0] * num_levels)
            continue
        if code == "<unk>":
            path = [code] * num_levels
            path[: num_levels - 1] = [None] * (num_levels - 1)
        else:
            path = list(reversed(ancestor_fn(code))) + [code]
            if len(path) > num_levels:
                path = path[: num_levels - 1] + [code]
            while len(path) < num_levels:
                path.append(path[-1])
        row = []
        for level, name in enumerate(path):
            if name is None:
                row.append(0)
                continue
            vocab = vocabs[level]
            if name not in vocab:
                vocab[name] = len(vocab) + 1
            row.append(vocab[name])
        rows.append(row)
    sizes = [len(vocab) + 1 for vocab in vocabs]
    return torch.tensor(rows, dtype=torch.long), sizes


class HierarchicalCodeEmbedding(nn.Module):
    """Ontology-aware code embedding (paper Eq. 1, from CGL).

    The embedding of a code is the concatenation of one learned embedding per
    hierarchy level (the code and its ancestors). With ``code_levels=None`` it
    falls back to a flat embedding table of the same total width, which is
    the "flat embedding" ablation.

    Args:
        num_codes: Vocabulary size N.
        dim_per_level: Embedding size of one level.
        code_levels: ``(N, L)`` level indices from :func:`build_code_levels`,
            or None for a flat embedding.
        level_sizes: Number of entries of each level (required with
            ``code_levels``).
        num_levels: L, used for the width of the flat embedding.

    Examples:
        >>> emb = HierarchicalCodeEmbedding(4, 8, None, None, num_levels=2)
        >>> emb().shape
        torch.Size([4, 16])
    """

    def __init__(
        self,
        num_codes: int,
        dim_per_level: int,
        code_levels: torch.Tensor | None,
        level_sizes: list[int] | None,
        num_levels: int,
    ) -> None:
        super().__init__()
        self.output_dim = dim_per_level * num_levels
        self.hierarchical = code_levels is not None
        if self.hierarchical:
            if level_sizes is None:
                raise ValueError("level_sizes is required with code_levels.")
            self.register_buffer("code_levels", code_levels, persistent=False)
            self.level_embeddings = nn.ModuleList(
                nn.Embedding(size, dim_per_level) for size in level_sizes
            )
        else:
            self.flat = nn.Embedding(num_codes, self.output_dim)

    def forward(self) -> torch.Tensor:
        """Returns the embedding of every code, shape ``(N, output_dim)``."""
        if not self.hierarchical:
            return self.flat.weight
        return torch.cat(
            [
                emb(self.code_levels[:, level])
                for level, emb in enumerate(self.level_embeddings)
            ],
            dim=-1,
        )


class Incidences:
    """Batched patient hypergraphs as a flat list of (visit, code) incidences.

    Hyperedges (visits) of all patients are numbered ``b * V + v``. Codes that
    appear in a patient's record are "local nodes", numbered per patient, so
    the same code gets a different embedding in each patient after message
    passing. Codes a patient never had keep a shared, patient-independent
    embedding.

    Args:
        codes: ``(B, V, C)`` code indices with 0 as padding.
        num_codes: Vocabulary size N.

    Examples:
        >>> inc = Incidences(torch.tensor([[[2, 3], [3, 0]]]), num_codes=4)
        >>> inc.edge.tolist(), inc.code.tolist(), inc.num_local
        ([0, 0, 1], [2, 3, 3], 2)
    """

    def __init__(self, codes: torch.Tensor, num_codes: int) -> None:
        batch, visits, _ = codes.shape
        self.batch_size = batch
        self.num_visits = visits
        self.num_codes = num_codes
        b, v, c = (codes > 0).nonzero(as_tuple=True)
        n = codes[b, v, c]
        key = torch.unique((b * visits + v) * num_codes + n)
        self.edge = key // num_codes
        self.code = key % num_codes
        self.batch = self.edge // visits
        self.visit_mask = (codes > 0).any(-1)
        local_key = self.batch * num_codes + self.code
        self.local_keys, self.node = torch.unique(local_key, return_inverse=True)
        self.num_local = self.local_keys.numel()
        self.local_code = self.local_keys % num_codes
        self.local_batch = self.local_keys // num_codes

    @property
    def num_edges(self) -> int:
        """Number of hyperedge slots, ``B * V`` (padding visits included)."""
        return self.batch_size * self.num_visits

    def edge_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Averages per-incidence values over each hyperedge.

        Args:
            values: ``(num_incidences, D)``.

        Returns:
            ``(B * V, D)``; empty (padding) visits are zero.
        """
        out = values.new_zeros(self.num_edges, values.shape[-1])
        out.index_add_(0, self.edge, values)
        count = values.new_zeros(self.num_edges).index_add_(
            0, self.edge, values.new_ones(self.edge.numel())
        )
        return out / count.clamp(min=1).unsqueeze(-1)

    def lookup_local(self, batch: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Local node id of each (patient, code) pair, or -1 if not local.

        Args:
            batch: Patient index of each pair.
            code: Code index of each pair.

        Returns:
            A long tensor of local node ids, -1 where the patient never had
            the code.
        """
        key = batch * self.num_codes + code
        pos = torch.searchsorted(self.local_keys, key).clamp(max=self.num_local - 1)
        found = self.local_keys[pos] == key
        return torch.where(found, pos, torch.full_like(pos, -1))


class UniGINConv(nn.Module):
    """UniGIN hypergraph convolution (Huang and Yang, 2021), dense-free.

    Two-stage aggregation: each visit averages its codes, then each code sums
    the visits that contain it, ``x' = W((1 + eps) x + sum_e mean_{u in e} x_u)``.
    Codes outside the patient's record receive no messages, so their update
    ``W((1 + eps) x)`` is shared across patients and computed once.

    Args:
        in_dim: Input size.
        out_dim: Output size.

    Examples:
        >>> conv = UniGINConv(4, 6)
        >>> inc = Incidences(torch.tensor([[[1, 2]]]), num_codes=3)
        >>> local, shared = conv(torch.randn(2, 4), torch.randn(3, 4), inc)
        >>> local.shape, shared.shape
        (torch.Size([2, 6]), torch.Size([3, 6]))
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.eps = nn.Parameter(torch.tensor([0.1]))

    def forward(
        self, x_local: torch.Tensor, x_shared: torch.Tensor, inc: Incidences
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates local (per-patient) and shared code embeddings.

        Args:
            x_local: ``(num_local, in_dim)`` embeddings of codes in records.
            x_shared: ``(N, in_dim)`` embeddings of all codes.
            inc: The batch's incidences.

        Returns:
            The updated ``(x_local, x_shared)``.
        """
        x_edge = inc.edge_mean(x_local[inc.node])
        messages = torch.zeros_like(x_local).index_add_(0, inc.node, x_edge[inc.edge])
        x_local = self.linear((1 + self.eps) * x_local + messages)
        x_shared = self.linear((1 + self.eps) * x_shared)
        return x_local, x_shared


class PhenotypeExtractor(nn.Module):
    """One temporal-phenotype channel: false-negative recovery plus masking.

    1. False-negative recovery (paper Eqs. 4-5): scores every (code, visit)
       pair not in the record by a multi-head weighted cosine similarity
       between the code and the visit embeddings, and adds the top
       ``add_ratio * |record|`` pairs. As in the authors' code, the selection
       is not differentiable (``cos_weight`` receives no gradient). With
       ``random_links=True`` the same number of pairs is drawn at random
       instead, a control for the recovery ablation.
    2. Masking (paper Eqs. 6-8): an MLP scores each incidence of the enriched
       hypergraph and a straight-through relaxed Bernoulli (Gumbel) sample
       keeps or drops it. Masks are sampled in eval mode too (the authors
       evaluate with sampled masks). With ``stochastic_eval=False`` the eval
       mask is the deterministic ``prob > 0.5``; the learned keep
       probabilities are small (mean 0.07 on MIMIC-III), so that mask keeps
       almost nothing and scores far lower.

    Args:
        emb_dim: Code embedding size after message passing.
        num_cos_heads: Number of weighted-cosine heads ``n_c``.
        add_ratio: Ratio of added links to existing links (0 disables).
        temperature: Gumbel-Softmax temperature.
        mlp_dim: Hidden size of the masking MLP.
        random_links: Add random instead of similarity-ranked links.
        stochastic_eval: Sample masks in eval mode (default, as the authors
            do). False uses the deterministic ``prob > 0.5`` mask.

    Examples:
        >>> ext = PhenotypeExtractor(emb_dim=8, num_cos_heads=2, add_ratio=0.5)
        >>> ext.add_ratio
        0.5
    """

    def __init__(
        self,
        emb_dim: int,
        num_cos_heads: int = 10,
        add_ratio: float = 0.2,
        temperature: float = 1.0,
        mlp_dim: int = 256,
        random_links: bool = False,
        stochastic_eval: bool = True,
    ) -> None:
        super().__init__()
        if add_ratio < 0:
            raise ValueError(f"add_ratio must be non-negative, got {add_ratio}.")
        self.add_ratio = add_ratio
        self.temperature = temperature
        self.random_links = random_links
        self.stochastic_eval = stochastic_eval
        self.cos_weight = nn.Parameter(torch.randn(num_cos_heads, emb_dim))
        self.mask_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, 1)
        )

    @torch.no_grad()
    def recover(
        self,
        x_local: torch.Tensor,
        x_shared: torch.Tensor,
        x_edge: torch.Tensor,
        inc: Incidences,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selects likely-missing (visit, code) pairs.

        Args:
            x_local: ``(num_local, D)`` local code embeddings.
            x_shared: ``(N, D)`` shared code embeddings.
            x_edge: ``(B * V, D)`` visit embeddings.
            inc: The batch's incidences.

        Returns:
            ``(edge, code)`` index tensors of the added incidences.
        """
        batch, visits, n = inc.batch_size, inc.num_visits, inc.num_codes
        device = x_shared.device
        n_add = (inc.batch.bincount(minlength=batch).float() * self.add_ratio).long()
        k_max = int(n_add.max().item()) if batch > 0 else 0
        empty = torch.zeros(0, dtype=torch.long, device=device)
        if k_max == 0:
            return empty, empty

        if self.random_links:
            score = torch.rand(batch, n, visits, device=device)
        else:
            w = self.cos_weight.unsqueeze(1)
            edge_m = F.normalize(x_edge.unsqueeze(0) * w, dim=-1)
            shared_m = F.normalize(x_shared.unsqueeze(0) * w, dim=-1)
            local_m = F.normalize(x_local.unsqueeze(0) * w, dim=-1)
            shared_s = torch.einsum("cnd,ced->ne", shared_m, edge_m) / w.shape[0]
            score = shared_s.view(n, batch, visits).permute(1, 0, 2).contiguous()
            local_s = torch.einsum("cpd,ced->pe", local_m, edge_m) / w.shape[0]
            local_s = local_s.view(-1, batch, visits)
            rows = torch.arange(inc.num_local, device=device)
            score[inc.local_batch, inc.local_code] = local_s[rows, inc.local_batch]

        score[inc.batch, inc.code, inc.edge % visits] = -math.inf
        score[:, :2, :] = -math.inf  # <pad> and <unk> are never added
        score.masked_fill_(~inc.visit_mask.unsqueeze(1), -math.inf)
        k_max = min(k_max, n * visits)
        value, flat = score.view(batch, -1).topk(k_max, dim=-1)
        keep = (torch.arange(k_max, device=device) < n_add.unsqueeze(1)) & (
            value > -math.inf
        )
        b_idx = torch.arange(batch, device=device).unsqueeze(1).expand_as(flat)
        code = flat // visits
        edge = b_idx * visits + flat % visits
        return edge[keep], code[keep]

    def sample_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through relaxed Bernoulli sample of incidence masks.

        Args:
            logits: Per-incidence logits.

        Returns:
            Hard 0/1 masks with the gradient of the relaxed sample.
        """
        if not self.training and not self.stochastic_eval:
            return (logits > 0).float()
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
        soft = torch.sigmoid((logits + noise) / self.temperature)
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach()

    def forward(
        self,
        x_local: torch.Tensor,
        x_shared: torch.Tensor,
        x_edge: torch.Tensor,
        inc: Incidences,
    ) -> dict[str, torch.Tensor]:
        """Extracts one phenotype.

        Args:
            x_local: ``(num_local, D)`` local code embeddings.
            x_shared: ``(N, D)`` shared code embeddings.
            x_edge: ``(B * V, D)`` visit embeddings.
            inc: The batch's incidences.

        Returns:
            A dict with, per incidence of the enriched hypergraph, ``edge``,
            ``code``, node embedding ``x``, keep probability ``prob``, sampled
            ``mask``, and ``added`` (True for recovered links).
        """
        add_edge, add_code = self.recover(x_local, x_shared, x_edge, inc)
        add_local = inc.lookup_local(add_edge // inc.num_visits, add_code)
        add_x = torch.where(
            (add_local >= 0).unsqueeze(-1),
            x_local[add_local.clamp(min=0)],
            x_shared[add_code],
        )
        edge = torch.cat([inc.edge, add_edge])
        code = torch.cat([inc.code, add_code])
        x = torch.cat([x_local[inc.node], add_x])
        logits = self.mask_mlp(torch.cat([x, x_edge[edge]], dim=-1)).squeeze(-1)
        added = torch.cat(
            [
                torch.zeros_like(inc.edge, dtype=torch.bool),
                torch.ones_like(add_edge, dtype=torch.bool),
            ]
        )
        return {
            "edge": edge,
            "code": code,
            "x": x,
            "prob": torch.sigmoid(logits),
            "mask": self.sample_mask(logits),
            "added": added,
        }


class PhenotypeEncoder(nn.Module):
    """Embeds a temporal phenotype with a GRU and visit attention (Eqs. 10-12).

    Visit embeddings are the masked sums of their codes' embeddings. With
    ``use_time_gaps`` the log of (1 + days since the previous visit) is
    appended to each visit embedding, the time-gap extension (not in the
    paper).

    Args:
        emb_dim: Code embedding size.
        hidden_dim: GRU hidden size (phenotype embedding size).
        use_time_gaps: Feed visit time gaps to the GRU.

    Examples:
        >>> enc = PhenotypeEncoder(emb_dim=8, hidden_dim=4)
        >>> visits = torch.randn(2, 3, 8)
        >>> mask = torch.tensor([[1, 1, 0], [1, 0, 0]]).bool()
        >>> enc(visits, mask).shape
        torch.Size([2, 4])
    """

    def __init__(
        self, emb_dim: int, hidden_dim: int, use_time_gaps: bool = False
    ) -> None:
        super().__init__()
        self.use_time_gaps = use_time_gaps
        self.gru = nn.GRU(emb_dim + int(use_time_gaps), hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        visit_emb: torch.Tensor,
        visit_mask: torch.Tensor,
        time_gaps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encodes a batch of phenotypes.

        Args:
            visit_emb: ``(B, V, D)`` phenotype visit embeddings.
            visit_mask: ``(B, V)`` True for real visits (padding at the end).
            time_gaps: ``(B, V)`` days since the previous visit, required with
                ``use_time_gaps``.

        Returns:
            ``(B, hidden_dim)`` phenotype embeddings.
        """
        if self.use_time_gaps:
            if time_gaps is None:
                raise ValueError("use_time_gaps=True needs the visit time gaps.")
            gaps = torch.log1p(time_gaps.clamp(min=0).float()).unsqueeze(-1)
            visit_emb = torch.cat([visit_emb, gaps], dim=-1)
        hidden, _ = self.gru(visit_emb)
        score = self.attention(hidden).squeeze(-1)
        # A record with no real visit attends uniformly instead of producing NaN.
        visit_mask = visit_mask | ~visit_mask.any(-1, keepdim=True)
        score = score.masked_fill(~visit_mask, -math.inf)
        alpha = torch.softmax(score, dim=-1)
        return (alpha.unsqueeze(-1) * hidden).sum(dim=1)


class FidelityDecoder(nn.Module):
    """GRU decoder reconstructing the patient hypergraph (paper App. C).

    The concatenated phenotype embeddings initialise a GRU that, with teacher
    forcing, predicts each visit's multi-hot code vector from the previous
    visit's codes (embedded with the input code embeddings).

    Args:
        latent_dim: Size of the concatenated phenotype embeddings.
        emb_dim: Code embedding size (also the GRU size).
        num_codes: Vocabulary size N.

    Examples:
        >>> dec = FidelityDecoder(latent_dim=6, emb_dim=4, num_codes=5)
        >>> dec(torch.randn(2, 6), torch.randn(2, 3, 4)).shape
        torch.Size([2, 3, 5])
    """

    def __init__(self, latent_dim: int, emb_dim: int, num_codes: int) -> None:
        super().__init__()
        self.to_context = nn.Linear(latent_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.out = nn.Linear(emb_dim, num_codes)

    def forward(
        self, latent: torch.Tensor, prev_visit_emb: torch.Tensor
    ) -> torch.Tensor:
        """Reconstructs visits.

        Args:
            latent: ``(B, latent_dim)`` concatenated phenotype embeddings.
            prev_visit_emb: ``(B, V, emb_dim)`` sum of the code embeddings of
                the previous visit (zeros for the first visit).

        Returns:
            ``(B, V, N)`` probabilities of each code at each visit.
        """
        hidden = self.to_context(latent).unsqueeze(0)
        output, _ = self.gru(F.relu(prev_visit_emb), hidden)
        return torch.sigmoid(self.out(output))


class SHy(BaseModel):
    """Self-Explaining Hypergraph Neural Network (Yu et al., CHIL 2025).

    Paper: https://arxiv.org/abs/2502.10689

    Pipeline, for a batch of patients:

    1. Hierarchical ICD-9 code embeddings (:class:`HierarchicalCodeEmbedding`).
    2. ``num_layers`` UniGIN layers over each patient hypergraph
       (:class:`UniGINConv`).
    3. ``num_phenotypes`` channels, each recovering likely-missing diagnoses
       and sampling a sub-hypergraph, a temporal phenotype
       (:class:`PhenotypeExtractor`).
    4. A GRU with attention embeds each phenotype (:class:`PhenotypeEncoder`).
    5. Self-attention over the phenotypes gives their weights ``alpha``; the
       prediction is the alpha-weighted sum of per-phenotype softmax
       predictions (paper Eqs. 13-15).
    6. Loss ``L_pred + eps * L_fidelity + eta * L_distinct - omega * L_alpha``
       (paper Eq. 16), where the fidelity term reconstructs the input with
       :class:`FidelityDecoder`.

    Deliberate differences from the authors' code:

    * Self-attention runs over the K phenotypes of each patient
      (``batch_first``); the released code applies ``nn.MultiheadAttention``
      to a ``(batch, K, dim)`` tensor without ``batch_first`` and so attends
      across patients.
    * ``omega`` defaults to 0, the value the authors' released training
      command passes; their README table lists 0.04. The two score within
      seed noise of each other on MIMIC-III; 0.04 spreads the phenotype
      weights more evenly.
    * ``stochastic_eval=False`` offers deterministic eval masks
      (``prob > 0.5``), which the released code has no equivalent of. They
      keep far fewer (visit, code) pairs than sampled masks and score much
      lower, so sampling stays the default.
    * ``add_ratio`` and ``temperature`` are shared by the K channels instead
      of being given per channel, and recovery never adds the ``<pad>`` /
      ``<unk>`` tokens.

    ``y_prob`` is a mixture of softmax outputs and sums to 1 per patient, as
    in the paper, so rank-based metrics (``recall@k``, ``ndcg@k``) are the
    right ones; metrics that threshold at 0.5 are not meaningful here.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` with one
            ``nested_sequence`` input of diagnosis codes and one multilabel
            output, e.g. from
            :class:`~pyhealth.tasks.NextVisitDiagnosisPredictionMIMIC3`.
        feature_key: The diagnosis input key. Default ``"conditions"``.
        time_gap_key: The visit-gap input key used when ``use_time_gaps``.
        dim_per_level: Embedding size of one ICD-9 level (paper d_c = 32).
        num_levels: Number of hierarchy levels L (paper 4).
        use_hierarchy: Use hierarchical embeddings; False gives a flat table.
        code_ancestors: Maps a code to its ancestors (closest first). Default
            None uses :func:`icd9_ancestor_fn`, which downloads the ICD-9-CM
            table on first use. Pass a dict or function for other
            vocabularies or for tests.
        hgnn_dim: Hidden size of each UniGIN head.
        num_heads: Number of UniGIN heads (hidden layers are
            ``hgnn_dim * num_heads`` wide).
        num_layers: Number of UniGIN layers Z (paper 2). 0 replaces message
            passing with a linear layer.
        emb_dim: Code embedding size after message passing.
        num_phenotypes: Number of temporal phenotypes K (paper 5).
        add_ratio: Ratio p of recovered links (paper code 0.2); 0 disables
            false-negative recovery.
        random_links: Add random links instead of similarity-ranked ones.
        temperature: Gumbel-Softmax temperature.
        num_cos_heads: Weighted-cosine heads n_c.
        hidden_dim: Phenotype embedding size.
        key_dim: Self-attention size over phenotypes.
        num_attention_heads: Self-attention heads (must divide ``key_dim``).
        dropout: Dropout after each hidden UniGIN layer.
        use_time_gaps: Feed visit time gaps to the phenotype GRU (extension).
        loss_weights: ``(eps, eta, omega)`` weights of the fidelity,
            distinctness and alpha losses. Default ``(0.003, 0.00025, 0.0)``,
            the authors' released command; their README lists ``omega=0.04``.
        stochastic_eval: Sample phenotype masks in eval mode (default, as in
            the authors' evaluation; seed torch for reproducible scores).
            False uses deterministic ``prob > 0.5`` masks.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "conditions": [["a", "b"], ["c"]],
        ...      "next_conditions": ["a", "c"]},
        ...     {"patient_id": "p1", "conditions": [["b"], ["a", "d"]],
        ...      "next_conditions": ["d"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"next_conditions": "multilabel"},
        ... )
        >>> model = SHy(dataset, code_ancestors={}, num_phenotypes=2,
        ...             hgnn_dim=8, num_heads=1, emb_dim=8, hidden_dim=8,
        ...             key_dim=8, num_attention_heads=2, dim_per_level=4)
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> out["y_prob"].shape, out["phenotype_weights"].shape
        (torch.Size([2, 3]), torch.Size([2, 2]))
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "conditions",
        time_gap_key: str = "visit_gaps",
        dim_per_level: int = 32,
        num_levels: int = 4,
        use_hierarchy: bool = True,
        code_ancestors: dict[str, list[str]] | AncestorFn | None = None,
        hgnn_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        emb_dim: int = 128,
        num_phenotypes: int = 5,
        add_ratio: float = 0.2,
        random_links: bool = False,
        temperature: float = 1.0,
        num_cos_heads: int = 10,
        hidden_dim: int = 128,
        key_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.001,
        use_time_gaps: bool = False,
        loss_weights: tuple[float, float, float] = (0.003, 0.00025, 0.0),
        stochastic_eval: bool = True,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("SHy expects exactly one (multilabel) output.")
        if feature_key not in self.feature_keys:
            raise ValueError(f"Input {feature_key!r} not in the dataset schema.")
        if use_time_gaps and time_gap_key not in self.feature_keys:
            raise ValueError(
                f"use_time_gaps=True needs the {time_gap_key!r} input; build the "
                "task with include_visit_gaps=True."
            )
        if num_phenotypes < 1:
            raise ValueError("num_phenotypes must be at least 1.")
        self.label_key = self.label_keys[0]
        self.feature_key = feature_key
        self.time_gap_key = time_gap_key
        self.num_phenotypes = num_phenotypes
        self.use_time_gaps = use_time_gaps
        self.loss_weights = tuple(loss_weights)
        if len(self.loss_weights) != 3:
            raise ValueError("loss_weights must be (epsilon, eta, omega).")

        processor = dataset.input_processors[feature_key]
        if not isinstance(processor, NestedSequenceProcessor):
            raise TypeError(f"Input {feature_key!r} must be a nested_sequence.")
        vocab = processor.code_vocab
        self.codes = [code for code, _ in sorted(vocab.items(), key=lambda x: x[1])]
        self.num_codes = len(self.codes)

        code_levels, level_sizes = None, None
        if use_hierarchy:
            if code_ancestors is None:
                ancestor_fn = icd9_ancestor_fn()
            elif callable(code_ancestors):
                ancestor_fn = code_ancestors
            else:
                ancestor_fn = lambda code: list(code_ancestors.get(code, []))
            code_levels, level_sizes = build_code_levels(
                self.codes, ancestor_fn, num_levels
            )
        self.code_embedding = HierarchicalCodeEmbedding(
            self.num_codes, dim_per_level, code_levels, level_sizes, num_levels
        )
        in_dim = self.code_embedding.output_dim

        dims = [in_dim] + [hgnn_dim * num_heads] * max(num_layers - 1, 0)
        self.convs = (
            nn.ModuleList(
                UniGINConv(d_in, d_out)
                for d_in, d_out in zip(dims, dims[1:] + [emb_dim])
            )
            if num_layers > 0
            else nn.ModuleList()
        )
        self.no_hgnn = nn.Linear(in_dim, emb_dim) if num_layers == 0 else None
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        self.extractors = nn.ModuleList(
            PhenotypeExtractor(
                emb_dim,
                num_cos_heads=num_cos_heads,
                add_ratio=add_ratio,
                temperature=temperature,
                random_links=random_links,
                stochastic_eval=stochastic_eval,
            )
            for _ in range(num_phenotypes)
        )
        self.encoder = PhenotypeEncoder(emb_dim, hidden_dim, use_time_gaps)
        self.decoder = FidelityDecoder(
            hidden_dim * num_phenotypes, in_dim, self.num_codes
        )

        output_size = self.label_processor.output_size()
        if num_phenotypes > 1:
            self.w_query = nn.Linear(hidden_dim, key_dim)
            self.w_key = nn.Linear(hidden_dim, key_dim)
            self.w_value = nn.Linear(hidden_dim, key_dim)
            self.self_attention = nn.MultiheadAttention(
                key_dim, num_attention_heads, batch_first=True
            )
            self.phenotype_attention = nn.Linear(key_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, output_size)

    def _encode_codes(
        self, inc: Incidences, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs message passing; returns local and shared code embeddings."""
        x_local, x_shared = x0[inc.local_code], x0
        if self.no_hgnn is not None:
            return self.act(self.no_hgnn(x_local)), self.act(self.no_hgnn(x_shared))
        for i, conv in enumerate(self.convs):
            x_local, x_shared = conv(x_local, x_shared, inc)
            if i < len(self.convs) - 1:
                x_local = self.dropout(self.act(x_local))
                x_shared = self.dropout(self.act(x_shared))
        return self.act(x_local), self.act(x_shared)

    def _fuse(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weights phenotypes by self-attention and mixes their predictions.

        Args:
            latent: ``(B, K, hidden_dim)`` phenotype embeddings.

        Returns:
            ``(B, num_labels)`` probabilities and ``(B, K)`` weights alpha.
        """
        separate = torch.softmax(self.classifier(latent), dim=-1)
        if self.num_phenotypes == 1:
            alpha = latent.new_ones(latent.shape[0], 1)
        else:
            attended, _ = self.self_attention(
                self.w_query(latent),
                self.w_key(latent),
                self.w_value(latent),
                need_weights=False,
            )
            alpha = torch.softmax(self.phenotype_attention(attended).squeeze(-1), -1)
        return (alpha.unsqueeze(-1) * separate).sum(dim=1), alpha

    def _distinctness(
        self, phenotypes: list[dict[str, torch.Tensor]], inc: Incidences
    ) -> torch.Tensor:
        """Distinctness loss (the paper's L_distinct): ``||I - M_j^T M_j||`` per visit.

        ``M_j`` is the (codes x K) mask matrix of visit j; the loss is averaged
        over each patient's visits, then over patients.
        """
        k = len(phenotypes)
        n = inc.num_codes
        keys = torch.cat([p["edge"] * n + p["code"] for p in phenotypes])
        which = torch.cat(
            [torch.full_like(p["edge"], i) for i, p in enumerate(phenotypes)]
        )
        masks = torch.cat([p["mask"] for p in phenotypes])
        unique, inverse = torch.unique(keys, return_inverse=True)
        m = masks.new_zeros(unique.numel(), k).index_put(
            (inverse, which), masks, accumulate=True
        )
        gram = masks.new_zeros(inc.num_edges, k, k).index_add_(
            0, unique // n, m.unsqueeze(-1) * m.unsqueeze(-2)
        )
        eye = torch.eye(k, device=gram.device)
        per_visit = torch.linalg.matrix_norm(eye - gram)
        visit_mask = inc.visit_mask.reshape(-1).float()
        per_patient = (per_visit * visit_mask).view(inc.batch_size, -1).sum(-1)
        per_patient = per_patient / inc.visit_mask.sum(-1).clamp(min=1)
        return per_patient.mean()

    def _phenotypes(self, **kwargs) -> dict[str, object]:
        """Shared forward computation up to the phenotype embeddings."""
        codes = kwargs[self.feature_key].to(self.device).long()
        if codes.dim() != 3:
            raise ValueError(
                f"Expected a (batch, visits, codes) input, got {tuple(codes.shape)}."
            )
        inc = Incidences(codes, self.num_codes)
        x0 = self.code_embedding()
        x_local, x_shared = self._encode_codes(inc, x0)
        x_edge = inc.edge_mean(x_local[inc.node])

        gaps = None
        if self.use_time_gaps:
            gaps = kwargs[self.time_gap_key].to(self.device).float()
            if gaps.dim() != 2 or gaps.shape[1] < inc.num_visits:
                raise ValueError(
                    f"{self.time_gap_key!r} must be (batch, visits), got {gaps.shape}."
                )
            gaps = gaps[:, : inc.num_visits]

        phenotypes, latent = [], []
        for extractor in self.extractors:
            p = extractor(x_local, x_shared, x_edge, inc)
            visit_emb = x_shared.new_zeros(inc.num_edges, x_shared.shape[-1])
            visit_emb.index_add_(0, p["edge"], p["mask"].unsqueeze(-1) * p["x"])
            visit_emb = visit_emb.view(inc.batch_size, inc.num_visits, -1)
            latent.append(self.encoder(visit_emb, inc.visit_mask, gaps))
            phenotypes.append(p)
        return {
            "inc": inc,
            "x0": x0,
            "phenotypes": phenotypes,
            "latent": torch.stack(latent, dim=1),
        }

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`:
                the ``feature_key`` tensor ``(B, V, C)`` of code indices, the
                label tensor ``(B, num_labels)`` and, with ``use_time_gaps``,
                the ``time_gap_key`` tensor ``(B, V)``.

        Returns:
            A dict with ``logit`` (log of ``y_prob``), ``y_prob`` (``(B,
            num_labels)`` next-visit code probabilities, summing to 1 per
            patient as in the paper), ``phenotype_weights`` (``(B, K)``) and,
            if labels are given, ``loss``, ``y_true`` and the four loss terms
            ``loss_pred``, ``loss_fidelity``, ``loss_distinct``,
            ``loss_alpha``.
        """
        state = self._phenotypes(**kwargs)
        inc: Incidences = state["inc"]
        latent: torch.Tensor = state["latent"]
        y_prob, alpha = self._fuse(latent)
        y_prob = y_prob.clamp(1e-8, 1 - 1e-7)
        results = {
            "logit": torch.log(y_prob),
            "y_prob": y_prob,
            "phenotype_weights": alpha,
        }
        if self.label_key not in kwargs:
            return results

        y_true = kwargs[self.label_key].to(self.device).float()
        loss_pred = F.binary_cross_entropy(y_prob, y_true)

        # Fidelity: reconstruct each visit from the previous one (teacher forcing).
        x0 = state["x0"]
        visit_sum = (
            x0.new_zeros(inc.num_edges, x0.shape[-1])
            .index_add_(0, inc.edge, x0[inc.code])
            .view(inc.batch_size, inc.num_visits, -1)
        )
        prev = torch.cat([torch.zeros_like(visit_sum[:, :1]), visit_sum[:, :-1]], 1)
        recon = self.decoder(latent.flatten(1), prev).clamp(1e-7, 1 - 1e-7)
        target = torch.zeros_like(recon).view(-1, self.num_codes)
        target[inc.edge, inc.code] = 1.0
        target = target.view_as(recon)
        bce = F.binary_cross_entropy(recon, target, reduction="none").mean(-1)
        visit_mask = inc.visit_mask.float()
        per_patient = (bce * visit_mask).sum(-1) / visit_mask.sum(-1).clamp(min=1)
        loss_fidelity = per_patient.mean()

        if self.num_phenotypes > 1:
            loss_distinct = self._distinctness(state["phenotypes"], inc)
            loss_alpha = (alpha.std(dim=1) - alpha.norm(dim=1)).mean()
        else:
            loss_distinct = loss_alpha = y_prob.new_zeros(())

        eps, eta, omega = self.loss_weights
        loss = (
            loss_pred + eps * loss_fidelity + eta * loss_distinct - omega * loss_alpha
        )
        results.update(
            loss=loss,
            y_true=y_true,
            loss_pred=loss_pred.detach(),
            loss_fidelity=loss_fidelity.detach(),
            loss_distinct=loss_distinct.detach(),
            loss_alpha=loss_alpha.detach(),
        )
        return results

    @torch.no_grad()
    def explain(self, **kwargs) -> list[dict[str, object]]:
        """Returns the temporal phenotypes and their weights for each patient.

        Args:
            **kwargs: A batch, as for :meth:`forward` (labels not needed).

        Returns:
            One dict per patient with ``weights`` (list of K floats) and
            ``phenotypes``: K lists of ``(visit_index, code, added)`` tuples
            kept by the masks, where ``added`` marks recovered diagnoses.

        Examples:
            >>> explanations = model.explain(**batch)  # doctest: +SKIP
            >>> explanations[0]["weights"]  # doctest: +SKIP
            [0.31, 0.12, 0.22, 0.2, 0.15]
        """
        was_training = self.training
        self.eval()
        try:
            state = self._phenotypes(**kwargs)
            _, alpha = self._fuse(state["latent"])
        finally:
            self.train(was_training)
        inc: Incidences = state["inc"]
        out: list[dict[str, object]] = [
            {"weights": alpha[b].tolist(), "phenotypes": []}
            for b in range(inc.batch_size)
        ]
        for p in state["phenotypes"]:
            per_patient: list[list[tuple[int, str, bool]]] = [
                [] for _ in range(inc.batch_size)
            ]
            kept = p["mask"] > 0.5
            for edge, code, added in zip(
                p["edge"][kept].tolist(),
                p["code"][kept].tolist(),
                p["added"][kept].tolist(),
            ):
                b, v = divmod(edge, inc.num_visits)
                per_patient[b].append((v, self.codes[code], added))
            for b in range(inc.batch_size):
                out[b]["phenotypes"].append(sorted(per_patient[b]))
        return out
