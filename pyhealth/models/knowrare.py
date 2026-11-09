"""KnowRare: knowledge-guided domain adaptation for rare ICU conditions.

Implements KnowRare from Zhu et al., "Bridging data gaps of rare conditions
in ICU: a multi-disease adaptation approach for clinical prediction" (npj
Digital Medicine 2025, https://arxiv.org/abs/2507.06432). KnowRare predicts
outcomes for ICU patients whose primary condition is rare, in four steps:

1. Build a condition knowledge graph (KG) with three relations between
   conditions: diagnosis co-occurrence, similarity of the patients' records,
   and shared drugs (:class:`ConditionKnowledgeGraph`, Eqs. 1-4).
2. Embed the KG with TuckER (:class:`TuckER`, Eq. 5) and, for each target
   rare condition, pick the ``k`` conditions whose embeddings are most
   cosine-similar (:func:`select_source_conditions`, Eq. 6).
3. Pre-train the encoder (an LSTM over the time series plus an MLP over the
   static variables) on all conditions by next-step prediction
   (:meth:`KnowRare.pretrain`, Eq. 8).
4. For each target, fine-tune a copy of the pre-trained model on the target
   and its source conditions with joint adversarial domain adaptation: a
   discriminator sees the latent representation and the predicted outcome
   and tries to tell the conditions apart, while the encoder learns to fool
   it (:meth:`KnowRare.set_domains`, Eqs. 9-12). Samples are weighted by the
   inverse prevalence of their condition.

This is a clean reimplementation from the paper. The authors' code
(https://github.com/JasonZuu/KnowRare) is under CC BY-NC-ND 4.0, which does
not allow derivative works, so no code was copied from it; it was read only
to confirm details the paper leaves open (layer sizes, the default ``k``).
Choices made here where the paper is silent or the code differs:

- The minimax game of Eq. 10 is trained with a gradient reversal layer: the
  discriminator minimizes its cross-entropy while the encoder and
  classifier receive its gradient multiplied by ``-adversarial_weight``.
  This is the simultaneous-update form of the same objective and lets the
  standard :class:`~pyhealth.trainer.Trainer` train the model with one
  optimizer. The authors' code alternates updates with two optimizers.
- The discriminator reads ``[h_T, y_prob]`` and classifies the ``k + 1``
  selected conditions (Eq. 9). The authors' code passes logits through a
  linear layer and classifies all conditions of the training set.
- The inverse-prevalence weights ``1 / p(v)`` multiply the per-sample loss,
  as the paper describes; the authors' code resamples batches instead.
- Next-step prediction runs the LSTM once over the whole series (its output
  at step ``t`` is the state after ``x_1..x_t``) rather than once per prefix.
- The paper's learning-rate warm-up and decay are left to the caller.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

__all__ = [
    "ConditionKnowledgeGraph",
    "KnowRare",
    "TuckER",
    "select_source_conditions",
]


# ------------------------------------------------------------ knowledge graph
def _keep_top(weights: np.ndarray, ratio: float) -> np.ndarray:
    """Zeroes all but the top ``ratio`` share of the off-diagonal weights."""
    weights = weights.copy()
    np.fill_diagonal(weights, 0.0)
    off = ~np.eye(len(weights), dtype=bool)
    if ratio < 1.0 and off.any():
        threshold = np.quantile(weights[off], 1.0 - ratio)
        weights[weights < threshold] = 0.0
    return weights


class ConditionKnowledgeGraph:
    """Condition knowledge graph with three weighted relations.

    Nodes are conditions; ``weights[relation][i, j]`` is the weight of the
    edge from condition ``i`` to condition ``j`` (0 for no edge). The three
    relations follow the paper:

    - ``"diagnosis"`` (Eq. 1): co-occurrence of two conditions among the
      diagnoses of the same admission, normalized by the row sum, so
      ``w[i, j] = CoOcc(i, j) / sum_k CoOcc(i, k)``.
    - ``"record"`` (Eqs. 2-3): each condition is summarized by the mean and
      standard deviation, per variable, of its patients' time series
      ``s_v``; ``w[i, j] = 1 / (1 + ||s_i - s_j||)``, top half kept.
    - ``"drug"`` (Eq. 4): Jaccard similarity of the drugs given to the
      patients with each condition, top half kept.

    Args:
        conditions: Node names, in index order.
        weights: Relation name to an ``[N, N]`` weight matrix.

    Examples:
        >>> import numpy as np
        >>> graph = ConditionKnowledgeGraph(
        ...     ["a", "b"], {"drug": np.array([[0.0, 0.5], [0.5, 0.0]])}
        ... )
        >>> graph.triples().tolist()
        [[0, 0, 1], [1, 0, 0]]
    """

    RELATIONS: tuple[str, ...] = ("diagnosis", "record", "drug")

    def __init__(self, conditions: Sequence[str], weights: dict[str, np.ndarray]):
        self.conditions = list(conditions)
        self.index = {c: i for i, c in enumerate(self.conditions)}
        self.relations = list(weights)
        self.weights = {r: np.asarray(w, dtype=np.float64) for r, w in weights.items()}
        for name, w in self.weights.items():
            if w.shape != (len(self.conditions), len(self.conditions)):
                raise ValueError(f"weights[{name!r}] must be [N, N], got {w.shape}")

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[dict[str, Any]],
        condition_key: str = "condition",
        diagnoses_key: str | None = "diagnoses",
        drugs_key: str | None = "drugs",
        record_key: str | None = "dynamic",
        visit_key: str | None = "hadm_id",
        record_top_ratio: float = 0.5,
        drug_top_ratio: float = 0.5,
    ) -> ConditionKnowledgeGraph:
        """Builds the graph from (training) samples.

        Use the training split only, so that validation and test patients do
        not shape the graph. Each relation is skipped when its key is None.

        Args:
            samples: Samples with the keys below, e.g. a ``SampleDataset``
                from :class:`~pyhealth.tasks.KnowRareICUMortalityMIMIC4`.
            condition_key: The sample's condition (a node).
            diagnoses_key: The admission's diagnosis categories; only those
                that are also nodes count towards co-occurrence.
            drugs_key: The admission's drug names.
            record_key: A ``[T, F]`` (or ``[F]``) numeric record.
            visit_key: Counts each admission's diagnoses once when several
                samples (ICU stays) share it; None counts every sample.
            record_top_ratio: Share of record edges kept (0.5 in the paper).
            drug_top_ratio: Share of drug edges kept (0.5 in the paper).

        Returns:
            The graph, with nodes sorted by name.

        Examples:
            >>> samples = [
            ...     {"condition": "a", "diagnoses": ["a", "b"], "drugs": ["x"]},
            ...     {"condition": "b", "diagnoses": ["b"], "drugs": ["x", "y"]},
            ... ]
            >>> graph = ConditionKnowledgeGraph.from_samples(
            ...     samples, record_key=None, visit_key=None
            ... )
            >>> graph.conditions, graph.relations
            (['a', 'b'], ['diagnosis', 'drug'])
            >>> graph.weights["diagnosis"].tolist()
            [[0.0, 1.0], [1.0, 0.0]]
        """
        conditions: list[str] = []
        diagnoses: list[list[str]] = []
        drugs: dict[str, set] = {}
        sums: dict[str, np.ndarray] = {}
        squares: dict[str, np.ndarray] = {}
        counts: Counter = Counter()
        seen_visits: set = set()
        for sample in samples:
            condition = str(sample[condition_key])
            conditions.append(condition)
            visit = sample.get(visit_key) if visit_key is not None else None
            if diagnoses_key is not None and (visit is None or visit not in seen_visits):
                diagnoses.append([str(d) for d in sample[diagnoses_key]])
                if visit is not None:
                    seen_visits.add(visit)
            if drugs_key is not None:
                drugs.setdefault(condition, set()).update(sample[drugs_key])
            if record_key is not None:
                record = torch.as_tensor(sample[record_key], dtype=torch.float64)
                rows = record.reshape(-1, record.shape[-1]).numpy()
                sums[condition] = sums.get(condition, 0.0) + rows.sum(0)
                squares[condition] = squares.get(condition, 0.0) + (rows**2).sum(0)
                counts[condition] += rows.shape[0]

        nodes = sorted(set(conditions))
        index = {c: i for i, c in enumerate(nodes)}
        n = len(nodes)
        weights: dict[str, np.ndarray] = {}

        if diagnoses_key is not None:
            cooc = np.zeros((n, n))
            for codes in diagnoses:
                present = sorted({index[c] for c in codes if c in index})
                for i in present:
                    for j in present:
                        if i != j:
                            cooc[i, j] += 1
            totals = cooc.sum(1, keepdims=True)
            weights["diagnosis"] = np.divide(
                cooc, totals, out=np.zeros_like(cooc), where=totals > 0
            )

        if record_key is not None:
            stats = []
            for c in nodes:
                mean = sums[c] / counts[c]
                var = np.maximum(squares[c] / counts[c] - mean**2, 0.0)
                stats.append(np.concatenate([mean, np.sqrt(var)]))
            stats = np.stack(stats)
            distance = np.linalg.norm(stats[:, None, :] - stats[None, :, :], axis=-1)
            weights["record"] = _keep_top(1.0 / (1.0 + distance), record_top_ratio)

        if drugs_key is not None:
            jaccard = np.zeros((n, n))
            for i, a in enumerate(nodes):
                for j, b in enumerate(nodes):
                    union = drugs[a] | drugs[b]
                    if i != j and union:
                        jaccard[i, j] = len(drugs[a] & drugs[b]) / len(union)
            weights["drug"] = _keep_top(jaccard, drug_top_ratio)

        return cls(nodes, weights)

    def triples(self, top_edge_ratio: float = 1.0) -> np.ndarray:
        """Returns the edges as ``(head, relation, tail)`` index triples.

        Args:
            top_edge_ratio: Share of each relation's edges kept, highest
                weights first (the paper's KG-sparsity study; the authors
                keep 60% on MIMIC-III and 5% on eICU).

        Returns:
            An ``int64`` array of shape ``[E, 3]``.

        Examples:
            >>> import numpy as np
            >>> w = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
            >>> graph = ConditionKnowledgeGraph(["a", "b", "c"], {"drug": w})
            >>> graph.triples(top_edge_ratio=0.5).tolist()
            [[0, 0, 1], [1, 0, 0]]
        """
        triples = []
        for r, name in enumerate(self.relations):
            w = self.weights[name]
            heads, tails = np.nonzero(w > 0)
            order = np.argsort(-w[heads, tails], kind="stable")
            keep = order[: round(len(order) * top_edge_ratio)]
            for h, t in sorted(zip(heads[keep], tails[keep])):
                triples.append((h, r, t))
        return np.asarray(triples, dtype=np.int64).reshape(-1, 3)


# --------------------------------------------------------------------- TuckER
class TuckER(nn.Module):
    """TuckER knowledge graph embedding (Balazevic et al., EMNLP 2019).

    Scores a triple ``(h, r, t)`` as ``W x_1 e_h x_2 w_r x_3 e_t`` with a
    shared core tensor ``W`` (Eq. 5 of the KnowRare paper), with batch
    normalization and dropout as in the TuckER paper, and is trained with
    binary cross-entropy against all tails at once (1-N scoring). KnowRare
    uses the entity embeddings ``E`` to compare conditions.

    Args:
        num_entities: Number of entities (conditions).
        num_relations: Number of relations.
        entity_dim: Entity embedding width.
        relation_dim: Relation embedding width.
        input_dropout: Dropout on the head embedding.
        hidden_dropout1: Dropout on the relation-specific core matrix.
        hidden_dropout2: Dropout before scoring the tails.

    Examples:
        >>> import torch
        >>> model = TuckER(num_entities=5, num_relations=2, entity_dim=4,
        ...                relation_dim=3)
        >>> model(torch.tensor([0, 1]), torch.tensor([1, 0])).shape
        torch.Size([2, 5])
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int = 32,
        relation_dim: int = 32,
        input_dropout: float = 0.3,
        hidden_dropout1: float = 0.4,
        hidden_dropout2: float = 0.5,
    ) -> None:
        super().__init__()
        self.entity = nn.Embedding(num_entities, entity_dim)
        self.relation = nn.Embedding(num_relations, relation_dim)
        self.core = nn.Parameter(
            torch.empty(relation_dim, entity_dim, entity_dim).uniform_(-1.0, 1.0)
        )
        nn.init.xavier_normal_(self.entity.weight)
        nn.init.xavier_normal_(self.relation.weight)
        self.bn0 = nn.BatchNorm1d(entity_dim)
        self.bn1 = nn.BatchNorm1d(entity_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

    def forward(self, head: Tensor, relation: Tensor) -> Tensor:
        """Logits of every entity as the tail of ``(head, relation, ?)``.

        Args:
            head: Head entity indices, shape ``[B]``.
            relation: Relation indices, shape ``[B]``.

        Returns:
            Logits of shape ``[B, num_entities]``.
        """
        e = self.input_dropout(self.bn0(self.entity(head)))
        r = self.relation(relation)
        d = e.shape[-1]
        core = (r @ self.core.view(r.shape[-1], -1)).view(-1, d, d)
        x = torch.bmm(e.unsqueeze(1), self.hidden_dropout1(core)).squeeze(1)
        x = self.hidden_dropout2(self.bn1(x))
        return x @ self.entity.weight.t()

    @property
    def entity_embeddings(self) -> Tensor:
        """The entity embedding matrix, detached, shape ``[N, entity_dim]``."""
        return self.entity.weight.detach()

    def fit(
        self,
        triples: np.ndarray | Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 128,
        label_smoothing: float = 0.0,
        seed: int | None = None,
    ) -> list[float]:
        """Trains the embeddings on the graph's triples (1-N scoring).

        Args:
            triples: ``[E, 3]`` ``(head, relation, tail)`` indices, e.g.
                from :meth:`ConditionKnowledgeGraph.triples`.
            epochs: Passes over the ``(head, relation)`` pairs.
            lr: Adam learning rate.
            batch_size: Pairs per batch.
            label_smoothing: TuckER's label smoothing (0 disables it).
            seed: Seed for the batch order; None uses torch's global state.

        Returns:
            The mean training loss of every epoch.

        Examples:
            >>> import numpy as np
            >>> model = TuckER(3, 1, entity_dim=4, relation_dim=2)
            >>> losses = model.fit(np.array([[0, 0, 1], [1, 0, 2]]), epochs=3,
            ...                    seed=0)
            >>> len(losses)
            3
        """
        triples = torch.as_tensor(np.asarray(triples), dtype=torch.long)
        if triples.numel() == 0:
            raise ValueError("TuckER.fit needs at least one triple")
        device = self.entity.weight.device
        pairs, inverse = torch.unique(triples[:, :2], dim=0, return_inverse=True)
        targets = torch.zeros(len(pairs), self.entity.num_embeddings)
        targets[inverse, triples[:, 2]] = 1.0
        if label_smoothing > 0:
            targets = (1.0 - label_smoothing) * targets + 1.0 / targets.shape[1]
        generator = torch.Generator()
        generator.manual_seed(
            int(torch.randint(0, 2**31 - 1, ()).item()) if seed is None else seed
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        self.train()
        for _ in range(epochs):
            order = torch.randperm(len(pairs), generator=generator)
            losses = []
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                if len(batch) < 2:  # batch norm needs two rows
                    continue
                h, r = pairs[batch, 0].to(device), pairs[batch, 1].to(device)
                loss = F.binary_cross_entropy_with_logits(
                    self(h, r), targets[batch].to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            history.append(float(np.mean(losses)) if losses else float("nan"))
        self.eval()
        return history


def select_source_conditions(
    embeddings: Tensor | np.ndarray,
    conditions: Sequence[str],
    target: str,
    k: int,
) -> list[str]:
    """Picks the ``k`` conditions most similar to ``target`` (Eq. 6).

    Similarity is the cosine similarity of the condition embeddings, e.g.
    :attr:`TuckER.entity_embeddings`. The paper selects 10% of all
    conditions (``k = int(0.1 * len(conditions)) + 1`` in the authors'
    code).

    Args:
        embeddings: ``[N, d]`` embeddings, row ``i`` for ``conditions[i]``.
        conditions: The condition names.
        target: The target rare condition (must be in ``conditions``).
        k: Number of source conditions.

    Returns:
        The source conditions, most similar first (the target excluded).

    Examples:
        >>> import torch
        >>> emb = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        >>> select_source_conditions(emb, ["a", "b", "c"], "a", k=1)
        ['b']
    """
    embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    conditions = list(conditions)
    t = conditions.index(target)
    similarity = F.cosine_similarity(embeddings[t : t + 1], embeddings, dim=-1)
    similarity[t] = -float("inf")
    order = torch.argsort(similarity, descending=True, stable=True)
    return [conditions[i] for i in order[: min(k, len(conditions) - 1)].tolist()]


# ---------------------------------------------------------------------- model
class _GradientReverse(torch.autograd.Function):
    """Identity forward; multiplies the gradient by ``-scale`` backward."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * grad, None


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LeakyReLU(), nn.Linear(hidden, outputs))


def _dense(value: Any, device: torch.device) -> Tensor:
    if isinstance(value, tuple):  # processors that return (value, mask, ...)
        value = value[0]
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class KnowRare(BaseModel):
    """KnowRare predictor with joint adversarial domain adaptation.

    The encoder follows the paper's Supplementary C.1: a one-layer LSTM over
    the ``[T, F]`` time series (``temporal_key``) and a two-layer MLP with
    LeakyReLU over the static variables (``static_key``), both of width
    ``hidden_dim``; their outputs are concatenated into the latent
    representation ``h_T`` and a two-layer MLP classifier predicts the
    outcome. Binary, multiclass and multilabel labels are supported.

    The full method trains in two stages (see the module docstring and the
    example notebook ``examples/knowrare_mimic4_icu_mortality.ipynb``):

    - :meth:`pretrain` fits the encoder and a two-layer MLP decoder by
      next-step prediction of the series on all conditions.
    - For each target rare condition, a copy of the pre-trained model is
      fine-tuned with :class:`~pyhealth.trainer.Trainer` on the target and
      its source conditions after :meth:`set_domains`. During training the
      loss then adds the discriminator's cross-entropy through a gradient
      reversal layer, and every sample's prediction loss is weighted by
      the inverse prevalence of its condition (Eq. 12).

    Without :meth:`set_domains` the model is a plain LSTM classifier.
    ``forward`` works without labels and returns ``y_prob`` and ``logit``
    (plus ``embed``, the latent representation).

    Args:
        dataset: The dataset (used for the schemas and input widths).
        temporal_key: The ``[T, F]`` tensor time-series field.
        static_key: Optional ``[S]`` tensor field of static variables.
        condition_key: The field naming each sample's condition (domain).
        hidden_dim: Width of the LSTM, the MLPs and the discriminator (128
            in the paper).
        adversarial_weight: ``lambda`` in Eq. 12, the weight of the reversed
            discriminator gradient (tuned over 0.005-0.1 by the authors).
        inverse_prevalence_weighting: Weight the loss by ``1 / p(v)`` after
            :meth:`set_domains`.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": f"p{i}", "series": [[float(i), 1.0]] * 4,
        ...      "static": [0.5 * i], "condition": "ab"[i % 2], "y": i % 2}
        ...     for i in range(6)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"series": "tensor", "static": "tensor"}, {"y": "binary"}
        ... )
        >>> model = KnowRare(dataset, temporal_key="series", hidden_dim=8)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=6)))
        >>> sorted(model(**batch))
        ['embed', 'logit', 'loss', 'y_prob', 'y_true']
        >>> model.set_domains("a", ["b"], conditions=batch["condition"])
        >>> model.domains
        ['a', 'b']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        temporal_key: str = "dynamic",
        static_key: str | None = "static",
        condition_key: str = "condition",
        hidden_dim: int = 128,
        adversarial_weight: float = 0.02,
        inverse_prevalence_weighting: bool = True,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("KnowRare supports exactly one label key")
        if self.mode not in ("binary", "multiclass", "multilabel"):
            raise ValueError(f"KnowRare does not support mode {self.mode!r}")
        self.label_key = self.label_keys[0]
        self.temporal_key = temporal_key
        self.static_key = static_key
        self.condition_key = condition_key
        self.hidden_dim = hidden_dim
        self.adversarial_weight = adversarial_weight
        self.inverse_prevalence_weighting = inverse_prevalence_weighting

        sample = dataset[0]
        series = _dense(sample[temporal_key], torch.device("cpu"))
        if series.dim() != 2:
            raise ValueError(
                f"{temporal_key!r} must be a [T, F] tensor, got {tuple(series.shape)}"
            )
        self.num_features = series.shape[-1]
        self.temporal_encoder = nn.LSTM(self.num_features, hidden_dim, batch_first=True)
        latent_dim = hidden_dim
        self.static_encoder = None
        if static_key is not None:
            static_dim = _dense(sample[static_key], torch.device("cpu")).numel()
            self.static_encoder = nn.Sequential(
                _mlp(static_dim, hidden_dim, hidden_dim), nn.LeakyReLU()
            )
            latent_dim += hidden_dim
        self.latent_dim = latent_dim
        self.output_size = self.label_processor.output_size()
        self.classifier = _mlp(latent_dim, hidden_dim, self.output_size)
        self.decoder = _mlp(latent_dim, hidden_dim, self.num_features)
        self.discriminator: nn.Module | None = None
        self.domains: list[str] = []
        self._domain_index: dict[str, int] = {}
        self._condition_weights: dict[str, float] = {}

    # ------------------------------------------------------------- encoding
    def _static(self, batch: dict[str, Any], length: int) -> Tensor | None:
        if self.static_encoder is None:
            return None
        static = _dense(batch[self.static_key], self.device).reshape(length, -1)
        return self.static_encoder(static)

    def _latent(self, states: Tensor, static: Tensor | None) -> Tensor:
        """Concatenates LSTM states ``[..., H]`` with the static embedding."""
        if static is None:
            return states
        shape = (*states.shape[:-1], static.shape[-1])
        if states.dim() == 3:
            static = static.unsqueeze(1)
        return torch.cat([static.expand(shape), states], dim=-1)

    def encode(self, **batch: Any) -> Tensor:
        """The latent representation ``h_T``, shape ``[B, latent_dim]``.

        Args:
            **batch: A batch from a PyHealth dataloader.

        Returns:
            The static embedding concatenated with the last LSTM state.
        """
        series = _dense(batch[self.temporal_key], self.device)
        _, (last, _) = self.temporal_encoder(series)
        return self._latent(last[-1], self._static(batch, series.shape[0]))

    # -------------------------------------------------------------- domains
    def set_domains(
        self,
        target: str,
        sources: Sequence[str],
        conditions: Iterable[str] | None = None,
    ) -> None:
        """Sets up adversarial adaptation to ``target`` from ``sources``.

        Creates a fresh discriminator over the ``k + 1`` conditions (target
        first) and, when ``conditions`` is given, the inverse-prevalence
        weights ``w_v = 1 / p(v)``, scaled so their mean over the given
        samples is 1. Samples whose condition is not among the domains get
        weight 1 and are left out of the adversarial loss.

        Args:
            target: The target rare condition.
            sources: The source conditions, e.g. from
                :func:`select_source_conditions`.
            conditions: The condition of every training sample, for the
                weights.

        Examples:
            >>> model.set_domains("a", ["b", "c"], conditions=["a", "b", "b"])  # doctest: +SKIP
            >>> model.condition_weights  # doctest: +SKIP
            {'a': 1.5, 'b': 0.75}
        """
        self.domains = [target, *[s for s in sources if s != target]]
        self._domain_index = {c: i for i, c in enumerate(self.domains)}
        self.discriminator = _mlp(
            self.latent_dim + self.output_size, self.hidden_dim, len(self.domains)
        ).to(self.device)
        self._condition_weights = {}
        if conditions is not None and self.inverse_prevalence_weighting:
            counts = Counter(str(c) for c in conditions)
            total = sum(counts.values())
            weights = {c: total / n for c, n in counts.items()}
            scale = len(weights)  # mean over samples of total / n_v
            self._condition_weights = {c: w / scale for c, w in weights.items()}

    @property
    def condition_weights(self) -> dict[str, float]:
        """The inverse-prevalence weight of each condition (empty if unset)."""
        return dict(self._condition_weights)

    # --------------------------------------------------------------- forward
    def forward(self, **batch: Any) -> dict[str, Tensor]:
        """Predicts the outcome; adds the adversarial loss while training.

        Args:
            **batch: A batch from a PyHealth dataloader.

        Returns:
            ``logit``, ``y_prob`` and ``embed`` (``h_T``); with labels also
            ``loss`` and ``y_true``. While training with domains set,
            ``loss`` includes the discriminator loss and ``loss_adv`` holds
            it alone.
        """
        embed = self.encode(**batch)
        logit = self.classifier(embed)
        y_prob = self.label_processor.activation(logit)
        results: dict[str, Tensor] = {"logit": logit, "y_prob": y_prob, "embed": embed}
        if self.label_key not in batch:
            return results

        y_true = batch[self.label_key].to(self.device)
        if self.mode == "multiclass":
            y_true = y_true.long().reshape(-1)
        else:
            y_true = y_true.float().reshape(logit.shape)
        loss = self.label_processor.loss(logit, y_true, reduction="none")
        if loss.dim() > 1:
            loss = loss.mean(dim=-1)
        conditions = [str(c) for c in batch.get(self.condition_key, [])]
        if self._condition_weights and len(conditions) == len(loss):
            weights = torch.tensor(
                [self._condition_weights.get(c, 1.0) for c in conditions],
                device=self.device,
            )
            loss = loss * weights
        loss = loss.mean()

        if (
            self.training
            and self.discriminator is not None
            and self.adversarial_weight > 0
            and len(conditions) == len(embed)
        ):
            domain = torch.tensor(
                [self._domain_index.get(c, -1) for c in conditions], device=self.device
            )
            known = domain >= 0
            if known.any():
                joint = torch.cat([embed, y_prob], dim=-1)[known]
                joint = _GradientReverse.apply(joint, self.adversarial_weight)
                loss_adv = F.cross_entropy(self.discriminator(joint), domain[known])
                results["loss_adv"] = loss_adv.detach()
                loss = loss + loss_adv
        results["loss"] = loss
        results["y_true"] = y_true
        return results

    # ----------------------------------------------------------- pretraining
    def pretrain_loss(self, **batch: Any) -> Tensor:
        """Next-step prediction loss of Eq. 8 (mean squared error).

        The decoder predicts ``x_{t+1}`` from ``h_t``, the static embedding
        concatenated with the LSTM state after ``x_1..x_t``, for
        ``t = 1..T-1``.

        Args:
            **batch: A batch from a PyHealth dataloader (labels unused).

        Returns:
            A scalar loss.
        """
        series = _dense(batch[self.temporal_key], self.device)
        if series.shape[1] < 2:
            raise ValueError("pre-training needs series of at least 2 steps")
        states, _ = self.temporal_encoder(series[:, :-1])
        hidden = self._latent(states, self._static(batch, series.shape[0]))
        return F.mse_loss(self.decoder(hidden), series[:, 1:])

    def pretrain(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int | None = 10,
    ) -> dict[str, list[float]]:
        """Condition-agnostic pre-training by next-step prediction.

        Trains the encoder and decoder with Adam on :meth:`pretrain_loss`.
        With a validation loader, keeps the weights of the epoch with the
        lowest validation loss and stops after ``patience`` epochs without
        improvement (the paper uses 10).

        Args:
            train_dataloader: Batches of all conditions' training samples.
            val_dataloader: Optional validation batches.
            epochs: Maximum number of epochs (100 in the paper).
            lr: Adam learning rate.
            patience: Early-stopping patience; None disables it.

        Returns:
            ``{"train": [...], "val": [...]}`` mean losses per epoch.

        Examples:
            >>> history = model.pretrain(train_loader, val_loader, epochs=5)  # doctest: +SKIP
        """
        params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith(("classifier.", "discriminator."))
        ]
        optimizer = torch.optim.Adam(params, lr=lr)
        history: dict[str, list[float]] = {"train": [], "val": []}
        best, best_state, waited = float("inf"), None, 0
        for _ in range(epochs):
            self.train()
            losses = []
            for batch in train_dataloader:
                loss = self.pretrain_loss(**batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            history["train"].append(float(np.mean(losses)))
            if val_dataloader is None:
                continue
            self.eval()
            with torch.no_grad():
                val = float(np.mean([self.pretrain_loss(**b).item() for b in val_dataloader]))
            history["val"].append(val)
            if val < best:
                best, waited = val, 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                waited += 1
                if patience is not None and waited >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return history
