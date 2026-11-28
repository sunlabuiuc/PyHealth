"""HiCu: hierarchical curriculum learning for automated ICD coding.

Authors: Huyen Nguyen (hnguyen654), Aaron Tran (atran403)
Course: CSE 6250 Big Data for Health, Georgia Tech, Fall 2026

Paper: Weiming Ren, Ruijing Zeng, Tongzi Wu, Tianshu Zhu, Rahul G. Krishnan.
"HiCu: Leveraging Hierarchy for Curriculum Learning in Automated ICD Coding."
Machine Learning for Healthcare (MLHC), PMLR 182:198-223, 2022.
https://proceedings.mlr.press/v182/ren22a.html
Adapted from the authors' implementation, https://github.com/wren93/HiCu-ICD
(ICD-9 hierarchy tables, MultiResCNN decoder, knowledge transfer, loss settings).

The model is a MultiResCNN text encoder (Li and Yu, 2020) with a per-label
attention decoder. Instead of predicting the leaf ICD codes directly, it is
trained level by level down the ICD code tree (chapter -> block -> 3-digit
code -> 1-decimal code -> 2-decimal code). When training moves to the next
level, each child code's attention query starts from its parent's query
(knowledge transfer), and a Poincare (hyperbolic) embedding of the code's
position in the tree corrects it so that siblings can differ.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel

# ---------------------------------------------------------------------------
# ICD-9 hierarchy (chapters and blocks of ICD-9-CM)
# ---------------------------------------------------------------------------

# fmt: off
_DIAGNOSIS_BLOCKS: dict[str, list[str]] = {
    "001-139": [
        "001-009", "010-018", "020-027", "030-041", "042-042", "045-049",
        "050-059", "060-066", "070-079", "080-088", "090-099", "100-104",
        "110-118", "120-129", "130-136", "137-139",
    ],
    "140-239": [
        "140-149", "150-159", "160-165", "170-176", "179-189", "190-199",
        "200-209", "210-229", "230-234", "235-238", "239-239",
    ],
    "240-279": ["240-246", "249-259", "260-269", "270-279"],
    "280-289": [f"{c}-{c}" for c in range(280, 290)],
    "290-319": ["290-294", "295-299", "300-316", "317-319"],
    "320-389": [
        "320-327", "330-337", "338-338", "339-339", "340-349", "350-359",
        "360-379", "380-389",
    ],
    "390-459": [
        "390-392", "393-398", "401-405", "410-414", "415-417", "420-429",
        "430-438", "440-449", "451-459",
    ],
    "460-519": [
        "460-466", "470-478", "480-488", "490-496", "500-508", "510-519",
    ],
    "520-579": [
        "520-529", "530-539", "540-543", "550-553", "555-558", "560-569",
        "570-579",
    ],
    "580-629": [
        "580-589", "590-599", "600-608", "610-612", "614-616", "617-629",
    ],
    "630-679": [
        "630-639", "640-649", "650-659", "660-669", "670-677", "678-679",
    ],
    "680-709": ["680-686", "690-698", "700-709"],
    "710-739": ["710-719", "720-724", "725-729", "730-739"],
    "740-759": [f"{c}-{c}" for c in range(740, 760)],
    "760-779": ["760-763", "764-779"],
    "780-799": ["780-789", "790-796", "797-799"],
    "800-999": [
        "800-804", "805-809", "810-819", "820-829", "830-839", "840-848",
        "850-854", "860-869", "870-879", "880-887", "890-897", "900-904",
        "905-909", "910-919", "920-924", "925-929", "930-939", "940-949",
        "950-957", "958-959", "960-979", "980-989", "990-995", "996-999",
    ],
    "V01-V91": [
        "V01-V09", "V10-V19", "V20-V29", "V30-V39", "V40-V49", "V50-V59",
        "V60-V69", "V70-V82", "V83-V84", "V85-V85", "V86-V86", "V87-V87",
        "V88-V88", "V89-V89", "V90-V90", "V91-V91",
    ],
    "E000-E999": [
        "E000-E000", "E001-E030", "E800-E807", "E810-E819", "E820-E825",
        "E826-E829", "E830-E838", "E840-E845", "E846-E849", "E850-E858",
        "E860-E869", "E870-E876", "E878-E879", "E880-E888", "E890-E899",
        "E900-E909", "E910-E915", "E916-E928", "E929-E929", "E930-E949",
        "E950-E959", "E960-E969", "E970-E979", "E980-E989", "E990-E999",
    ],
}

_PROCEDURE_CHAPTERS: list[str] = [
    "00-00", "01-05", "06-07", "08-16", "17-17", "18-20", "21-29", "30-34",
    "35-39", "40-41", "42-54", "55-59", "60-64", "65-71", "72-75", "76-84",
    "85-86", "87-99",
]
# fmt: on

ICD9_DEPTH = 5
_OTHER = "OTHER"


def _in_range(value: str, span: str) -> bool:
    """Whether a code prefix such as ``"401"`` or ``"V45"`` lies in ``span``."""
    start, end = span.split("-")
    if value[0].isalpha() != start[0].isalpha():
        return False
    if value[0].isalpha():
        if value[0] != start[0]:
            return False
        value, start, end = value[1:], start[1:], end[1:]
    return int(start) <= int(value) <= int(end)


def icd9_code_path(code: str) -> list[str]:
    """Returns the path from the top of the ICD-9-CM tree down to ``code``.

    The path has :data:`ICD9_DEPTH` (5) levels: chapter, block, integer code,
    code with one decimal, code with two decimals. Shorter codes are padded
    by repeating the code itself, which is the "augmented label tree" of the
    HiCu paper (Section 3.3.1), so every code ends at the same depth.

    Diagnosis codes have a 3-character integer part (``"401.9"``,
    ``"V45.81"``) or 4 for E codes (``"E849.7"``); procedure codes have 2
    (``"38.93"``). Codes without a dot, as stored in MIMIC-III, are read as
    diagnosis codes. Codes that cannot be parsed are put under an
    ``"OTHER"`` branch instead of raising, so unusual labels still train.

    Args:
        code: An ICD-9-CM diagnosis or procedure code.

    Returns:
        The node names from the chapter down to ``code``; the last is
        ``code`` exactly as given.

    Examples:
        >>> icd9_code_path("401.9")
        ['390-459', '401-405', '401', '401.9', '401.9']
        >>> icd9_code_path("38.93")
        ['35-39', '38-38', '38', '38.9', '38.93']
    """
    label = code
    code = str(code).strip().upper()
    if "." in code:
        first, second = code.split(".", 1)
    else:
        size = 4 if code.startswith("E") else 3
        first, second = code[:size], code[size:]
    path: list[str] = []
    try:
        if first[0] in "VE" or len(first) == 3:
            chapters = [c for c in _DIAGNOSIS_BLOCKS if _in_range(first, c)]
            chapter = chapters[0]
            block = next(b for b in _DIAGNOSIS_BLOCKS[chapter] if _in_range(first, b))
        elif len(first) == 2 and first.isdigit():
            chapter = next(c for c in _PROCEDURE_CHAPTERS if _in_range(first, c))
            block = f"{first}-{first}"
        else:
            raise ValueError(code)
        path = [chapter, block, first]
        for i in range(min(len(second), 2)):
            path.append(f"{first}.{second[: i + 1]}")
    except (ValueError, IndexError, StopIteration):
        path = [_OTHER, _OTHER, code]
    # pad with the deepest node, then end with the label exactly as given,
    # so "4019" and "401.9" both sit under the node "401.9"
    while len(path) < ICD9_DEPTH:
        path.append(path[-1])
    return path[: ICD9_DEPTH - 1] + [label]


# ---------------------------------------------------------------------------
# Poincare embeddings (Nickel and Kiela, 2017)
# ---------------------------------------------------------------------------


def _poincare_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Distance between points ``u`` and ``v`` in the Poincare ball."""
    sq = ((u - v) ** 2).sum(-1)
    nu = (u**2).sum(-1).clamp(max=1 - 1e-5)
    nv = (v**2).sum(-1).clamp(max=1 - 1e-5)
    x = 1 + 2 * sq / ((1 - nu) * (1 - nv))
    return torch.acosh(x.clamp(min=1 + 1e-7))


def train_poincare_embeddings(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    dim: int = 50,
    epochs: int = 50,
    negatives: int = 10,
    lr: float = 0.1,
    batch_size: int = 512,
    seed: int = 0,
) -> torch.Tensor:
    """Embeds a tree in the Poincare ball with Riemannian SGD.

    For each edge ``(u, v)`` the loss is the softmax cross-entropy of
    ``-d(u, v)`` against ``-d(u, v')`` for ``negatives`` random nodes
    ``v'``, as in Nickel and Kiela (2017). The original HiCu code used
    gensim's ``PoincareModel`` with 50 dimensions, 10 negatives and 50
    epochs; this is a small dependency-free version of the same objective.

    Args:
        edges: Parent-child pairs of node indices.
        num_nodes: Number of nodes.
        dim: Embedding dimension. Default is 50.
        epochs: Passes over the edges. Default is 50.
        negatives: Negative samples per edge. Default is 10.
        lr: Learning rate. Default is 0.1.
        batch_size: Edges per step. Default is 512.
        seed: Random seed. Default is 0.

    Returns:
        A ``(num_nodes, dim)`` tensor of points inside the unit ball.

    Examples:
        >>> # a root (0) with two children (1, 2)
        >>> emb = train_poincare_embeddings([(0, 1), (0, 2)], num_nodes=3, dim=2)
        >>> emb.shape
        torch.Size([3, 2])
        >>> bool((emb.norm(dim=1) < 1).all())
        True
    """
    generator = torch.Generator().manual_seed(seed)
    emb = (torch.rand(num_nodes, dim, generator=generator) - 0.5) * 2e-3
    if not edges or num_nodes < 2:
        return emb
    pairs = torch.tensor(list(edges), dtype=torch.long)
    for epoch in range(epochs):
        # a short "burn-in" with a smaller learning rate, as in the paper
        step_lr = lr / 10 if epoch < min(10, epochs // 5) else lr
        order = torch.randperm(len(pairs), generator=generator)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[order[start : start + batch_size]]
            neg = torch.randint(num_nodes, (len(batch), negatives), generator=generator)
            emb.requires_grad_(True)
            u = emb[batch[:, 0]].unsqueeze(1)
            candidates = torch.cat([batch[:, 1:2], neg], dim=1)
            dist = _poincare_distance(u, emb[candidates])
            # summed, not averaged, so each edge takes a full SGD step as in
            # the per-edge updates of the original implementation
            loss = F.cross_entropy(
                -dist, torch.zeros(len(batch), dtype=torch.long), reduction="sum"
            )
            (grad,) = torch.autograd.grad(loss, emb)
            with torch.no_grad():
                # Riemannian gradient: rescale by the inverse metric tensor
                scale = ((1 - (emb**2).sum(-1, keepdim=True)) ** 2) / 4
                emb = emb - step_lr * scale * grad
                norm = emb.norm(dim=-1, keepdim=True)
                emb = torch.where(norm >= 1, emb / norm * (1 - 1e-5), emb)
    return emb.detach()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification (Ben-Baruch et al.).

    Down-weights easy negatives so training focuses on the rare positive
    labels: ``gamma_neg`` focuses the negatives, ``gamma_pos`` the
    positives, and ``clip`` shifts negative probabilities down by a margin
    (equations 12-13 of the HiCu paper). With all three set to 0 this is
    binary cross-entropy.

    Args:
        gamma_neg: Focusing parameter for negatives. Default is 1.
        gamma_pos: Focusing parameter for positives. Default is 0.
        clip: Probability margin ``m`` for negatives. Default is 0.05.
        reduction: ``"sum"`` (as in the HiCu code) or ``"mean"``.
        eps: Clamp for the logarithm. Default is 1e-8.

    Examples:
        >>> loss_fn = AsymmetricLoss()
        >>> logits = torch.tensor([[2.0, -3.0, -4.0]])
        >>> targets = torch.tensor([[1.0, 0.0, 0.0]])
        >>> loss_fn(logits, targets).shape
        torch.Size([])
    """

    def __init__(
        self,
        gamma_neg: float = 1.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = "sum",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction}")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            logits: Raw scores of shape ``(batch, labels)``.
            targets: 0/1 targets of the same shape.

        Returns:
            The scalar loss.
        """
        prob = torch.sigmoid(logits)
        prob_neg = 1 - prob
        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(prob.clamp(min=self.eps))
        loss = loss + (1 - targets) * torch.log(prob_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = prob * targets + prob_neg * (1 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                weight = torch.pow(1 - pt, gamma)
            loss = loss * weight
        return -loss.sum() if self.reduction == "sum" else -loss.mean()


class ResidualBlock(nn.Module):
    """Two 1D convolutions with a residual shortcut (MultiResCNN).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Odd convolution width.
        dropout: Dropout after the block.

    Examples:
        >>> block = ResidualBlock(100, 50, kernel_size=3, dropout=0.2)
        >>> block(torch.randn(2, 100, 30)).shape
        torch.Size([2, 50, 30])
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.left = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps ``(batch, in_channels, length)`` to ``(batch, out, length)``."""
        return self.dropout(torch.tanh(self.left(x) + self.shortcut(x)))


class MultiResCNNEncoder(nn.Module):
    """Multi-filter residual CNN text encoder (Li and Yu, 2020).

    Each filter width has its own channel: a convolution with ``tanh``
    followed by a residual block. The channel outputs are concatenated, so
    every token gets a ``len(filter_sizes) * num_filter_maps`` feature.

    Args:
        vocab_size: Number of word ids, padding (0) and unknown (1) included.
        embedding_dim: Word embedding size.
        filter_sizes: Odd convolution widths, one channel each.
        num_filter_maps: Output channels per filter width.
        dropout: Dropout on embeddings and after each residual block.

    Examples:
        >>> encoder = MultiResCNNEncoder(
        ...     vocab_size=1000, embedding_dim=100, filter_sizes=(3, 5),
        ...     num_filter_maps=50, dropout=0.2,
        ... )
        >>> encoder(torch.randint(0, 1000, (2, 40))).shape
        torch.Size([2, 40, 100])
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filter_sizes: Sequence[int],
        num_filter_maps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.channels = nn.ModuleList()
        for size in filter_sizes:
            base = nn.Conv1d(embedding_dim, embedding_dim, size, padding=size // 2)
            nn.init.xavier_uniform_(base.weight)
            self.channels.append(
                nn.ModuleDict(
                    {
                        "base": base,
                        "residual": ResidualBlock(
                            embedding_dim, num_filter_maps, size, dropout
                        ),
                    }
                )
            )
        self.output_dim = len(filter_sizes) * num_filter_maps

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encodes ``(batch, length)`` ids to ``(batch, length, output_dim)``."""
        x = self.embed_dropout(self.embedding(token_ids)).transpose(1, 2)
        outputs = []
        for channel in self.channels:
            h = torch.tanh(channel["base"](x))
            outputs.append(channel["residual"](h).transpose(1, 2))
        return torch.cat(outputs, dim=2)


# ---------------------------------------------------------------------------
# HiCu
# ---------------------------------------------------------------------------


class HiCu(BaseModel):
    """HiCu: MultiResCNN with hierarchical curriculum learning for ICD coding.

    Paper: Ren et al., "HiCu: Leveraging Hierarchy for Curriculum Learning in
    Automated ICD Coding", MLHC 2022.

    The model reads a clinical note (the dataset's one ``"text"`` input) and
    predicts its ICD codes (the dataset's one multilabel output). Every code
    is placed on a path of ``num_levels`` tree nodes (by default the ICD-9
    tree, see :func:`icd9_code_path`). Level ``k`` predicts the nodes at
    depth ``k``: a node is positive when any of the sample's codes is below
    it. The last level is the codes themselves, in the label processor's
    order.

    Train with a curriculum: call :meth:`set_level` with 0, 1, ...,
    ``num_levels - 1`` and train some epochs at each. Moving to the next
    level copies each parent's attention query and output weights to its
    children (knowledge transfer). With ``hyperbolic="add"`` (HiCu-A) or
    ``"cat"`` (HiCu-C), a fixed Poincare embedding of each node corrects its
    query, so that siblings starting from the same parent can differ.

    ``use_curriculum=False`` gives the plain MultiResCNN baseline: one
    level (the codes) with randomly initialized queries.

    Implementation notes, relative to the original code:

    * Text is lower-cased and split into alphabetic words; the vocabulary is
      built from the dataset (``min_word_freq``) unless ``vocab`` is given.
    * Padding positions are masked out of the attention.

    Args:
        dataset: The dataset to train the model. It must have exactly one
            text input and one multilabel output.
        code_paths: Optional map from every label to its path of tree nodes,
            all of the same length and ending with the label itself. Use
            this for another hierarchy (e.g. ICD-10) or a shuffled one.
            Default builds ICD-9 paths with :func:`icd9_code_path`.
        use_curriculum: Train over the hierarchy (HiCu) or only the leaves
            (MultiResCNN baseline). Default is True.
        hyperbolic: ``"add"`` (HiCu-A), ``"cat"`` (HiCu-C) or None for
            knowledge transfer without hyperbolic correction. Default "add".
        loss: ``"asl"`` (asymmetric loss) or ``"bce"``. Default is "asl".
        asl_gamma_neg: ASL focusing for negatives. Default is 1 (paper).
        asl_gamma_pos: ASL focusing for positives. Default is 0 (paper).
        asl_clip: ASL probability margin. Default is 0.05 (paper).
        asl_reduction: ``"sum"`` (paper code) or ``"mean"``. Default "sum".
        embedding_dim: Word embedding size. Default is 100.
        filter_sizes: Convolution widths. Default ``(3, 5, 9, 15, 19, 25)``.
        num_filter_maps: Channels per filter width. Default is 50.
        dropout: Dropout rate. Default is 0.2.
        hyperbolic_dim: Poincare embedding size. Default is 50.
        poincare_epochs: Epochs to train the Poincare embeddings. Default 50.
        max_length: Tokens read per note. Default is 4096.
        min_word_freq: Minimum count for a word to enter the vocabulary.
            Default is 3.
        vocab: Optional list of words (ids start at 2 in this order).
        pretrained_embeddings: Optional ``(len(vocab) + 2, embedding_dim)``
            word vectors, e.g. word2vec trained on the notes.
        seed: Seed for the Poincare embeddings. Default is 0.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import HiCu
        >>> samples = [
        ...     {"patient_id": "p0", "text": "hypertension on amlodipine",
        ...      "icd_codes": ["401.9"]},
        ...     {"patient_id": "p1", "text": "type ii diabetes glyburide",
        ...      "icd_codes": ["250.00", "401.9"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema={"text": "text"},
        ...     output_schema={"icd_codes": "multilabel"},
        ... )
        >>> model = HiCu(dataset, min_word_freq=1, poincare_epochs=2)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> for level in range(model.num_levels):
        ...     model.set_level(level)
        ...     out = model(**batch)  # train some epochs here
        >>> out["y_prob"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        code_paths: dict[str, Sequence[str]] | None = None,
        use_curriculum: bool = True,
        hyperbolic: str | None = "add",
        loss: str = "asl",
        asl_gamma_neg: float = 1.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        asl_reduction: str = "sum",
        embedding_dim: int = 100,
        filter_sizes: Sequence[int] = (3, 5, 9, 15, 19, 25),
        num_filter_maps: int = 50,
        dropout: float = 0.2,
        hyperbolic_dim: int = 50,
        poincare_epochs: int = 50,
        max_length: int = 4096,
        min_word_freq: int = 3,
        vocab: Sequence[str] | None = None,
        pretrained_embeddings: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("HiCu takes exactly one text input field.")
        if self.mode != "multilabel":
            raise ValueError("HiCu needs exactly one multilabel output field.")
        if hyperbolic not in (None, "add", "cat"):
            raise ValueError(f"hyperbolic must be 'add', 'cat' or None: {hyperbolic}")
        if loss not in ("asl", "bce"):
            raise ValueError(f"loss must be 'asl' or 'bce', got {loss}")
        if any(size % 2 == 0 for size in filter_sizes):
            raise ValueError("filter_sizes must be odd to keep the length.")
        if max_length < 1:
            raise ValueError("max_length must be positive.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.use_curriculum = use_curriculum
        self.hyperbolic = hyperbolic if use_curriculum else None
        self.max_length = max_length

        # words
        if vocab is None:
            vocab = self._build_vocab(dataset, min_word_freq)
        self.word2id: dict[str, int] = {w: i + 2 for i, w in enumerate(vocab)}
        self.encoder = MultiResCNNEncoder(
            vocab_size=len(self.word2id) + 2,
            embedding_dim=embedding_dim,
            filter_sizes=filter_sizes,
            num_filter_maps=num_filter_maps,
            dropout=dropout,
        )
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != self.encoder.embedding.weight.shape:
                raise ValueError(
                    "pretrained_embeddings must have shape "
                    f"{tuple(self.encoder.embedding.weight.shape)}"
                )
            self.encoder.embedding.weight.data.copy_(pretrained_embeddings)
        hidden = self.encoder.output_dim

        # label tree
        labels = sorted(
            self.label_processor.label_vocab,
            key=self.label_processor.label_vocab.get,
        )
        self.level_nodes, parents, leaf_nodes = self._build_tree(
            labels, code_paths, use_curriculum
        )
        self.num_levels = len(self.level_nodes)
        for level in range(self.num_levels):
            self.register_buffer(f"_parent_{level}", parents[level])
            self.register_buffer(f"_leaf_node_{level}", leaf_nodes[level])

        # per-level decoders: attention queries and output weights
        self.queries = nn.ParameterList()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        bound = 1 / math.sqrt(hidden)
        for nodes in self.level_nodes:
            for params in (self.queries, self.weights):
                param = torch.empty(len(nodes), hidden)
                nn.init.xavier_uniform_(param)
                params.append(nn.Parameter(param))
            self.biases.append(
                nn.Parameter(torch.empty(len(nodes)).uniform_(-bound, bound))
            )

        # hyperbolic correction
        self.hyperbolic_fc = nn.ModuleList()
        if self.hyperbolic is not None:
            embeddings = self._poincare_by_level(hyperbolic_dim, poincare_epochs, seed)
            for level, emb in enumerate(embeddings):
                self.register_buffer(f"_poincare_{level}", emb)
                in_dim = (
                    hyperbolic_dim if hyperbolic == "add" else (hidden + hyperbolic_dim)
                )
                self.hyperbolic_fc.append(nn.Linear(in_dim, hidden))

        if loss == "asl":
            self.asl: AsymmetricLoss | None = AsymmetricLoss(
                asl_gamma_neg, asl_gamma_pos, asl_clip, asl_reduction
            )
        else:
            self.asl = None
        self.current_level = self.num_levels - 1 if not use_curriculum else 0

    # ----------------------------------------------------------------- setup

    def _build_vocab(self, dataset: SampleDataset, min_freq: int) -> list[str]:
        """Counts the words of the training texts and keeps frequent ones."""
        counts: Counter = Counter()
        for sample in dataset:
            counts.update(self.tokenize(sample[self.feature_key]))
        return sorted(w for w, c in counts.items() if c >= min_freq)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lower-cases ``text`` and returns its alphabetic words."""
        return re.findall(r"[a-z]+", str(text).lower())

    @staticmethod
    def _build_tree(
        labels: list[str],
        code_paths: dict[str, Sequence[str]] | None,
        use_curriculum: bool,
    ) -> tuple[list[list[str]], list[torch.Tensor], list[torch.Tensor]]:
        """Indexes the tree nodes of each level.

        Returns:
            ``level_nodes[k]``: the node names at level ``k`` (the last level
            is ``labels`` in order); ``parents[k]``: the index of each level
            ``k`` node's parent at level ``k - 1`` (-1 at level 0);
            ``leaf_nodes[k]``: the level ``k`` node above each label.
        """
        if not use_curriculum:
            paths = {label: [label] for label in labels}
        elif code_paths is None:
            paths = {label: icd9_code_path(label) for label in labels}
        else:
            missing = [label for label in labels if label not in code_paths]
            if missing:
                raise ValueError(f"code_paths has no path for labels {missing[:5]}")
            paths = {label: list(code_paths[label]) for label in labels}
        depths = {len(p) for p in paths.values()}
        if len(depths) != 1:
            raise ValueError("All code paths must have the same length.")
        depth = depths.pop()
        for label, path in paths.items():
            if path[-1] != label:
                raise ValueError(f"The path of {label!r} must end with it.")

        level_nodes: list[list[str]] = []
        index: list[dict[str, int]] = []
        for level in range(depth):
            if level == depth - 1:
                nodes = list(labels)
            else:
                # name a node by its whole prefix so equal names under
                # different parents stay different nodes
                nodes = sorted({"/".join(p[: level + 1]) for p in paths.values()})
            level_nodes.append(nodes)
            index.append({n: i for i, n in enumerate(nodes)})

        def key(path: Sequence[str], level: int) -> str:
            return path[-1] if level == depth - 1 else "/".join(path[: level + 1])

        parents, leaf_nodes = [], []
        for level in range(depth):
            parent = torch.full((len(level_nodes[level]),), -1, dtype=torch.long)
            leaf = torch.empty(len(labels), dtype=torch.long)
            for i, label in enumerate(labels):
                path = paths[label]
                node = index[level][key(path, level)]
                leaf[i] = node
                if level > 0:
                    parent[node] = index[level - 1][key(path, level - 1)]
            parents.append(parent)
            leaf_nodes.append(leaf)
        return level_nodes, parents, leaf_nodes

    def _poincare_by_level(
        self, dim: int, epochs: int, seed: int
    ) -> list[torch.Tensor]:
        """Trains Poincare embeddings of the tree and splits them by level."""
        offsets = [0]
        for nodes in self.level_nodes:
            offsets.append(offsets[-1] + len(nodes))
        root = offsets[-1]
        edges = []
        for level in range(self.num_levels):
            parent = getattr(self, f"_parent_{level}")
            for i, p in enumerate(parent.tolist()):
                src = root if level == 0 else offsets[level - 1] + p
                edges.append((src, offsets[level] + i))
        emb = train_poincare_embeddings(
            edges, root + 1, dim=dim, epochs=epochs, seed=seed
        )
        return [emb[offsets[k] : offsets[k + 1]] for k in range(self.num_levels)]

    # ------------------------------------------------------------ curriculum

    def set_level(self, level: int, transfer: bool = True) -> None:
        """Moves the curriculum to ``level``.

        When ``level`` is the next level and ``transfer`` is True, every node
        at ``level`` starts from its parent's attention query and output
        weights (knowledge transfer, equation 9 of the paper).

        Args:
            level: Level to train and predict, ``0 .. num_levels - 1``.
            transfer: Copy the parents' weights when moving down one level.
                Default is True.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        if not 0 <= level < self.num_levels:
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]")
        if transfer and level == self.current_level + 1:
            parent = getattr(self, f"_parent_{level}")
            with torch.no_grad():
                self.queries[level].copy_(self.queries[level - 1][parent])
                self.weights[level].copy_(self.weights[level - 1][parent])
        self.current_level = level

    def level_targets(self, y_leaf: torch.Tensor, level: int) -> torch.Tensor:
        """Maps code targets to the targets of the nodes at ``level``.

        Args:
            y_leaf: 0/1 code targets of shape ``(batch, num_codes)``.
            level: Tree level.

        Returns:
            ``(batch, num_nodes)`` targets: 1 where any code below is 1.
        """
        nodes = getattr(self, f"_leaf_node_{level}")
        out = y_leaf.new_zeros(y_leaf.shape[0], len(self.level_nodes[level]))
        return out.index_add_(1, nodes, y_leaf).clamp_(max=1)

    # --------------------------------------------------------------- forward

    def _encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Turns texts into a padded ``(batch, length)`` tensor of word ids."""
        ids = [
            [self.word2id.get(w, 1) for w in self.tokenize(t)][: self.max_length]
            for t in texts
        ]
        length = max(1, max(len(i) for i in ids))
        out = torch.zeros(len(ids), length, dtype=torch.long)
        for row, seq in enumerate(ids):
            out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out.to(self.device)

    def _query(self, level: int) -> torch.Tensor:
        """The attention queries of ``level``, with hyperbolic correction."""
        query = self.queries[level]
        if self.hyperbolic == "add":
            query = query + self.hyperbolic_fc[level](
                getattr(self, f"_poincare_{level}")
            )
        elif self.hyperbolic == "cat":
            query = self.hyperbolic_fc[level](
                torch.cat([query, getattr(self, f"_poincare_{level}")], dim=1)
            )
        return query

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predicts the nodes of the current level.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`:
                the text field (a list of strings) and, optionally, the
                multilabel field (``(batch, num_codes)`` 0/1 tensor).

        Returns:
            A dictionary with ``logit`` and ``y_prob`` of shape
            ``(batch, nodes at the current level)``, and ``loss`` and
            ``y_true`` (targets mapped to the current level) when labels are
            given. At the last level the columns are the codes.
        """
        level = self.current_level
        token_ids = self._encode_text(kwargs[self.feature_key])
        hidden = self.encoder(token_ids)  # (batch, length, hidden)

        scores = hidden.matmul(self._query(level).t())  # (batch, length, nodes)
        scores = scores.masked_fill((token_ids == 0).unsqueeze(-1), -1e9)
        alpha = torch.softmax(scores, dim=1)
        context = alpha.transpose(1, 2).matmul(hidden)  # (batch, nodes, hidden)
        logit = (self.weights[level] * context).sum(-1) + self.biases[level]

        results = {"logit": logit, "y_prob": torch.sigmoid(logit)}
        if self.label_key in kwargs:
            y_leaf = kwargs[self.label_key].to(self.device).float()
            y_true = self.level_targets(y_leaf, level)
            if self.asl is not None:
                loss = self.asl(logit, y_true)
            elif level == self.num_levels - 1:
                loss = self.label_processor.loss(logit, y_true)
            else:
                loss = F.binary_cross_entropy_with_logits(logit, y_true)
            results["loss"] = loss
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = context.mean(dim=1)
        return results
