"""ProtoECGNet: case-based interpretable multi-label ECG classification.

Re-implementation of the model in "ProtoECGNet: Case-Based Interpretable Deep
Learning for Multi-Label ECG Classification with Contrastive Learning"
(Sethi et al., MLHC 2025, https://arxiv.org/abs/2504.08713). The reference
code is at https://github.com/bbj-lab/protoecgnet .

A 1D ResNet turns a 12-lead ECG into a feature map. Each *prototype branch*
compares (windows of) that map with learned prototype vectors that are assigned
to labels, and a last linear layer turns prototype similarities into logits.
Calling :meth:`project_prototypes` replaces every prototype with the
feature-space representation of a real training ECG segment, and returns the
source-sample and window indices to the caller. In global mode
(``proto_time_len=None``) the stored vector is a time-averaged feature rather
than a raw segment, so case-based explanations should display the matching
window from the source sample rather than the prototype vector itself. Store
the returned mapping alongside the checkpoint to recover those windows after
reloading.
"""

from collections.abc import Sequence
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import RandomSampler

from ..datasets import SampleDataset
from .base_model import BaseModel

STAGES = (
    "all",
    "feature_extractor",
    "prototypes",
    "joint",
    "classifier",
    "fusion",
)

#: Physician-defined grouping of the 71 PTB-XL labels into the three prototype
#: branches of the paper (Appendix G): 16 rhythm, 52 morphology, 3 global.
PTBXL_BRANCH_LABELS: dict[str, list[str]] = {
    "rhythm": [
        "1AVB", "2AVB", "3AVB", "AFIB", "AFLT", "BIGU", "IVCD", "PACE", "PSVT",
        "SARRH", "SBRAD", "SR", "STACH", "SVARR", "SVTAC", "TRIGU",
    ],
    "morphology": [
        "ABQRS", "ALMI", "AMI", "ANEUR", "ASMI", "CLBBB", "CRBBB", "HVOLT",
        "ILBBB", "ILMI", "IMI", "INJAL", "INJAS", "INJIL", "INJIN", "INJLA",
        "INVT", "IPLMI", "IPMI", "IRBBB", "ISCAL", "ISCAN", "ISCAS", "ISCIL",
        "ISCIN", "ISCLA", "ISC_", "LAFB", "LAO/LAE", "LMI", "LNGQT", "LOWT",
        "LPFB", "LPR", "LVH", "LVOLT", "NDT", "NST_", "NT_", "PAC", "PMI",
        "PRC(S)", "PVC", "QWAVE", "RAO/RAE", "RVH", "SEHYP", "STD_", "STE_",
        "TAB_", "VCLVH", "WPW",
    ],
    "global": ["DIG", "EL", "NORM"],
}

#: Per-branch settings matching the paper (Sec. 3.3 and Appendix F): a 1D
#: ResNet-18 with global prototypes (5 per label) for rhythm; a 2D ResNet-18
#: with partial prototypes of 3 latent steps, top-5 pooling and 18 prototypes
#: per label for morphology; and a 2D ResNet-18 whose global prototypes span
#: the full latent time axis (32 steps for a 12 x 1000 ECG, i.e. a
#: 512 x 1 x 32 prototype) with 7 prototypes per label for the global labels.
PTBXL_BRANCH_KWARGS: dict[str, dict] = {
    "rhythm": {
        "backbone": "resnet1d",
        "proto_time_len": None,
        "prototypes_per_label": 5,
    },
    "morphology": {
        "backbone": "resnet2d",
        "proto_time_len": 3,
        "topk": 5,
        "prototypes_per_label": 18,
    },
    "global": {
        "backbone": "resnet2d",
        "proto_time_len": 32,
        "prototypes_per_label": 7,
    },
}


class BasicBlock1d(nn.Module):
    """Residual block with two 1D convolutions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the first convolution.
        kernel_size: Convolution kernel size.

    Examples:
        >>> import torch
        >>> block = BasicBlock1d(64, 128, stride=2)
        >>> block(torch.randn(2, 64, 100)).shape
        torch.Size([2, 128, 50])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, pad, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 1, pad, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a skip connection that is downsampled when stride > 1 or channels change."""
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet1dBackbone(nn.Module):
    """1D ResNet feature extractor that keeps the temporal axis.

    With the defaults (ResNet-18 layout) a 12 x 1000 input gives a
    ``(batch, 512, 32)`` feature map.

    Args:
        in_channels: Number of ECG leads.
        base_width: Channels of the first stage; later stages use 2x, 4x, 8x.
        layers: Number of residual blocks in each of the four stages.
        kernel_size: Convolution kernel size inside the blocks.

    Attributes:
        out_channels: Number of channels of the output feature map.

    Examples:
        >>> import torch
        >>> backbone = ResNet1dBackbone()
        >>> backbone(torch.randn(2, 12, 1000)).shape
        torch.Size([2, 512, 32])
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, 7, 2, 3, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        blocks: list[nn.Module] = []
        channels = base_width
        for stage, count in enumerate(layers):
            width = base_width * 2**stage
            for index in range(count):
                stride = 2 if (stage > 0 and index == 0) else 1
                blocks.append(BasicBlock1d(channels, width, stride, kernel_size))
                channels = width
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the signal through the stem (conv-7, BN, ReLU, maxpool) and then all residual stages, preserving the temporal axis."""
        return self.blocks(self.stem(x))


class ResNet2dBackbone(nn.Module):
    """ResNet-18 that reads a 12-lead ECG as a one-channel image.

    The input ``(batch, leads, time)`` is treated as a ``1 x leads x time``
    image. The first convolution has a ``leads x 7`` kernel (so all leads are
    mixed immediately) and the global pooling is dropped to keep the temporal
    axis, giving a ``(batch, 512, time')`` map (``time' = 32`` for 1000 samples).

    Args:
        in_leads: Number of ECG leads.
        pretrained: If True, initializes the residual stages from ImageNet
            weights (requires a download); the first convolution is always
            trained from scratch because its shape differs.

    Attributes:
        out_channels: Channels of the output feature map (512).

    Examples:
        >>> import torch
        >>> backbone = ResNet2dBackbone()
        >>> backbone(torch.randn(2, 12, 1000)).shape
        torch.Size([2, 512, 32])
    """

    def __init__(self, in_leads: int = 12, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        net.conv1 = nn.Conv2d(1, 64, (in_leads, 7), (1, 2), (0, 3), bias=False)
        # avgpool and fc are not used; replace with no-ops to drop their parameters.
        net.avgpool = nn.Identity()
        net.fc = nn.Identity()
        self.net = net
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unsqueezes to a 1-channel image, runs the ResNet-18 stem and four residual stages, then squeezes the lead axis to restore a 1D temporal map."""
        net = self.net
        x = net.maxpool(net.relu(net.bn1(net.conv1(x.unsqueeze(1)))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return x.squeeze(2)


class PrototypeBranch(nn.Module):
    """One prototype branch predicting a subset of the labels.

    Prototypes are assigned to a single label (``prototypes_per_label`` each)
    or to a pair of labels (``joint_prototypes_per_pair`` each). In ``global``
    mode (``proto_time_len=None``) the feature map is average-pooled over
    time; otherwise a prototype covers ``proto_time_len`` consecutive time
    steps and is matched against every window of that length (``partial``
    prototypes). The similarity of a prototype to an ECG is the mean of its
    ``topk`` best window similarities.

    Args:
        in_channels: Number of ECG leads.
        num_labels: Number of labels handled by this branch.
        proto_dim: Channels of the prototype space.
        prototypes_per_label: Prototypes assigned to each single label.
        joint_prototypes_per_pair: Prototypes assigned to each label pair.
        proto_time_len: Temporal length of a prototype, or None for global.
        backbone: ``"resnet1d"`` (ResNet-18 layout over time) or
            ``"resnet2d"`` (torchvision ResNet-18 over a leads x time image).
        pretrained: ImageNet initialization of the 2D backbone.
        base_width: Width of the 1D backbone's first stage.
        layers: Residual blocks per stage of the 1D backbone.
        similarity: ``"cosine"`` (scaled cosine, as in the paper) or ``"l2"``
            (ProtoPNet-style ``log((d + 1) / (d + eps))`` of the L2 distance).
        topk: Number of best windows averaged into the similarity.
        similarity_scale: Scale ``a`` of the cosine similarity. Defaults to
            ``sqrt(proto_dim * window)``.
        activation: ``"log"`` or ``"linear"`` map used when ``similarity="l2"``.
        last_layer_connection_weight: Initial weight between a prototype and
            its own label(s); other connections start at ``-0.5``.
        epsilon: Small constant of the log activation.

    Examples:
        >>> import torch
        >>> branch = PrototypeBranch(in_channels=12, num_labels=5)
        >>> out = branch(torch.randn(2, 12, 1000))
        >>> out["logits"].shape
        torch.Size([2, 5])
    """

    def __init__(
        self,
        in_channels: int,
        num_labels: int,
        proto_dim: int = 512,
        prototypes_per_label: int = 5,
        joint_prototypes_per_pair: int = 0,
        proto_time_len: int | None = None,
        backbone: str = "resnet1d",
        pretrained: bool = False,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        similarity: str = "cosine",
        topk: int = 1,
        similarity_scale: float | None = None,
        activation: str = "log",
        last_layer_connection_weight: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if similarity not in ("cosine", "l2"):
            raise ValueError("similarity must be 'cosine' or 'l2'")
        if activation not in ("log", "linear"):
            raise ValueError("activation must be 'log' or 'linear'")
        if backbone not in ("resnet1d", "resnet2d"):
            raise ValueError("backbone must be 'resnet1d' or 'resnet2d'")
        if proto_time_len is not None and proto_time_len < 1:
            raise ValueError("proto_time_len must be a positive integer or None")
        if topk < 1:
            raise ValueError("topk must be at least 1")
        self.num_labels = num_labels
        self.proto_dim = proto_dim
        self.proto_time_len = proto_time_len
        self.similarity = similarity
        self.topk = topk
        self.activation = activation
        self.epsilon = epsilon
        self.window = 1 if proto_time_len is None else proto_time_len
        self.scale = (
            float(similarity_scale)
            if similarity_scale is not None
            else float((proto_dim * self.window) ** 0.5)
        )

        if backbone == "resnet1d":
            self.backbone: nn.Module = ResNet1dBackbone(
                in_channels, base_width, layers
            )
        else:
            self.backbone = ResNet2dBackbone(in_channels, pretrained)
        feat = self.backbone.out_channels
        self.add_on = nn.Sequential(
            nn.Conv1d(feat, proto_dim, 1),
            nn.ReLU(),
            nn.Conv1d(proto_dim, proto_dim, 1),
            nn.Sigmoid(),
        )
        self.pretrain_head = nn.Linear(feat, num_labels)

        identity = self._make_identity(
            num_labels, prototypes_per_label, joint_prototypes_per_pair
        )
        self.register_buffer("prototype_class_identity", identity)
        self.num_prototypes = identity.shape[0]
        self.prototype_vectors = nn.Parameter(
            torch.randn(self.num_prototypes, proto_dim, self.window)
        )
        self.classifier = nn.Linear(self.num_prototypes, num_labels, bias=False)
        self.set_last_layer_connections(last_layer_connection_weight)

    @staticmethod
    def _make_identity(
        num_labels: int, per_label: int, per_pair: int
    ) -> torch.Tensor:
        """Builds the binary prototype-to-label assignment matrix, with per-label rows first and then a shared row for every label pair."""
        pairs = list(combinations(range(num_labels), 2))
        total = num_labels * per_label + len(pairs) * per_pair
        identity = torch.zeros(total, num_labels)
        for label in range(num_labels):
            identity[label * per_label : (label + 1) * per_label, label] = 1.0
        offset = num_labels * per_label
        for index, (first, second) in enumerate(pairs):
            start = offset + index * per_pair
            identity[start : start + per_pair, first] = 1.0
            identity[start : start + per_pair, second] = 1.0
        return identity

    def set_last_layer_connections(
        self, correct: float = 1.0, incorrect: float = -0.5
    ) -> None:
        """Initializes the classifier from the prototype-label assignment.

        Args:
            correct: Weight between a prototype and its own label(s).
            incorrect: Weight between a prototype and all other labels.
        """
        own = self.prototype_class_identity.t()  # (labels, prototypes)
        with torch.no_grad():
            self.classifier.weight.copy_(correct * own + incorrect * (1 - own))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Projects backbone features to prototype space; collapses the time axis to a single step when using global prototypes (``proto_time_len=None``)."""
        fmap = self.add_on(self.backbone(x))
        if self.proto_time_len is None:
            fmap = fmap.mean(dim=2, keepdim=True)
        return fmap

    def windows(self, fmap: torch.Tensor) -> torch.Tensor:
        """Slides a window of ``proto_time_len`` steps over the time axis and flattens channel and time into a single vector per window, matching the prototype shape for dot-product similarity."""
        if fmap.shape[2] < self.window:
            raise ValueError(
                f"Feature map has {fmap.shape[2]} time steps but proto_time_len "
                f"is {self.window}; use a smaller proto_time_len or longer ECGs."
            )
        cut = fmap.unfold(2, self.window, 1)  # (B, D, W, L)
        return cut.permute(0, 2, 1, 3).reshape(fmap.shape[0], cut.shape[2], -1)

    def distances(self, windows: torch.Tensor) -> torch.Tensor:
        """Computes pairwise L2 distances using ||a-b||²=||a||²-2aᵀb+||b||², avoiding an explicit loop over the P prototypes."""
        protos = self.prototype_vectors.flatten(1)
        d2 = (
            windows.pow(2).sum(-1, keepdim=True)
            - 2 * windows @ protos.t()
            + protos.pow(2).sum(-1)
        )
        return torch.sqrt(d2.clamp_min(0.0) + 1e-8)

    def similarity_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """Similarity of every window to every prototype ``(batch, W, P)``.

        Larger values mean a closer match for both similarity types.
        """
        if self.similarity == "cosine":
            protos = F.normalize(self.prototype_vectors.flatten(1), dim=1)
            return self.scale * (F.normalize(windows, dim=-1) @ protos.t())
        distance = self.distances(windows)
        if self.activation == "log":
            return torch.log((distance + 1) / (distance + self.epsilon))
        return -distance

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the branch.

        Args:
            x: ECG batch of shape ``(batch, leads, time)``.

        Returns:
            ``(logits, similarity, best_window)``: logits ``(batch,
            num_labels)``, the pooled prototype similarities ``(batch,
            num_prototypes)`` and, for explanations, the index of the
            best-matching window per prototype ``(batch, num_prototypes)``.
        """
        sims = self.similarity_windows(self.windows(self.feature_map(x)))
        k = min(self.topk, sims.shape[1])
        pooled = sims.topk(k, dim=1).values.mean(dim=1)
        best_window = sims.argmax(dim=1)
        return self.classifier(pooled), pooled, best_window

    def pretrain_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of the plain ResNet head used by the feature-extractor stage."""
        return self.pretrain_head(self.backbone(x).mean(dim=2))

    def prototype_terms(
        self,
        similarity: torch.Tensor,
        y: torch.Tensor,
        cooccurrence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prototype regularizers (Eqs. 4-7 of the paper).

        Args:
            similarity: Prototype similarities ``(batch, num_prototypes)``.
            y: Multi-hot labels of this branch ``(batch, num_labels)``.
            cooccurrence: Optional label co-occurrence matrix of this branch
                ``(num_labels, num_labels)``; enables the contrastive term.

        Returns:
            A dict with the ``clst`` (clustering), ``sep`` (separation),
            ``div`` (diversity) and ``cnrst`` (contrastive) terms.
        """
        identity = self.prototype_class_identity
        # A prototype is "positive" for a sample if it is assigned to at least
        # one label present in the sample; otherwise all its labels are absent
        # and it is "negative" (Eqs. 4-5 of the paper).
        positive = (y.float() @ identity.t()) > 0  # (B, P)
        negative = ~positive
        neg_inf = torch.finfo(similarity.dtype).min

        def masked_max(mask: torch.Tensor) -> torch.Tensor:
            """Per-sample max similarity over ``mask``; 0 where mask is empty."""
            best = similarity.masked_fill(~mask, neg_inf).max(dim=1).values
            return torch.where(mask.any(dim=1), best, torch.zeros_like(best))

        terms = {
            "clst": -masked_max(positive).mean(),
            "sep": masked_max(negative).mean(),
        }
        protos = F.normalize(self.prototype_vectors.flatten(1), dim=1)
        gram = protos @ protos.t()
        count = self.num_prototypes
        eye = torch.eye(count, device=gram.device)
        terms["div"] = (gram - eye).pow(2).sum() / count**2
        if cooccurrence is None:
            terms["cnrst"] = gram.new_zeros(())
        else:
            cos = gram.clamp(-1 + 1e-6, 1 - 1e-6)
            pos = (identity @ cooccurrence @ identity.t()).clamp(max=1.0)
            neg = 1.0 - pos
            pos = pos / (pos.sum() + 1e-6)
            neg = neg / (neg.sum() + 1e-6)
            terms["cnrst"] = -((cos * pos).sum() - (cos * neg).sum()) / count**0.5
        return terms


class ProtoECGNet(BaseModel):
    """Case-based interpretable multi-label ECG classifier.

    The model contains one or several :class:`PrototypeBranch` modules. Each
    branch is responsible for a disjoint group of labels (for example rhythm
    labels with global prototypes and morphology labels with partial
    prototypes). Branch outputs are combined by scattering their logits into
    the full label space (``fusion="average"``) or by a learned linear layer
    over all prototype similarities (``fusion="learned"``, trained in the
    ``"fusion"`` stage).

    Training follows the reference procedure and can be driven by PyHealth's
    ``Trainer`` by switching stages with :meth:`set_stage`:

    1. ``"feature_extractor"``: train the backbone with a plain linear head.
    2. ``"joint"``: train backbone, add-on layers and prototypes.
    3. :meth:`project_prototypes`: replace every prototype by the most similar
       real training segment of its label(s). Steps 2-3 can be repeated.
    4. ``"classifier"``: train only the last layer on the projected
       prototypes. For ``fusion="learned"`` finish with ``"fusion"``.

    ``"all"`` trains everything with one loss and is a quick baseline. In the
    ``"classifier"`` and ``"fusion"`` stages an L1 penalty (``lam_l1``) on the
    weights between a prototype and labels it is *not* assigned to keeps the
    final layers sparse, as in the paper.

    Args:
        dataset: A fitted :class:`~pyhealth.datasets.SampleDataset` with a
            ``signal`` tensor input and a multilabel output.
        label_groups: Optional mapping from branch name to the label names the
            branch handles (see :data:`PTBXL_BRANCH_LABELS`). Names absent
            from the dataset vocabulary are dropped, and the groups must then
            cover every label exactly once. Defaults to one branch with all
            labels.
        branch_kwargs: Optional per-branch overrides of
            :class:`PrototypeBranch` arguments, keyed by branch name (see
            :data:`PTBXL_BRANCH_KWARGS`).
        in_channels: Number of ECG leads. Defaults to 12.
        proto_dim: Channels of the prototype space. Defaults to 512.
        prototypes_per_label: Prototypes per label. Defaults to 5.
        joint_prototypes_per_pair: Prototypes per label pair. Defaults to 0.
        proto_time_len: Prototype length (None means global). Defaults to None.
        similarity: ``"cosine"`` (default) or ``"l2"``.
        topk: Windows averaged into a similarity. Defaults to 1.
        base_width: Width of the 1D ResNet's first stage. Defaults to 64.
        layers: Residual blocks per 1D stage. Defaults to ``(2, 2, 2, 2)``.
        fusion: ``"average"`` (scatter each branch's logits into the full
            label space with no cross-branch interaction) or ``"learned"``
            (train a linear layer over all prototype similarities in the
            ``"fusion"`` stage). Defaults to ``"average"``.
        lam_clst: Weight of the clustering loss. Defaults to 0.004.
        lam_sep: Weight of the separation loss. Defaults to 0.0004.
        lam_div: Weight of the diversity loss. Defaults to 250.
        lam_cnrst: Weight of the contrastive loss. Defaults to 300. It is
            only active once a co-occurrence matrix has been set.
        lam_l1: Weight of the L1 penalty of the classifier/fusion stages.
            Defaults to 0 (off).
        class_weights: Optional positive-class weights ``(num_labels,)``.
        label_cooccurrence: Optional label co-occurrence matrix.

    Note:
        ``ProtoECGNet`` applies loss weighting via its own ``class_weights``
        buffer. :meth:`set_pos_weight` is overridden to route the inherited
        :class:`~pyhealth.models.BaseModel` argument into ``class_weights``
        (including the ``"balanced"`` option), so callers like autoimprove
        work unchanged.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "a", "signal": [[0.0] * 256] * 12, "labels": ["X"]},
        ...     {"patient_id": "b", "signal": [[1.0] * 256] * 12, "labels": ["Y"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"labels": "multilabel"},
        ... )
        >>> model = ProtoECGNet(dataset, base_width=4, layers=(1, 1, 1, 1))
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> out["logit"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        label_groups: dict[str, list[str]] | None = None,
        branch_kwargs: dict[str, dict] | None = None,
        in_channels: int = 12,
        proto_dim: int = 512,
        prototypes_per_label: int = 5,
        joint_prototypes_per_pair: int = 0,
        proto_time_len: int | None = None,
        similarity: str = "cosine",
        topk: int = 1,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        fusion: str = "average",
        lam_clst: float = 0.004,
        lam_sep: float = 0.0004,
        lam_div: float = 250.0,
        lam_cnrst: float = 300.0,
        lam_l1: float = 0.0,
        class_weights: torch.Tensor | None = None,
        label_cooccurrence: torch.Tensor | None = None,
    ) -> None:
        super().__init__(dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError("ProtoECGNet expects one signal input and one label.")
        if self.mode != "multilabel":
            raise ValueError("ProtoECGNet requires a multilabel output schema.")
        if fusion not in ("average", "learned"):
            raise ValueError("fusion must be 'average' or 'learned'")
        self.signal_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.fusion = fusion
        self.stage = "all"
        self.lam_clst, self.lam_sep = lam_clst, lam_sep
        self.lam_div, self.lam_cnrst, self.lam_l1 = lam_div, lam_cnrst, lam_l1

        vocab = dataset.output_processors[self.label_key].label_vocab
        self.label_names = [
            name for name, _ in sorted(vocab.items(), key=lambda i: i[1])
        ]
        self.num_labels = len(self.label_names)
        groups = self._prepare_groups(label_groups)
        if any("." in name for name in groups):
            raise ValueError("branch names must not contain '.'")
        index_of = {name: i for i, name in enumerate(self.label_names)}

        self.branch_names = list(groups)
        self.branches = nn.ModuleDict()
        for name, members in groups.items():
            self.register_buffer(
                f"index_{name}", torch.tensor([index_of[m] for m in members])
            )
            options = {
                "proto_dim": proto_dim,
                "prototypes_per_label": prototypes_per_label,
                "joint_prototypes_per_pair": joint_prototypes_per_pair,
                "proto_time_len": proto_time_len,
                "similarity": similarity,
                "topk": topk,
                "base_width": base_width,
                "layers": layers,
            }
            options.update((branch_kwargs or {}).get(name, {}))
            self.branches[name] = PrototypeBranch(
                in_channels, len(members), **options
            )

        total = sum(b.num_prototypes for b in self.branches.values())
        self.fusion_classifier: nn.Linear | None = (
            nn.Linear(total, self.num_labels) if fusion == "learned" else None
        )
        self.register_buffer("fusion_mask", self._build_fusion_mask(total))
        # Real-tensor defaults so state_dict keys are stable regardless of whether
        # label statistics were provided. ones(class_weights) is BCE's identity;
        # zeros(label_cooccurrence) + the .any() gate in forward keeps the
        # contrastive term off until a real matrix is supplied.
        self.register_buffer("class_weights", torch.ones(self.num_labels))
        self.register_buffer(
            "label_cooccurrence", torch.zeros(self.num_labels, self.num_labels)
        )
        if class_weights is not None:
            self.class_weights = class_weights.float()
        if label_cooccurrence is not None:
            self.label_cooccurrence = label_cooccurrence.float()
        # Modules whose BN running stats must stay fixed after projection.
        # Populated by set_stage; consumed by train().
        self._frozen_modules: set[nn.Module] = set()
        # Set by project_prototypes; not in state_dict — persist alongside ckpt.
        self.projection_info: dict[str, list[dict | None]] | None = None

    def _prepare_groups(
        self, groups: dict[str, list[str]] | None
    ) -> dict[str, list[str]]:
        """Drops unknown labels and validates that groups partition the labels."""
        if groups is None:
            return {"all": list(self.label_names)}
        known = set(self.label_names)
        cleaned = {
            name: [m for m in members if m in known]
            for name, members in groups.items()
        }
        cleaned = {name: members for name, members in cleaned.items() if members}
        flat = [m for members in cleaned.values() for m in members]
        if sorted(flat) != sorted(self.label_names):
            raise ValueError(
                "label_groups must contain every label exactly once; got "
                f"{len(flat)} known entries for {len(self.label_names)} labels."
            )
        return cleaned

    def _build_fusion_mask(self, total: int) -> torch.Tensor:
        """Assembles the cross-branch prototype-to-label assignment mask, used to identify non-owned connections for the L1 sparsity penalty in the fusion stage."""
        mask = torch.zeros(self.num_labels, total)
        offset = 0
        for name in self.branch_names:
            branch = self.branches[name]
            index = self.branch_index(name)
            mask[index, offset : offset + branch.num_prototypes] = (
                branch.prototype_class_identity.t()
            )
            offset += branch.num_prototypes
        return mask

    def branch_index(self, name: str) -> torch.Tensor:
        """Returns the global label indices owned by branch ``name``; used for scattering branch logits into the full label space and slicing the co-occurrence matrix."""
        return getattr(self, f"index_{name}")

    def set_pos_weight(
        self,
        pos_weight,
        dataset=None,
    ) -> None:
        """Honor :meth:`BaseModel.set_pos_weight` by routing it to ``class_weights``.

        ``ProtoECGNet`` applies positive-class weighting through its own
        ``class_weights`` buffer, so the inherited method would otherwise be a
        no-op (silently dropping e.g. ``pos_weight="balanced"`` from
        ``autoimprove``). This override delegates to the base implementation
        to parse the argument (scalar, sequence, tensor, ``"balanced"``, or
        ``None``) and then mirrors ``self._pos_weight`` into
        ``self.class_weights`` so BCE picks it up. ``None`` resets
        ``class_weights`` to ones.

        Args:
            pos_weight: Same semantics as :meth:`BaseModel.set_pos_weight`.
            dataset: Required when ``pos_weight="balanced"``.
        """
        super().set_pos_weight(pos_weight, dataset)
        if self._pos_weight is None:
            self.class_weights = torch.ones(self.num_labels, device=self.device)
        else:
            self.class_weights = self._pos_weight.to(self.device).float()

    def set_label_statistics(
        self,
        class_weights: torch.Tensor | None = None,
        label_cooccurrence: torch.Tensor | None = None,
    ) -> None:
        """Sets class weights and the label co-occurrence matrix.

        Both are backed by persistent buffers initialised to ``ones`` (identity
        for BCE's ``pos_weight``) and ``zeros`` (gates the contrastive term off
        via an ``.any()`` check in :meth:`forward`); supplying a real
        ``label_cooccurrence`` enables the contrastive term, and supplying an
        all-zeros matrix explicitly also leaves it off.

        Args:
            class_weights: Positive-class weights ``(num_labels,)``.
            label_cooccurrence: Co-occurrence matrix ``(num_labels, num_labels)``.
        """
        if class_weights is not None:
            self.class_weights = class_weights.float().to(self.device)
        if label_cooccurrence is not None:
            self.label_cooccurrence = label_cooccurrence.float().to(self.device)

    @staticmethod
    def compute_label_statistics(
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes class weights and Jaccard label co-occurrence.

        Args:
            labels: Multi-hot training labels ``(num_samples, num_labels)``.

        Returns:
            ``(class_weights, cooccurrence)``. Class weights follow the
            reference code, ``(total - count) / count`` where ``total`` is the
            sum of all label counts.
        """
        y = labels.float()
        counts = y.sum(dim=0).clamp_min(1.0)
        weights = (counts.sum() - counts) / counts
        inter = y.t() @ y
        union = (counts.unsqueeze(0) + counts.unsqueeze(1) - inter).clamp_min(1e-6)
        return weights, inter / union

    def set_stage(self, stage: str) -> None:
        """Selects which parameters are trained.

        Args:
            stage: One of ``"all"``, ``"feature_extractor"``, ``"prototypes"``,
                ``"joint"``, ``"classifier"`` or ``"fusion"``.
        """
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if stage == "fusion" and self.fusion_classifier is None:
            raise ValueError("the 'fusion' stage requires fusion='learned'")
        trainable = {
            "feature_extractor": ("backbone", "pretrain_head"),
            "prototypes": ("prototype_vectors",),
            "joint": ("backbone", "add_on", "prototype_vectors"),
            "classifier": ("classifier",),
        }
        for name, param in self.named_parameters():
            if stage == "all":
                param.requires_grad = True
            elif stage == "fusion":
                param.requires_grad = name.startswith("fusion_classifier")
            else:
                parts = name.split(".")
                param.requires_grad = (
                    parts[0] == "branches" and parts[2] in trainable[stage]
                )
        self.stage = stage
        # Recompute once here so train() is O(|frozen|) with no allocation.
        # Use recurse=True so a PrototypeBranch whose backbone has trainable
        # children is never added even though prototype_vectors itself is frozen.
        self._frozen_modules = {
            m for m in self.modules()
            if any(True for _ in m.parameters(recurse=False))
            and not any(p.requires_grad for p in m.parameters(recurse=True))
        }

    def train(self, mode: bool = True) -> "ProtoECGNet":
        super().train(mode)
        if mode:
            for m in self._frozen_modules:
                m.eval()
        return self

    def _scatter(self, per_branch: dict[str, torch.Tensor], batch: int) -> torch.Tensor:
        """Writes branch logits into the full label space."""
        full = torch.zeros(batch, self.num_labels, device=self.device)
        for name, logits in per_branch.items():
            full = full.index_copy(1, self.branch_index(name), logits)
        return full

    def _l1_penalty(self) -> torch.Tensor:
        """L1 norm of the weights between prototypes and non-assigned labels."""
        if self.stage == "fusion":
            weight = self.fusion_classifier.weight
            return (weight.abs() * (1.0 - self.fusion_mask)).sum()
        total = self.fusion_mask.new_zeros(())
        for name in self.branch_names:
            branch = self.branches[name]
            own = branch.prototype_class_identity.t()
            total = total + (branch.classifier.weight.abs() * (1.0 - own)).sum()
        return total

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: Must contain the ECG tensor under the signal key
                ``(batch, leads, time)``; may contain the multi-hot labels.

        Returns:
            A dict with ``logit`` and ``y_prob`` ``(batch, num_labels)``,
            ``similarity`` (all prototype similarities, concatenated over
            branches) and, when labels are given, ``loss`` and ``y_true``.
        """
        x = kwargs[self.signal_key].to(self.device).float()
        y = kwargs.get(self.label_key)
        y = None if y is None else y.to(self.device).float()
        batch = x.shape[0]
        extra: list[torch.Tensor] = []

        if self.stage == "feature_extractor":
            logits = self._scatter(
                {n: self.branches[n].pretrain_logits(x) for n in self.branch_names},
                batch,
            )
            sims = x.new_zeros(batch, 0)
        else:
            outputs = {n: self.branches[n](x) for n in self.branch_names}
            sims = torch.cat([o[1] for o in outputs.values()], dim=1)
            logits = self._scatter({n: o[0] for n, o in outputs.items()}, batch)
            if self.stage == "fusion":
                logits = self.fusion_classifier(sims)
            if y is not None and self.stage in ("all", "joint", "prototypes"):
                # Hoist the one-time .any() check out of the per-branch loop.
                cnrst_on = self.lam_cnrst > 0 and bool(self.label_cooccurrence.any())
                for name, (_, sim, _) in outputs.items():
                    index = self.branch_index(name)
                    cooc = (
                        self.label_cooccurrence[index][:, index] if cnrst_on else None
                    )
                    t = self.branches[name].prototype_terms(sim, y[:, index], cooc)
                    extra.append(
                        self.lam_clst * t["clst"]
                        + self.lam_sep * t["sep"]
                        + self.lam_div * t["div"]
                        + self.lam_cnrst * t["cnrst"]
                    )
            if y is not None and self.lam_l1 > 0 and self.stage in (
                "classifier",
                "fusion",
            ):
                extra.append(self.lam_l1 * self._l1_penalty())

        result = {"logit": logits, "y_prob": torch.sigmoid(logits), "similarity": sims}
        if y is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.class_weights
            )
            for term in extra:
                loss = loss + term
            result["loss"] = loss
            result["y_true"] = y
        return result

    @torch.no_grad()
    def project_prototypes(self, dataloader) -> dict[str, list[dict | None]]:
        """Replaces each prototype with the most similar real training segment.

        For every prototype only samples that carry all of its label(s) are
        considered. The prototype is set to the feature window with the
        highest similarity, so that it corresponds to an actual ECG segment.
        In global mode (``proto_time_len=None``), the stored vector is a
        time-averaged feature rather than a raw ECG segment.

        The result is also stored on ``self.projection_info`` for the current
        session. It is **not** part of ``state_dict``; save it separately
        alongside the model checkpoint to support case-based explanations after
        reloading (e.g. ``torch.save(model.projection_info, "proj.pt")``).

        Args:
            dataloader: Training dataloader (``shuffle=False``) yielding
                batches with the signal and the multi-hot labels.

        Returns:
            For each branch a list (one entry per prototype) with the
            ``sample`` index within the loader order, the ``window`` index and
            the ``similarity`` of the chosen segment, or None if no sample
            carried the prototype's labels (the prototype is then unchanged).

        Raises:
            ValueError: If ``dataloader`` uses a :class:`RandomSampler`.
                Sample index tracking requires a deterministic order.
        """
        if getattr(dataloader, "shuffle_samples", False) or isinstance(
            getattr(dataloader, "sampler", None), RandomSampler
        ):
            raise ValueError(
                "project_prototypes requires a deterministic loader "
                "(shuffle=False). Sample index mapping is wrong with a "
                "shuffled loader."
            )
        was_training = self.training
        self.eval()
        best = {
            name: [None] * self.branches[name].num_prototypes
            for name in self.branch_names
        }
        offset = 0
        for batch in dataloader:
            x = batch[self.signal_key].to(self.device).float()
            y = batch[self.label_key].to(self.device).float()
            for name in self.branch_names:
                branch = self.branches[name]
                index = self.branch_index(name)
                windows = branch.windows(branch.feature_map(x))  # (B, W, N)
                sim = branch.similarity_windows(windows)  # (B, W, P)
                own = branch.prototype_class_identity  # (P, L)
                needed = own.sum(dim=1)
                have = (y[:, index] @ own.t()) >= needed - 1e-6  # (B, P)
                for p in range(branch.num_prototypes):
                    valid = have[:, p].nonzero(as_tuple=True)[0]
                    if len(valid) == 0:
                        continue
                    scores = sim[valid, :, p]  # (V, W)
                    value, flat = scores.flatten().max(dim=0)
                    row, win = flat // scores.shape[1], flat % scores.shape[1]
                    current = best[name][p]
                    if current is None or value.item() > current["similarity"]:
                        best[name][p] = {
                            "sample": offset + valid[row].item(),
                            "window": win.item(),
                            "similarity": value.item(),
                            "vector": windows[valid[row], win].clone(),
                        }
            offset += x.shape[0]
        for name in self.branch_names:
            branch = self.branches[name]
            for p, entry in enumerate(best[name]):
                if entry is not None:
                    vector = entry.pop("vector").view(
                        branch.proto_dim, branch.window
                    )
                    branch.prototype_vectors.data[p] = vector
        self.train(was_training)
        self.projection_info = best
        return best

    @torch.no_grad()
    def prototype_contributions(
        self, signal: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-prototype contributions to each logit, for explanations.

        Args:
            signal: ECG batch ``(batch, leads, time)``.

        Returns:
            For every branch a tensor equal to similarity times classifier
            weight. Shape is ``(batch, num_labels, num_branch_prototypes)``
            when ``stage="fusion"`` (fusion classifier weights; summing over
            all branches and prototypes gives ``logit - fusion_classifier.bias``
            because the branch classifiers have no bias but the fusion
            classifier does), or ``(batch, num_branch_labels,
            num_branch_prototypes)`` otherwise (branch classifier weights,
            which have no bias so contributions sum exactly to the branch
            logits).
        """
        was_training = self.training
        self.eval()
        x = signal.to(self.device).float()
        use_fusion = self.stage == "fusion" and self.fusion_classifier is not None
        out = {}
        offset = 0
        for name in self.branch_names:
            branch = self.branches[name]
            _, sim, _ = branch(x)
            if use_fusion:
                w = self.fusion_classifier.weight[:, offset : offset + branch.num_prototypes]
                out[name] = sim.unsqueeze(1) * w.unsqueeze(0)
                offset += branch.num_prototypes
            else:
                out[name] = sim.unsqueeze(1) * branch.classifier.weight.unsqueeze(0)
        self.train(was_training)
        return out
