"""ECGBranchResNet: a black-box, label-grouped ResNet with a learned fusion head.

Author: Hanho Ryu (NetID: hanhor2, hanhor2@illinois.edu)

Paper: ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning (Sethi et al., MLHC 2025).
Paper link: https://arxiv.org/abs/2504.08713

ProtoECGNet (:class:`~pyhealth.models.ProtoECGNet`) trains one prototype
branch per label group and then a linear *fusion* layer over all prototype
similarities. The paper compares it with black-box ResNets, but none of those
baselines gets a comparable trained fusion head: the "aggregated branch
black-boxes" simply concatenate per-branch predictions. The paper's fusion
gain (0.9120 for aggregated prototype branches vs 0.9213 with fusion) can
therefore come from the fusion layer rather than from prototypes.

``ECGBranchResNet`` is the missing control. It uses the *same* label groups
and the *same* per-branch backbones as ProtoECGNet (1D ResNet-18 or the 2D
ResNet-18 that reads a 12-lead ECG as a ``1 x 12 x T`` image), but each
branch ends in an ordinary pooled linear head. Its fusion stage trains a
linear layer on the frozen branch outputs, exactly mirroring ProtoECGNet's
``"fusion"`` stage, including the optional L1 penalty on cross-group weights.
With a single group it reduces to the paper's full-label black-box ResNet.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel
from .protoecgnet import ResNet1dBackbone, ResNet2dBackbone

#: Training stages accepted by :meth:`ECGBranchResNet.set_stage`.
BRANCH_RESNET_STAGES = ("all", "branches", "fusion")

#: Inputs of the learned fusion head.
BRANCH_RESNET_FUSIONS = ("none", "logits", "features")


class ResNetBranch(nn.Module):
    """One black-box branch: backbone, temporal average pooling, linear head.

    Args:
        in_channels: Number of ECG leads.
        num_labels: Number of labels predicted by this branch.
        backbone: ``"resnet1d"`` (:class:`~pyhealth.models.ResNet1dBackbone`)
            or ``"resnet2d"`` (:class:`~pyhealth.models.ResNet2dBackbone`).
        dropout: Dropout probability before the linear head.
        pretrained: ImageNet initialization of the 2D backbone.
        base_width: Width of the 1D backbone's first stage.
        layers: Residual blocks per stage of the 1D backbone.

    Attributes:
        feature_dim: Size of the pooled feature vector.

    Raises:
        ValueError: If ``backbone`` is unknown or ``dropout`` is not in
            ``[0, 1)``.

    Examples:
        >>> import torch
        >>> branch = ResNetBranch(12, 3, base_width=4, layers=(1, 1, 1, 1))
        >>> logits, features = branch(torch.randn(2, 12, 256))
        >>> logits.shape, features.shape
        (torch.Size([2, 3]), torch.Size([2, 32]))
    """

    def __init__(
        self,
        in_channels: int,
        num_labels: int,
        backbone: str = "resnet1d",
        dropout: float = 0.0,
        pretrained: bool = False,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if backbone not in ("resnet1d", "resnet2d"):
            raise ValueError("backbone must be 'resnet1d' or 'resnet2d'")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if backbone == "resnet1d":
            self.backbone: nn.Module = ResNet1dBackbone(in_channels, base_width, layers)
        else:
            self.backbone = ResNet2dBackbone(in_channels, pretrained)
        self.feature_dim: int = self.backbone.out_channels
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_dim, num_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the branch.

        Args:
            x: ECG batch ``(batch, leads, time)``.

        Returns:
            ``(logits, features)`` of shapes ``(batch, num_labels)`` and
            ``(batch, feature_dim)``.
        """
        features = self.backbone(x).mean(dim=2)
        return self.head(self.dropout(features)), features


class ECGBranchResNet(BaseModel):
    """Black-box multi-branch ECG ResNet with an optional learned fusion head.

    The labels are partitioned into groups (``label_groups``), and each group
    gets its own :class:`ResNetBranch`. Branch logits are scattered back into
    the full label space. With ``fusion != "none"`` a linear fusion head maps
    the concatenated branch outputs to all labels:

    * ``"logits"``: input is the concatenated branch logits
      (``num_labels``-dim). With ``fusion_init="branches"`` the head starts
      as the identity, so it begins exactly at the scattered branch
      prediction.
    * ``"features"``: input is the concatenated pooled branch features
      (``512 x num_branches``). This is the black-box analogue of
      ProtoECGNet's fusion over prototype similarities.

    Training follows ProtoECGNet's protocol through :meth:`set_stage`:

    1. ``"branches"``: train all branches; the fusion head is bypassed.
    2. ``"fusion"``: freeze the branches (BatchNorm statistics included) and
       train only the fusion head, with an optional L1 penalty
       (``lam_l1``) on weights that connect a branch to labels of *other*
       groups, as in ProtoECGNet. With ``fusion_init="branches"`` (default)
       entering this stage after ``"branches"`` copies the trained branch
       heads into the fusion head (:meth:`init_fusion_from_branches`), so the
       fused model starts at the aggregated branch prediction and only has
       to learn cross-group corrections.

    ``"all"`` trains everything end to end through the fusion head (or the
    scattered branch logits when ``fusion="none"``). With
    ``fusion="features"`` the branch heads are not on that path, so use the
    two-stage protocol to train them.

    Args:
        dataset: A fitted :class:`~pyhealth.datasets.SampleDataset` with one
            ``tensor`` signal input ``(leads, time)`` and one multilabel
            output.
        label_groups: Mapping from branch name to the label codes it
            predicts. Labels absent from the dataset vocabulary are dropped;
            the groups must then cover every label exactly once. Defaults to
            a single branch ``"all"`` with every label (the paper's
            full-label black-box baseline).
        branch_kwargs: Optional per-branch overrides of
            :class:`ResNetBranch` arguments (``backbone``, ``dropout``,
            ``pretrained``, ``base_width``, ``layers``), keyed by branch name.
        in_channels: Number of ECG leads. Defaults to 12.
        backbone: Default backbone of every branch. Defaults to
            ``"resnet1d"``.
        dropout: Default dropout before each branch head. Defaults to 0.3,
            the paper's value.
        base_width: Default width of the 1D backbone. Defaults to 64.
        layers: Default residual blocks per 1D stage. Defaults to
            ``(2, 2, 2, 2)`` (ResNet-18 layout).
        fusion: ``"none"``, ``"logits"`` or ``"features"``. Defaults to
            ``"none"``.
        lam_l1: Weight of the cross-group L1 penalty in the ``"fusion"``
            stage. Defaults to 0.0.
        fusion_init: How the fusion head is initialized, for both
            ``"logits"`` and ``"features"`` fusion. ``"branches"`` starts the
            head at the branch prediction: the identity for ``"logits"``
            (set at construction), and a copy of the branch heads for
            ``"features"`` when the ``"fusion"`` stage starts after
            ``"branches"`` (:meth:`init_fusion_from_branches`). ``"random"``
            keeps PyTorch's default ``nn.Linear`` initialization, as
            ProtoECGNet does. Defaults to ``"branches"``.
        class_weights: Optional positive-class weights ``(num_labels,)``.
            Defaults to the label processor's ``pos_weight``, or ones.

    Raises:
        ValueError: If the dataset does not have exactly one input and one
            multilabel output, or the groups or options are invalid.

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
        >>> model = ECGBranchResNet(
        ...     dataset,
        ...     label_groups={"g0": ["X"], "g1": ["Y"]},
        ...     fusion="features",
        ...     base_width=4,
        ...     layers=(1, 1, 1, 1),
        ... )
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        label_groups: Mapping[str, Sequence[str]] | None = None,
        branch_kwargs: Mapping[str, Mapping[str, object]] | None = None,
        in_channels: int = 12,
        backbone: str = "resnet1d",
        dropout: float = 0.3,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        fusion: str = "none",
        lam_l1: float = 0.0,
        fusion_init: str = "branches",
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__(dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError("ECGBranchResNet expects one signal input and one label")
        if self.mode != "multilabel":
            raise ValueError("ECGBranchResNet requires a multilabel output schema")
        if fusion not in BRANCH_RESNET_FUSIONS:
            raise ValueError(f"fusion must be one of {BRANCH_RESNET_FUSIONS}")
        if lam_l1 < 0:
            raise ValueError("lam_l1 must be non-negative")
        if fusion_init not in ("branches", "random"):
            raise ValueError("fusion_init must be 'branches' or 'random'")
        self.signal_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.fusion = fusion
        self.lam_l1 = lam_l1
        self.fusion_init = fusion_init
        self.stage = "all"

        vocab = self.label_processor.label_vocab
        self.label_names: list[str] = [
            name for name, _ in sorted(vocab.items(), key=lambda item: item[1])
        ]
        self.num_labels = len(self.label_names)
        groups = self._prepare_groups(label_groups)
        unknown = set(branch_kwargs or {}) - set(groups)
        if unknown:
            raise ValueError(f"branch_kwargs has unknown branches: {sorted(unknown)}")
        index_of = {name: i for i, name in enumerate(self.label_names)}

        self.branch_names: list[str] = list(groups)
        self.label_groups: dict[str, list[str]] = groups
        self.branches = nn.ModuleDict()
        for name, members in groups.items():
            self.register_buffer(
                f"index_{name}", torch.tensor([index_of[m] for m in members])
            )
            options: dict[str, object] = {
                "backbone": backbone,
                "dropout": dropout,
                "base_width": base_width,
                "layers": layers,
            }
            options.update((branch_kwargs or {}).get(name, {}))
            self.branches[name] = ResNetBranch(in_channels, len(members), **options)

        self.fusion_classifier: nn.Linear | None = None
        fusion_mask = torch.ones(self.num_labels, 0)
        if fusion != "none":
            fusion_mask = self._build_fusion_mask()
            self.fusion_classifier = nn.Linear(fusion_mask.shape[1], self.num_labels)
            if fusion == "logits" and fusion_init == "branches":
                with torch.no_grad():
                    self.fusion_classifier.weight.copy_(fusion_mask)
                    self.fusion_classifier.bias.zero_()
        # 1 where an input of the fusion head belongs to the label's own group.
        self.register_buffer("fusion_mask", fusion_mask)

        self.register_buffer("class_weights", torch.ones(self.num_labels))
        if class_weights is None:
            class_weights = getattr(self.label_processor, "weight", None)
        if class_weights is not None:
            weight = torch.as_tensor(class_weights).float().reshape(-1)
            if weight.numel() == 1:
                weight = weight.expand(self.num_labels).clone()
            elif weight.numel() != self.num_labels:
                raise ValueError(
                    "class_weights must be a scalar or have one entry per label "
                    f"({self.num_labels}); got {weight.numel()}"
                )
            self.class_weights = weight

    def _prepare_groups(
        self, label_groups: Mapping[str, Sequence[str]] | None
    ) -> dict[str, list[str]]:
        """Drops unknown labels and checks that the groups partition the labels.

        Args:
            label_groups: Requested grouping, or None for a single branch.

        Returns:
            Non-empty groups restricted to the dataset vocabulary.

        Raises:
            ValueError: If a branch name is invalid or the cleaned groups do
                not contain every label exactly once.
        """
        if label_groups is None:
            return {"all": list(self.label_names)}
        known = set(self.label_names)
        cleaned: dict[str, list[str]] = {}
        for name, members in label_groups.items():
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError(
                    f"branch names must be non-empty strings without '.'; got {name!r}"
                )
            kept = [m for m in members if m in known]
            if kept:
                cleaned[name] = kept
        flat = [m for members in cleaned.values() for m in members]
        if sorted(flat) != sorted(self.label_names):
            raise ValueError(
                "label_groups must contain every label exactly once; got "
                f"{len(flat)} known entries for {len(self.label_names)} labels"
            )
        return cleaned

    def _build_fusion_mask(self) -> torch.Tensor:
        """Builds the ``(num_labels, fusion_inputs)`` own-group mask.

        Returns:
            A 0/1 tensor that is 1 where a fusion input comes from the branch
            that owns the label (for ``"logits"`` this is the identity
            permuted into branch order).
        """
        blocks = []
        for name in self.branch_names:
            index = self.branch_index(name)
            width = (
                len(index)
                if self.fusion == "logits"
                else self.branches[name].feature_dim
            )
            block = torch.zeros(self.num_labels, width)
            if self.fusion == "logits":
                block[index, torch.arange(len(index))] = 1.0
            else:
                block[index] = 1.0
            blocks.append(block)
        return torch.cat(blocks, dim=1)

    def branch_index(self, name: str) -> torch.Tensor:
        """Global label indices predicted by branch ``name``.

        Args:
            name: Branch name.

        Returns:
            A long tensor of label indices in the branch's label order.
        """
        return getattr(self, f"index_{name}")

    def set_stage(self, stage: str) -> None:
        """Selects which parameters are trained.

        Args:
            stage: ``"all"``, ``"branches"`` or ``"fusion"``.

        Raises:
            ValueError: If the stage is unknown, or ``"fusion"`` is requested
                without a fusion head.
        """
        if stage not in BRANCH_RESNET_STAGES:
            raise ValueError(f"stage must be one of {BRANCH_RESNET_STAGES}")
        if stage == "fusion" and self.fusion_classifier is None:
            raise ValueError("the 'fusion' stage requires fusion != 'none'")
        for name, param in self.named_parameters():
            is_fusion = name.startswith("fusion_classifier")
            if stage == "all":
                param.requires_grad = True
            elif stage == "branches":
                param.requires_grad = not is_fusion
            else:
                param.requires_grad = is_fusion
        if (
            stage == "fusion"
            and self.stage == "branches"
            and self.fusion_init == "branches"
        ):
            self.init_fusion_from_branches()
        self.stage = stage
        self.train(self.training)

    @torch.no_grad()
    def init_fusion_from_branches(self) -> None:
        """Copies the branch heads into the fusion head.

        Afterwards the fused logits equal the scattered branch logits (in
        eval mode), and every cross-group weight is zero. For
        ``fusion="logits"`` this restores the identity initialization.

        Raises:
            ValueError: If the model has no fusion head.
        """
        if self.fusion_classifier is None:
            raise ValueError("init_fusion_from_branches requires fusion != 'none'")
        weight = torch.zeros_like(self.fusion_classifier.weight)
        bias = torch.zeros_like(self.fusion_classifier.bias)
        offset = 0
        for name in self.branch_names:
            branch = self.branches[name]
            index = self.branch_index(name)
            if self.fusion == "logits":
                width = len(index)
                weight[index, offset + torch.arange(width)] = 1.0
            else:
                width = branch.feature_dim
                weight[index, offset : offset + width] = branch.head.weight
                bias[index] = branch.head.bias
            offset += width
        self.fusion_classifier.weight.copy_(weight)
        self.fusion_classifier.bias.copy_(bias)

    def train(self, mode: bool = True) -> "ECGBranchResNet":
        """Sets training mode, keeping frozen branches in eval mode.

        In the ``"fusion"`` stage the branches stay in eval mode, so their
        BatchNorm statistics are not updated and their dropout is off, and the
        fusion head sees the same branch outputs as at inference time.

        Args:
            mode: Whether to enable training mode.

        Returns:
            The model itself.
        """
        super().train(mode)
        if mode and self.stage == "fusion":
            self.branches.eval()
        return self

    def get_extra_state(self) -> dict:
        """Saves the training stage in ``state_dict``.

        Returns:
            ``{"stage": self.stage}``.
        """
        return {"stage": self.stage}

    def set_extra_state(self, state: object) -> None:
        """Restores the stage saved by :meth:`get_extra_state`.

        Args:
            state: The saved extra state.
        """
        if isinstance(state, dict) and "stage" in state:
            self.set_stage(state["stage"])

    def _uses_fusion_head(self) -> bool:
        """Whether the forward pass routes logits through the fusion head."""
        return self.fusion_classifier is not None and self.stage != "branches"

    def _l1_penalty(self) -> torch.Tensor:
        """L1 norm of fusion weights from a branch to labels of other groups."""
        weight = self.fusion_classifier.weight
        return (weight.abs() * (1.0 - self.fusion_mask)).sum()

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: Must contain the ECG batch ``(batch, leads, time)``
                under the signal key; may contain the multi-hot labels under
                the label key. ``embed=True`` adds the concatenated branch
                features under ``"embed"``.

        Returns:
            A dict with ``logit`` and ``y_prob`` ``(batch, num_labels)``,
            ``branch_logit`` (scattered branch logits before fusion) and, when
            labels are given, ``loss`` and ``y_true``.
        """
        x = kwargs[self.signal_key].to(self.device).float()
        batch = x.shape[0]
        branch_logits = []
        features = []
        scattered = x.new_zeros(batch, self.num_labels)
        for name in self.branch_names:
            logits, feats = self.branches[name](x)
            branch_logits.append(logits)
            features.append(feats)
            scattered = scattered.index_copy(1, self.branch_index(name), logits)

        if self._uses_fusion_head():
            fusion_input = torch.cat(
                branch_logits if self.fusion == "logits" else features, dim=1
            )
            logits = self.fusion_classifier(fusion_input)
        else:
            logits = scattered

        result = {
            "logit": logits,
            "y_prob": torch.sigmoid(logits),
            "branch_logit": scattered,
        }
        if kwargs.get("embed", False):
            result["embed"] = torch.cat(features, dim=1)
        y = kwargs.get(self.label_key)
        if y is not None:
            y = y.to(self.device).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.class_weights
            )
            if self.stage == "fusion" and self.lam_l1 > 0:
                loss = loss + self.lam_l1 * self._l1_penalty()
            result["loss"] = loss
            result["y_true"] = y
        return result
