from typing import Dict, List, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

ICD10_CHAPTER_MAP: Dict[str, str] = {
    "A": "A00-B99",
    "B": "A00-B99",
    "C": "C00-D49",
    "D": "C00-D49",  # D50-D89 is a separate chapter; handled below
    "E": "E00-E89",
    "F": "F01-F99",
    "G": "G00-G99",
    "H": "H00-H59",  # H60-H95 is a separate chapter; handled below
    "I": "I00-I99",
    "J": "J00-J99",
    "K": "K00-K95",
    "L": "L00-L99",
    "M": "M00-M99",
    "N": "N00-N99",
    "O": "O00-O9A",
    "P": "P00-P96",
    "Q": "Q00-Q99",
    "R": "R00-R99",
    "S": "S00-T88",
    "T": "S00-T88",
    "V": "V00-Y99",
    "W": "V00-Y99",
    "X": "V00-Y99",
    "Y": "V00-Y99",
    "Z": "Z00-Z99",
    "U": "U00-U85",
}

# D and H each span two chapters, split by numeric part.
_D_SPLIT = 50
_H_SPLIT = 60


def _get_icd10_chapter(code: str) -> str:
    """Map an ICD-10-CM code to its chapter range (e.g. "E11.321" -> "E00-E89")."""
    if not code or not code[0].isalpha():
        raise ValueError(f"Invalid ICD-10 code: {code}")

    first = code[0].upper()
    if first == "D":
        numeric = int(code[1:3]) if len(code) >= 3 and code[1:3].isdigit() else 0
        return "D50-D89" if numeric >= _D_SPLIT else "C00-D49"
    if first == "H":
        numeric = int(code[1:3]) if len(code) >= 3 and code[1:3].isdigit() else 0
        return "H60-H95" if numeric >= _H_SPLIT else "H00-H59"

    chapter = ICD10_CHAPTER_MAP.get(first)
    if chapter is None:
        raise ValueError(f"Unknown ICD-10 chapter for code: {code}")
    return chapter


def _get_icd10_category(code: str) -> str:
    """First 3 characters of the code (e.g. "E11.321" -> "E11")."""
    return code.replace(".", "")[:3].upper()


def build_icd10_hierarchy(code_list: List[str]) -> Dict:
    """Build a 3-level ICD-10-CM hierarchy (chapter -> category -> full code).

    Args:
        code_list: ICD-10-CM codes from the dataset label vocabulary.

    Returns:
        Dict with depth_to_codes, code_to_index, parent_to_children,
        and child_to_parent mappings for the decoder.
    """
    if not code_list:
        raise ValueError("code_list must not be empty")

    full_codes = sorted(set(code_list))
    chapters = sorted({_get_icd10_chapter(c) for c in full_codes})
    categories = sorted({_get_icd10_category(c) for c in full_codes})

    depth_to_codes: Dict[int, List[str]] = {
        0: chapters,
        1: categories,
        2: full_codes,
    }
    code_to_index: Dict[int, Dict[str, int]] = {
        d: {code: idx for idx, code in enumerate(codes)}
        for d, codes in depth_to_codes.items()
    }

    parent_to_children: Dict[int, Dict[int, List[int]]] = {0: {}, 1: {}}
    child_to_parent: Dict[int, Dict[int, int]] = {1: {}, 2: {}}

    for cat in categories:
        chapter = _get_icd10_chapter(cat)
        p_idx = code_to_index[0][chapter]
        c_idx = code_to_index[1][cat]
        parent_to_children[0].setdefault(p_idx, []).append(c_idx)
        child_to_parent[1][c_idx] = p_idx

    for fc in full_codes:
        cat = _get_icd10_category(fc)
        p_idx = code_to_index[1][cat]
        c_idx = code_to_index[2][fc]
        parent_to_children[1].setdefault(p_idx, []).append(c_idx)
        child_to_parent[2][c_idx] = p_idx

    return {
        "depth_to_codes": depth_to_codes,
        "code_to_index": code_to_index,
        "parent_to_children": parent_to_children,
        "child_to_parent": child_to_parent,
    }


class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss for sparse multi-label classification (Ben-Baruch et al., 2020).

    Args:
        gamma_neg: Focusing parameter for negatives.
        gamma_pos: Focusing parameter for positives.
        clip: Probability clipping threshold for negatives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)

        loss_pos = targets * torch.clamp(
            (1 - p).pow(self.gamma_pos) * torch.log(p + 1e-8),
            min=-100,
        )

        p_neg = (p - self.clip).clamp(min=1e-8)
        loss_neg = (1 - targets) * torch.clamp(
            p_neg.pow(self.gamma_neg) * torch.log(1 - p_neg + 1e-8),
            min=-100,
        )

        return -(loss_pos + loss_neg).mean()


class _ResidualConvBlock(nn.Module):
    """Single convolutional channel with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class MultiResCNNEncoder(nn.Module):
    """Parallel Conv1d branches with different kernel sizes, merged via 1x1 conv.

    Args:
        input_dim: Input channels (embedding dimension).
        num_filter_maps: Output channels after merge.
        kernel_sizes: Kernel sizes for parallel branches.
    """

    def __init__(
        self,
        input_dim: int,
        num_filter_maps: int,
        kernel_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9, 15, 19, 25]
        self.branches = nn.ModuleList(
            [_ResidualConvBlock(input_dim, num_filter_maps, ks) for ks in kernel_sizes]
        )
        self.merge = nn.Conv1d(
            num_filter_maps * len(kernel_sizes), num_filter_maps, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        return self.merge(concatenated)


class HierarchicalDecoder(nn.Module):
    """Per-label attention decoder with curriculum weight transfer between depths.

    Args:
        num_filter_maps: Encoder output channels.
        depth_sizes: Number of codes at each depth.
        child_to_parent: Child-to-parent index mapping per depth.
    """

    def __init__(
        self,
        num_filter_maps: int,
        depth_sizes: List[int],
        child_to_parent: Dict[int, Dict[int, int]],
    ):
        super().__init__()
        self.num_filter_maps = num_filter_maps
        self.depth_sizes = depth_sizes
        self.child_to_parent = child_to_parent
        self.current_depth = len(depth_sizes) - 1  # default: finest

        self.attention = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for size in depth_sizes:
            self.attention.append(nn.Linear(num_filter_maps, size))
            self.classifiers.append(nn.Linear(num_filter_maps, size))

    def set_depth(self, depth: int) -> None:
        """Switch active depth and copy parent weights to child positions."""
        if depth < 0 or depth >= len(self.depth_sizes):
            raise ValueError(
                f"depth must be in [0, {len(self.depth_sizes) - 1}], got {depth}"
            )
        self.current_depth = depth

        if depth == 0:
            return  # Nothing to transfer at the coarsest level.

        c2p = self.child_to_parent.get(depth, {})
        if not c2p:
            return

        parent_attn = self.attention[depth - 1]
        parent_cls = self.classifiers[depth - 1]
        child_attn = self.attention[depth]
        child_cls = self.classifiers[depth]

        with torch.no_grad():
            for child_idx, parent_idx in c2p.items():
                child_attn.weight.data[child_idx] = parent_attn.weight.data[parent_idx]
                child_attn.bias.data[child_idx] = parent_attn.bias.data[parent_idx]
                child_cls.weight.data[child_idx] = parent_cls.weight.data[parent_idx]
                child_cls.bias.data[child_idx] = parent_cls.bias.data[parent_idx]

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        d = self.current_depth

        # Per-label attention
        attn_scores = self.attention[d](encoded.transpose(1, 2))  # (B, S, C)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = attn_weights.transpose(1, 2) @ encoded.transpose(1, 2)  # (B, C, F)

        # Per-label classification via element-wise multiply + sum
        logits = (context * self.classifiers[d].weight).sum(dim=2) + self.classifiers[d].bias
        return logits


class HiCu(BaseModel):
    """HiCu: Hierarchical Curriculum Learning for ICD coding.

    MultiResCNN encoder + per-label attention decoder with 3-level ICD-10
    hierarchy (chapter -> category -> full code). Call set_depth() between
    training stages to transfer weights from coarse to fine codes.

    Paper: Ren et al., ML4H 2022. https://arxiv.org/abs/2208.02301

    Args:
        dataset: SampleDataset with multilabel ICD-10-CM output.
        num_filter_maps: CNN output channels.
        embedding_dim: Word embedding dimension.
        kernel_sizes: Kernel sizes for the multi-resolution CNN.
        asl_gamma_neg: ASL focusing parameter for negatives.
        asl_gamma_pos: ASL focusing parameter for positives.
        asl_clip: ASL probability clipping threshold.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_filter_maps: int = 50,
        embedding_dim: int = 100,
        kernel_sizes: Optional[List[int]] = None,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 1.0,
        asl_clip: float = 0.05,
        **kwargs,
    ):
        super(HiCu, self).__init__(dataset=dataset)

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9, 15, 19, 25]

        assert len(self.label_keys) == 1, "HiCu supports exactly one label key"
        self.label_key = self.label_keys[0]
        assert len(self.feature_keys) == 1, "HiCu expects exactly one text feature"
        self.text_key = self.feature_keys[0]

        self.num_filter_maps = num_filter_maps
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes

        label_processor = self.dataset.output_processors[self.label_key]
        label_vocab = list(label_processor.label_vocab.keys())
        self.hierarchy = build_icd10_hierarchy(label_vocab)

        depth_sizes = [len(self.hierarchy["depth_to_codes"][d]) for d in range(3)]
        self.depth_sizes = depth_sizes

        input_processor = self.dataset.input_processors[self.text_key]
        vocab_size = len(input_processor.code_vocab)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = MultiResCNNEncoder(embedding_dim, num_filter_maps, kernel_sizes)
        self.decoder = HierarchicalDecoder(
            num_filter_maps, depth_sizes, self.hierarchy["child_to_parent"]
        )
        self.asl_loss = AsymmetricLoss(asl_gamma_neg, asl_gamma_pos, asl_clip)

        self._build_label_mappings()
        self.current_depth = 2

    def _build_label_mappings(self) -> None:
        hierarchy = self.hierarchy
        full_codes = hierarchy["depth_to_codes"][2]
        full_idx = hierarchy["code_to_index"][2]

        for d in range(2):
            n_full = len(full_codes)
            n_depth = len(hierarchy["depth_to_codes"][d])
            mapping = torch.zeros(n_full, n_depth)

            for fc in full_codes:
                fi = full_idx[fc]
                if d == 0:
                    ancestor = _get_icd10_chapter(fc)
                else:
                    ancestor = _get_icd10_category(fc)
                ai = hierarchy["code_to_index"][d][ancestor]
                mapping[fi, ai] = 1.0

            self.register_buffer(f"_label_map_{d}", mapping)

    def _remap_labels(self, y_true: torch.Tensor, depth: int) -> torch.Tensor:
        if depth == 2:
            return y_true
        mapping = getattr(self, f"_label_map_{depth}")
        return (y_true @ mapping).clamp(max=1.0)

    def set_depth(self, depth: int) -> None:
        """Switch hierarchy depth (0=chapters, 1=categories, 2=full codes) and transfer weights."""
        self.current_depth = depth
        self.decoder.set_depth(depth)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        text = kwargs[self.text_key]
        if isinstance(text, tuple):
            text = text[0]
        text = text.to(self.device)

        embedded = self.word_embedding(text)
        encoded = self.encoder(embedded.permute(0, 2, 1))
        logits = self.decoder(encoded)

        y_true_full = kwargs[self.label_key].to(self.device).float()
        y_true = self._remap_labels(y_true_full, self.current_depth)
        loss = self.asl_loss(logits, y_true)
        y_prob = torch.sigmoid(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
