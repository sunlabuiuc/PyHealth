"""Label-grouped multi-label ECG classification on PTB-XL.

Author: Hanho Ryu (NetID: hanhor2, hanhor2@illinois.edu)

Paper: ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning (Sethi et al., MLHC 2025).
Paper link: https://arxiv.org/abs/2504.08713

ProtoECGNet splits the 71 PTB-XL SCP-ECG statements into three
physician-defined groups (rhythm, morphology, global) and trains one branch
per group. This module makes that grouping a first-class, swappable part of
the task so it can be tested as a hypothesis:

* :class:`PTBXLLabelGroupClassification` builds samples whose labels are
  restricted to a given label grouping, and optionally to one group (the
  per-branch rows of the paper's Table 1).
* :func:`label_cooccurrence` computes the Jaccard label co-occurrence matrix
  that the paper uses for its contrastive loss.
* :func:`spectral_label_groups` derives a data-driven grouping from that
  matrix (listed only as future work in the paper), and
  :func:`random_label_groups` gives a size-matched random control.
* :func:`match_label_groups` maps any grouping onto a reference grouping, so
  that each new group can reuse the branch architecture of the reference
  group it overlaps most.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from .ptbxl_multilabel_classification import (
    PTBXLMultilabelClassification,
    _aggregate_labels,
    _load_scp_statements,
)

#: Physician-defined grouping of the 71 PTB-XL statements used by ProtoECGNet
#: (Appendix G of the paper): 16 rhythm, 52 morphology and 3 global labels.
#: Kept identical to ``pyhealth.models.protoecgnet.PTBXL_BRANCH_LABELS``.
# fmt: off
PHYSICIAN_LABEL_GROUPS: dict[str, list[str]] = {
    "rhythm": [
        "1AVB", "2AVB", "3AVB", "AFIB", "AFLT", "BIGU", "IVCD", "PACE",
        "PSVT", "SARRH", "SBRAD", "SR", "STACH", "SVARR", "SVTAC", "TRIGU",
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
# fmt: on


def validate_label_groups(
    label_groups: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Checks that a label grouping is a partition and normalizes it.

    Args:
        label_groups: Mapping from group name to the label codes in the group.

    Returns:
        A new ``dict`` with the same group order and each group's labels
        sorted, so that equal groupings compare (and cache) equally.

    Raises:
        ValueError: If there are no groups, a group is empty, a group name is
            not a non-empty string or contains ``"."``, or a label appears in
            more than one group.
        TypeError: If a group is given as a single string.

    Examples:
        >>> validate_label_groups({"a": ["Y", "X"], "b": ["Z"]})
        {'a': ['X', 'Y'], 'b': ['Z']}
    """
    if not label_groups:
        raise ValueError("label_groups must contain at least one group")
    seen: dict[str, str] = {}
    cleaned: dict[str, list[str]] = {}
    for name, members in label_groups.items():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(
                f"group names must be non-empty strings without '.'; got {name!r}"
            )
        if isinstance(members, str):
            raise TypeError(f"group {name!r} must be a list of labels, not a str")
        members = list(members)
        if not members:
            raise ValueError(f"group {name!r} is empty")
        for label in members:
            if label in seen:
                raise ValueError(
                    f"label {label!r} is in both group {seen[label]!r} "
                    f"and group {name!r}"
                )
            seen[label] = name
        cleaned[name] = sorted(members)
    return cleaned


def label_cooccurrence(
    label_lists: Iterable[Sequence[str]],
    labels: Sequence[str],
) -> np.ndarray:
    """Jaccard co-occurrence matrix of labels over a set of samples.

    Entry ``(i, j)`` is ``|S_i & S_j| / |S_i | S_j|`` where ``S_i`` is the set
    of samples carrying label ``i``, as in the contrastive loss of ProtoECGNet.
    The diagonal is 1 for labels that occur and 0 for labels that never occur.

    Args:
        label_lists: One list of label codes per sample (e.g. the ``labels``
            field of the training samples). Codes not in ``labels`` are
            ignored.
        labels: The label vocabulary, which fixes the row/column order.

    Returns:
        A float64 array of shape ``(len(labels), len(labels))``.

    Examples:
        >>> label_cooccurrence([["A", "B"], ["A"], ["C"]], ["A", "B", "C"])
        array([[1. , 0.5, 0. ],
               [0.5, 1. , 0. ],
               [0. , 0. , 1. ]])
    """
    index = {label: i for i, label in enumerate(labels)}
    rows = [
        sorted({index[code] for code in sample if code in index})
        for sample in label_lists
    ]
    y = np.zeros((len(rows), len(labels)), dtype=np.float64)
    for row, cols in enumerate(rows):
        y[row, cols] = 1.0
    inter = y.T @ y
    counts = np.diag(inter)
    union = counts[:, None] + counts[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def spectral_label_groups(
    cooccurrence: np.ndarray,
    labels: Sequence[str],
    num_groups: int = 3,
    seed: int = 0,
    smoothing: float = 1e-3,
) -> dict[str, list[str]]:
    """Data-driven label grouping by spectral clustering of co-occurrence.

    The Jaccard matrix is used as a precomputed affinity. A small constant
    ``smoothing`` is added to every off-diagonal entry so that labels that
    never co-occur with anything still form a connected graph.

    Args:
        cooccurrence: Symmetric ``(num_labels, num_labels)`` affinity, usually
            from :func:`label_cooccurrence` on the training split only.
        labels: Label codes in the matrix's row order.
        num_groups: Number of groups (the paper uses 3).
        seed: Random seed of the k-means step.
        smoothing: Constant added to off-diagonal affinities.

    Returns:
        Groups named ``"cluster0"``, ``"cluster1"``, ... ordered by decreasing
        size (ties broken by first label), each with sorted labels.

    Raises:
        ValueError: If the matrix shape does not match ``labels`` or
            ``num_groups`` is not in ``[1, len(labels)]``.

    Examples:
        >>> import numpy as np
        >>> c = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> spectral_label_groups(c, ["A", "B", "C"], num_groups=2)
        {'cluster0': ['A', 'B'], 'cluster1': ['C']}
    """
    affinity = np.asarray(cooccurrence, dtype=np.float64)
    count = len(labels)
    if affinity.shape != (count, count):
        raise ValueError(
            f"cooccurrence has shape {affinity.shape}, expected ({count}, {count})"
        )
    if not 1 <= num_groups <= count:
        raise ValueError(f"num_groups must be in [1, {count}]; got {num_groups}")
    if num_groups == 1:
        return {"cluster0": sorted(labels)}
    from sklearn.cluster import SpectralClustering

    affinity = (affinity + affinity.T) / 2.0
    affinity = affinity + smoothing * (1.0 - np.eye(count))
    np.fill_diagonal(affinity, 1.0)
    assignment = SpectralClustering(
        n_clusters=num_groups,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(affinity)
    clusters = [
        sorted(label for label, a in zip(labels, assignment) if a == k)
        for k in range(num_groups)
    ]
    clusters = [members for members in clusters if members]
    clusters.sort(key=lambda members: (-len(members), members[0]))
    return {f"cluster{k}": members for k, members in enumerate(clusters)}


def random_label_groups(
    labels: Sequence[str],
    sizes: Sequence[int],
    seed: int = 0,
) -> dict[str, list[str]]:
    """Random label partition with fixed group sizes (a control grouping).

    Args:
        labels: Label codes to partition.
        sizes: Size of each group; must sum to ``len(labels)``. Use the sizes
            of the physician grouping to get a size-matched control.
        seed: Seed of the permutation.

    Returns:
        Groups named ``"random0"``, ``"random1"``, ... with the given sizes,
        each with sorted labels.

    Raises:
        ValueError: If ``sizes`` has a non-positive entry or does not sum to
            the number of labels.

    Examples:
        >>> groups = random_label_groups(["A", "B", "C"], sizes=[2, 1], seed=0)
        >>> [len(g) for g in groups.values()]
        [2, 1]
    """
    if any(size < 1 for size in sizes):
        raise ValueError("every group size must be positive")
    if sum(sizes) != len(labels):
        raise ValueError(
            f"sizes sum to {sum(sizes)} but there are {len(labels)} labels"
        )
    order = np.random.default_rng(seed).permutation(len(labels))
    shuffled = [labels[i] for i in order]
    groups: dict[str, list[str]] = {}
    start = 0
    for k, size in enumerate(sizes):
        groups[f"random{k}"] = sorted(shuffled[start : start + size])
        start += size
    return groups


def match_label_groups(
    label_groups: Mapping[str, Sequence[str]],
    reference: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Matches each group to the reference group it overlaps most.

    The matching is one-to-one when there are at most as many groups as
    reference groups (Hungarian assignment on label overlap, ties broken by
    group size); extra groups then fall back to their best-overlap reference.
    It is used to give data-driven or random groups the branch architecture
    of the physician group they resemble, so that a grouping ablation changes
    only *which labels* share a branch, not the branch types.

    Args:
        label_groups: Grouping to match.
        reference: Reference grouping, e.g. :data:`PHYSICIAN_LABEL_GROUPS`.

    Returns:
        Mapping from each group name in ``label_groups`` to a reference name.

    Examples:
        >>> match_label_groups(
        ...     {"g0": ["A", "B"], "g1": ["C"]},
        ...     {"big": ["A", "B", "D"], "small": ["C"]},
        ... )
        {'g0': 'big', 'g1': 'small'}
    """
    from scipy.optimize import linear_sum_assignment

    names = list(label_groups)
    ref_names = list(reference)
    if not names or not ref_names:
        raise ValueError("both groupings must contain at least one group")
    overlap = np.array(
        [
            [len(set(label_groups[n]) & set(reference[r])) for r in ref_names]
            for n in names
        ],
        dtype=np.float64,
    )
    # Break ties in favour of reference groups of similar size.
    size_gap = np.array(
        [
            [abs(len(label_groups[n]) - len(reference[r])) for r in ref_names]
            for n in names
        ],
        dtype=np.float64,
    )
    score = overlap - 1e-3 * size_gap
    match: dict[str, str] = {}
    if len(names) <= len(ref_names):
        rows, cols = linear_sum_assignment(-score)
        for row, col in zip(rows, cols):
            match[names[row]] = ref_names[col]
    for row, name in enumerate(names):
        if name not in match:
            match[name] = ref_names[int(score[row].argmax())]
    return {name: match[name] for name in names}


class PTBXLLabelGroupClassification(PTBXLMultilabelClassification):
    """Multi-label PTB-XL ECG classification under a configurable label grouping.

    Each ECG yields a ``(12, 10 * fs)`` signal and the subset of its SCP-ECG
    statements that belong to ``label_groups``. With ``group`` set, only that
    group's labels are kept and ECGs without any of them are skipped, which
    reproduces the per-branch training sets of ProtoECGNet (Table 1 of the
    paper). The grouping itself is a task parameter, so the physician
    grouping of the paper can be compared with data-driven
    (:func:`spectral_label_groups`) and random (:func:`random_label_groups`)
    groupings.

    Each sample also carries ``groups``, the names of the groups that are
    present in the ECG, for per-group analysis.

    Compatible with :class:`~pyhealth.datasets.PTBXLDataset`.

    The task subclasses :class:`~pyhealth.tasks.PTBXLMultilabelClassification`
    on purpose: it reuses that task's PTB-XL signal pipeline (record loading,
    crop/pad to 10 s, optional 0.5 Hz high-pass filter and per-lead z-score)
    and its SCP-statement aggregation with ``label_set="all"``, so there is a
    single preprocessing implementation for PTB-XL. Only the label selection
    differs, and labels are checked before a signal is loaded, so ECGs
    without any kept label cost no I/O.

    Args:
        scp_statements_path: Path to ``scp_statements.csv`` (use
            ``dataset.scp_statements_path``).
        label_groups: Mapping from group name to SCP codes. Defaults to
            :data:`PHYSICIAN_LABEL_GROUPS` (all 71 statements).
        group: Optional name of a single group to keep. Defaults to None
            (keep every group).
        min_likelihood: Minimum statement likelihood to keep (see
            :class:`~pyhealth.tasks.PTBXLMultilabelClassification`).
            Defaults to 0.0.
        standardize: ``"record"`` (per-lead z-score) or None. Defaults to
            ``"record"``.
        remove_baseline: Apply the paper's 0.5 Hz high-pass filter. Defaults
            to True, as in ProtoECGNet.

    Raises:
        ValueError: If ``label_groups`` is not a valid partition (see
            :func:`validate_label_groups`) or ``group`` is not one of its
            names.

    Attributes:
        task_name (str): ``"PTBXLLabelGroupClassification"``.
        input_schema (dict): ``{"signal": "tensor"}``.
        output_schema (dict): ``{"labels": "multilabel"}``.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLLabelGroupClassification
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl/1.0.3")  # doctest: +SKIP
        >>> task = PTBXLLabelGroupClassification(  # doctest: +SKIP
        ...     scp_statements_path=dataset.scp_statements_path,
        ...     group="rhythm",
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> sorted(samples[0])  # doctest: +SKIP
        ['groups', 'labels', 'patient_id', 'signal', 'strat_fold']
    """

    task_name: str = "PTBXLLabelGroupClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"labels": "multilabel"}

    def __init__(
        self,
        scp_statements_path: str,
        label_groups: Mapping[str, Sequence[str]] | None = None,
        group: str | None = None,
        min_likelihood: float = 0.0,
        standardize: str | None = "record",
        remove_baseline: bool = True,
    ) -> None:
        groups = validate_label_groups(
            PHYSICIAN_LABEL_GROUPS if label_groups is None else label_groups
        )
        if group is not None and group not in groups:
            raise ValueError(f"group must be one of {list(groups)}; got {group!r}")
        super().__init__(
            scp_statements_path=scp_statements_path,
            label_set="all",
            min_likelihood=min_likelihood,
            standardize=standardize,
            remove_baseline=remove_baseline,
        )
        # Stored as instance attributes so that they are part of the task's
        # cache key: a different grouping or group gets its own sample cache.
        self.label_groups = groups
        self.group = group
        kept = groups if group is None else {group: groups[group]}
        self._group_of = {
            label: name for name, members in kept.items() for label in members
        }

    @property
    def labels(self) -> list[str]:
        """Sorted label codes that this task can emit.

        Examples:
            >>> task = PTBXLLabelGroupClassification(  # doctest: +SKIP
            ...     "scp_statements.csv", group="global"
            ... )
            >>> task.labels  # doctest: +SKIP
            ['DIG', 'EL', 'NORM']
        """
        return sorted(self._group_of)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per ECG that has at least one kept label.

        Args:
            patient: A patient with ``records`` events from
                :class:`~pyhealth.datasets.PTBXLDataset`.

        Returns:
            A list of dicts with ``patient_id``, ``strat_fold``, ``signal``
            (float32 ``(12, T)``), ``labels`` (sorted kept SCP codes) and
            ``groups`` (sorted names of the groups present).
        """
        from pyhealth.datasets.ptbxl import parse_scp_codes

        statements = _load_scp_statements(self.scp_statements_path)
        samples = []
        for event in patient.get_events(event_type="records"):
            codes = _aggregate_labels(
                parse_scp_codes(event.scp_codes),
                statements,
                self.label_set,
                self.min_likelihood,
            )
            labels = sorted({code for code in codes if code in self._group_of})
            if not labels:
                # Skip before loading the signal: no I/O for unused ECGs.
                continue
            sampling_rate = int(event.sampling_rate)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "strat_fold": int(float(event.strat_fold)),
                    "signal": self._prepare_signal(event.signal_file, sampling_rate),
                    "labels": labels,
                    "groups": sorted({self._group_of[code] for code in labels}),
                }
            )
        return samples
