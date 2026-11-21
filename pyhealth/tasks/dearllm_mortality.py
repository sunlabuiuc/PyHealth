"""Mortality prediction tasks and graph helpers for the DearLLM model.

Contributor: Kavish Shah (GitHub: thekavishshah)

Paper: Yongxin Xu et al. DearLLM: Enhancing Personalized Healthcare via
Large Language Models-Deduced Feature Correlations. AAAI 2025.
https://doi.org/10.1609/aaai.v39i1.32079

Description:
    The paper's setting (Section 4.1): patients with at least two visits;
    the outcome of the last visit is the mortality label and the earlier
    visits are the input. Each patient gets one sample with:

    * ``conditions``: diagnosis codes of each input visit.
    * ``graph_nodes``: the distinct codes of the most recent
      ``graph_visits`` input visits (the paper uses 5); the nodes of the
      patient's LLM-deduced feature prior graph.
    * ``graph_node_weights``: the CF-IPF score of each node (paper Eq. 5-6).
    * ``graph_edges``: ``[source, target, weight]`` rows from a precomputed
      graph (see :func:`normalize_perplexity_graphs`).
    * ``mortality``: 1 if the patient died during the last visit.

    Building the graphs needs an LLM and is done once, offline, before
    training; ``examples/mortality_prediction/dearllm_mimic3.py`` shows the
    whole pipeline. Without graphs the task still works (every patient gets
    an edgeless graph), which is useful to compute IPF statistics first.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

#: Edge row used for a patient without edges. Its weight is 0, so it adds
#: nothing to the adjacency matrix; it only keeps the tensor shape (E, 3).
EMPTY_EDGE = [0, 0, 0.0]


def compute_ipf(patient_codes: Mapping[str, Iterable[str]]) -> dict[str, float]:
    """Inverse patient frequency of every code (paper Eq. 5).

    ``IPF(c) = log(P / (1 + P_c))`` where ``P`` is the number of patients
    and ``P_c`` the number of patients with code ``c``. Compute it on the
    training patients only.

    Args:
        patient_codes: Patient id -> all codes in that patient's history.

    Returns:
        Code -> IPF value.

    Examples:
        >>> ipf = compute_ipf({"p0": ["a", "b"], "p1": ["a"], "p2": ["c"]})
        >>> round(ipf["b"], 4)
        0.4055
    """
    num_patients = len(patient_codes)
    patient_freq: dict[str, int] = {}
    for codes in patient_codes.values():
        for code in set(codes):
            patient_freq[code] = patient_freq.get(code, 0) + 1
    return {
        code: math.log(num_patients / (1 + count))
        for code, count in patient_freq.items()
    }


def compute_cf_ipf(
    history_codes: list[str], nodes: list[str], ipf: Mapping[str, float] | None
) -> list[float]:
    """CF-IPF score of each node for one patient (paper Eq. 5-6).

    ``CF(c) = (# times c appears in the history) / (# codes in the history)``
    and ``CF-IPF(c) = CF(c) * IPF(c)``.

    Args:
        history_codes: All codes of the patient's input visits, with
            repeats.
        nodes: The graph nodes to score.
        ipf: Code -> IPF from :func:`compute_ipf`. Codes not in it (unseen
            in training) get the largest IPF in it. If None, the IPF factor
            is 1, i.e. only CF is used.

    Returns:
        One score per node.

    Examples:
        >>> compute_cf_ipf(["a", "a", "b", "c"], ["a", "b"], {"a": 1.0, "b": 2.0})
        [0.5, 0.5]
    """
    total = len(history_codes)
    if total == 0:
        return [0.0] * len(nodes)
    counts: dict[str, int] = {}
    for code in history_codes:
        counts[code] = counts.get(code, 0) + 1
    unseen_ipf = max(ipf.values()) if ipf else 1.0
    scores = []
    for code in nodes:
        cf = counts.get(code, 0) / total
        idf = 1.0 if ipf is None else ipf.get(code, unseen_ipf)
        scores.append(cf * idf)
    return scores


def normalize_perplexity_graphs(
    graphs: Mapping[str, Iterable[tuple[str, str, float]]],
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    bounds: tuple[float, float] | None = None,
) -> tuple[dict[str, list[tuple[str, str, float]]], tuple[float, float]]:
    """Turns conditional perplexities into edge weights in ``[0, 1]``.

    Paper Eq. 3: perplexities are clipped to ``[cppl_min, cppl_max]`` (the
    ``lower_quantile`` and ``upper_quantile`` of all training perplexities)
    and mapped to ``1 - (cppl - cppl_min) / (cppl_max - cppl_min)``, so a
    low perplexity (strong correlation) becomes a high weight. Edges with
    weight 0 (perplexity at or above ``cppl_max``) are pruned, as are
    non-finite perplexities (failed LLM calls), as in the official code.

    Compute ``bounds`` on the training graphs and pass them back in for the
    validation and test graphs.

    Args:
        graphs: Patient id -> ``(code_i, code_j, perplexity)`` triples for
            the directed edge ``code_i -> code_j``.
        lower_quantile: Quantile for ``cppl_min``. Default is 0.1.
        upper_quantile: Quantile for ``cppl_max``. Default is 0.9.
        bounds: ``(cppl_min, cppl_max)`` to use instead of computing them.

    Returns:
        ``(normalized_graphs, (cppl_min, cppl_max))``. Every patient id of
        ``graphs`` is kept, possibly with an empty edge list.

    Examples:
        >>> graphs = {"p0": [("a", "b", 1.0), ("b", "a", 3.0), ("a", "c", 5.0)]}
        >>> norm, b = normalize_perplexity_graphs(graphs, bounds=(1.0, 5.0))
        >>> norm["p0"]
        [('a', 'b', 1.0), ('b', 'a', 0.5)]
    """
    if bounds is None:
        values = [
            float(ppl)
            for edges in graphs.values()
            for _, _, ppl in edges
            if math.isfinite(float(ppl))
        ]
        if not values:
            raise ValueError("No finite perplexity values to compute bounds.")
        lower = float(np.quantile(values, lower_quantile))
        upper = float(np.quantile(values, upper_quantile))
    else:
        lower, upper = bounds
    span = upper - lower

    normalized: dict[str, list[tuple[str, str, float]]] = {}
    for patient_id, edges in graphs.items():
        kept = []
        for src, dst, ppl in edges:
            ppl = float(ppl)
            if not math.isfinite(ppl):
                continue
            if span <= 0:
                scaled = 0.0 if ppl <= lower else 1.0
            else:
                scaled = min(max((ppl - lower) / span, 0.0), 1.0)
            weight = 1.0 - scaled
            if weight > 0.0:
                kept.append((src, dst, weight))
        normalized[str(patient_id)] = kept
    return normalized, (lower, upper)


class _DearLLMMortalityPrediction(BaseTask):
    """Shared logic of the MIMIC-III and MIMIC-IV DearLLM tasks."""

    input_schema: ClassVar[dict[str, str]] = {
        "conditions": "nested_sequence",
        "graph_nodes": "sequence",
        "graph_node_weights": "tensor",
        "graph_edges": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    #: Attribute holding the diagnosis code on a ``diagnoses_icd`` event.
    code_attribute: ClassVar[str] = "icd_code"

    def __init__(
        self,
        graphs: Mapping[str, Iterable[tuple[str, str, float]]] | None = None,
        ipf: Mapping[str, float] | None = None,
        graph_visits: int = 5,
        drop_empty_graphs: bool = False,
    ):
        """Creates the task.

        Args:
            graphs: Patient id -> ``(code_i, code_j, weight)`` edges, already
                normalised (see :func:`normalize_perplexity_graphs`). Edges
                whose codes are not graph nodes are ignored. If None, every
                patient gets an edgeless graph.
            ipf: Code -> IPF from :func:`compute_ipf` (training patients).
                If None, node weights are CF only.
            graph_visits: Number of most recent input visits whose codes
                become graph nodes. Default is 5 (paper).
            drop_empty_graphs: Skip patients whose graph has no edges, as
                the official code does. Default is False.
        """
        super().__init__()
        if graph_visits < 1:
            raise ValueError("graph_visits must be at least 1.")
        self.graphs = (
            None if graphs is None else {str(k): list(v) for k, v in graphs.items()}
        )
        self.ipf = ipf
        self.graph_visits = graph_visits
        self.drop_empty_graphs = drop_empty_graphs

    def _visit_codes(self, patient: Any, admission: Any) -> list[str]:
        """Distinct diagnosis codes of one admission, in recorded order."""
        events = patient.get_events(
            event_type="diagnoses_icd",
            filters=[("hadm_id", "==", admission.hadm_id)],
        )
        codes: list[str] = []
        for event in events:
            code = getattr(event, self.code_attribute, None)
            if code is None:
                continue
            code = str(code).strip()
            if code and code not in codes:
                codes.append(code)
        return codes

    def extract_history(self, patient: Any) -> dict[str, Any] | None:
        """Visits, visit times and label of a patient (no graph fields).

        This is what the LLM step needs to build the prompt, and what
        :func:`compute_ipf` needs.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            None if the patient has fewer than two admissions or no coded
            input visit; otherwise a dict with ``visits`` (codes per input
            visit), ``timestamps`` (admission time per input visit) and
            ``mortality``.

        Examples:
            >>> task = DearLLMMortalityPredictionMIMIC3()
            >>> history = task.extract_history(patient)  # doctest: +SKIP
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return None
        flag = admissions[-1].hospital_expire_flag
        mortality = int(flag) if flag in (0, 1, "0", "1") else 0
        visits, timestamps = [], []
        for admission in admissions[:-1]:
            codes = self._visit_codes(patient, admission)
            if codes:
                visits.append(codes)
                timestamps.append(admission.timestamp)
        if not visits:
            return None
        return {"visits": visits, "timestamps": timestamps, "mortality": mortality}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the single DearLLM sample of a patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            A list with zero or one sample.

        Examples:
            >>> task = DearLLMMortalityPredictionMIMIC3()
            >>> samples = task(patient)  # doctest: +SKIP
        """
        history = self.extract_history(patient)
        if history is None:
            return []
        visits = history["visits"]

        nodes: list[str] = []
        for visit in visits[-self.graph_visits :]:
            for code in visit:
                if code not in nodes:
                    nodes.append(code)
        position = {code: i for i, code in enumerate(nodes)}

        edges = []
        patient_edges = (self.graphs or {}).get(str(patient.patient_id), [])
        for src, dst, weight in patient_edges:
            if src in position and dst in position and src != dst:
                edges.append([position[src], position[dst], float(weight)])
        if not edges:
            if self.drop_empty_graphs:
                return []
            edges = [list(EMPTY_EDGE)]

        all_codes = [code for visit in visits for code in visit]
        return [
            {
                "patient_id": patient.patient_id,
                "conditions": visits,
                "graph_nodes": nodes,
                "graph_node_weights": compute_cf_ipf(all_codes, nodes, self.ipf),
                "graph_edges": edges,
                "mortality": history["mortality"],
            }
        ]


class DearLLMMortalityPredictionMIMIC3(_DearLLMMortalityPrediction):
    """DearLLM mortality prediction on MIMIC-III (ICD-9 diagnoses).

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset` loaded with the
    ``diagnoses_icd`` table. See the module docstring for the sample fields.

    Args:
        graphs: Normalised patient graphs; see
            :func:`normalize_perplexity_graphs`. Default is None.
        ipf: Code -> IPF from :func:`compute_ipf`. Default is None.
        graph_visits: Most recent visits used for graph nodes. Default is 5.
        drop_empty_graphs: Skip patients without edges. Default is False.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import DearLLMMortalityPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4", tables=["diagnoses_icd"]
        ... )  # doctest: +SKIP
        >>> task = DearLLMMortalityPredictionMIMIC3(graphs=graphs, ipf=ipf)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "DearLLMMortalityPredictionMIMIC3"
    code_attribute: ClassVar[str] = "icd9_code"


class DearLLMMortalityPredictionMIMIC4(_DearLLMMortalityPrediction):
    """DearLLM mortality prediction on MIMIC-IV (ICD-9 and ICD-10 diagnoses).

    Works with :class:`~pyhealth.datasets.MIMIC4Dataset` (EHR) loaded with
    the ``diagnoses_icd`` table. See the module docstring for the sample
    fields. The paper also removes MIMIC-IV patients that overlap with
    MIMIC-III; this task does not, since the overlap cannot be identified
    from MIMIC-IV alone.

    Args:
        graphs: Normalised patient graphs; see
            :func:`normalize_perplexity_graphs`. Default is None.
        ipf: Code -> IPF from :func:`compute_ipf`. Default is None.
        graph_visits: Most recent visits used for graph nodes. Default is 5.
        drop_empty_graphs: Skip patients without edges. Default is False.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import DearLLMMortalityPredictionMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimiciv/2.2", ehr_tables=["diagnoses_icd"]
        ... )  # doctest: +SKIP
        >>> task = DearLLMMortalityPredictionMIMIC4(graphs=graphs, ipf=ipf)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "DearLLMMortalityPredictionMIMIC4"
    code_attribute: ClassVar[str] = "icd_code"
