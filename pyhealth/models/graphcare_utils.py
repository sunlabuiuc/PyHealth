"""GraphCare data pipeline utilities for PyHealth.

This module provides utilities to convert PyHealth ``SampleDataset`` records
and pre-built knowledge graph artifacts into ``torch_geometric`` Data objects
suitable for the :class:`~pyhealth.models.graphcare.GraphCare` model.

The pipeline mirrors the original GraphCare implementation
(https://github.com/pat-jj/GraphCare) but is refactored for cleaner
integration with PyHealth.

Typical usage::

    from pyhealth.models.graphcare_utils import (
        load_kg_artifacts,
        prepare_graphcare_data,
        build_graphcare_dataloaders,
    )

    # 1. Load pre-built KG artifacts
    artifacts = load_kg_artifacts(
        sample_dataset_path="sample_dataset_mimic3_mortality_th015.pkl",
        graph_path="graph_mimic3_mortality_th015.pkl",
        ent_emb_path="entity_embedding.pkl",
        rel_emb_path="relation_embedding.pkl",
        cluster_path="clusters_th015.json",
        cluster_rel_path="clusters_rel_th015.json",
        ccscm_id2clus_path="ccscm_id2clus.json",
        ccsproc_id2clus_path="ccsproc_id2clus.json",
    )

    # 2. Prepare PyG-compatible data
    prepared = prepare_graphcare_data(artifacts, task="mortality")

    # 3. Build dataloaders
    train_loader, val_loader, test_loader = build_graphcare_dataloaders(
        prepared, batch_size=64,
    )
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import pickle
import logging

import torch
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

# Lazy import torch_geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.utils import from_networkx, k_hop_subgraph

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def _check_torch_geometric():
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "GraphCare data utilities require torch-geometric. "
            "Install with: pip install torch-geometric"
        )


# ===========================================================================
# 1. Loading pre-built KG artifacts
# ===========================================================================


def load_kg_artifacts(
    sample_dataset_path: str,
    graph_path: str,
    ent_emb_path: str,
    rel_emb_path: str,
    cluster_path: str,
    cluster_rel_path: str,
    ccscm_id2clus_path: str,
    ccsproc_id2clus_path: str,
    atc3_id2clus_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load all pre-built KG artifacts required by GraphCare.

    These artifacts are produced by the GraphCare KG construction pipeline
    (LLM prompting + subgraph sampling + node/edge clustering).  See the
    original repo for generation scripts.

    Args:
        sample_dataset_path: Path to pickled PyHealth SampleDataset list.
        graph_path: Path to pickled NetworkX graph (the global KG).
        ent_emb_path: Path to pickled entity (node) embeddings.
        rel_emb_path: Path to pickled relation embeddings.
        cluster_path: Path to JSON node cluster mapping.
        cluster_rel_path: Path to JSON relation cluster mapping.
        ccscm_id2clus_path: Path to JSON CCS-CM code → cluster ID mapping.
        ccsproc_id2clus_path: Path to JSON CCS-Proc code → cluster ID mapping.
        atc3_id2clus_path: Optional path to JSON ATC3 drug code → cluster ID
            mapping.  Required for mortality/readmission tasks that include
            drug features.

    Returns:
        Dictionary with keys: ``sample_dataset``, ``graph``, ``ent_emb``,
        ``rel_emb``, ``cluster_map``, ``cluster_rel_map``,
        ``ccscm_id2clus``, ``ccsproc_id2clus``, ``atc3_id2clus``.
    """
    def _load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    artifacts = {
        "sample_dataset": _load_pkl(sample_dataset_path),
        "graph": _load_pkl(graph_path),
        "ent_emb": _load_pkl(ent_emb_path),
        "rel_emb": _load_pkl(rel_emb_path),
        "cluster_map": _load_json(cluster_path),
        "cluster_rel_map": _load_json(cluster_rel_path),
        "ccscm_id2clus": _load_json(ccscm_id2clus_path),
        "ccsproc_id2clus": _load_json(ccsproc_id2clus_path),
        "atc3_id2clus": _load_json(atc3_id2clus_path) if atc3_id2clus_path else None,
    }

    logger.info(
        f"Loaded KG artifacts: {len(artifacts['sample_dataset'])} patients, "
        f"graph has {artifacts['graph'].number_of_nodes()} nodes / "
        f"{artifacts['graph'].number_of_edges()} edges"
    )
    return artifacts


# ===========================================================================
# 2. Labelling & subgraph extraction
# ===========================================================================


def _flatten(lst):
    """Recursively flatten nested lists."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def label_ehr_nodes(
    sample_dataset: List[Dict],
    task: str,
    num_nodes: int,
    ccscm_id2clus: Dict[str, str],
    ccsproc_id2clus: Dict[str, str],
    atc3_id2clus: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Add ``ehr_node_set`` (one-hot) to each patient record.

    Maps each patient's conditions, procedures, and (optionally) drugs
    to their cluster node IDs and creates a binary indicator vector.

    Args:
        sample_dataset: List of patient sample dicts from PyHealth.
        task: Task name — ``"mortality"``, ``"readmission"``, ``"drugrec"``,
            or ``"lenofstay"``.
        num_nodes: Total number of cluster nodes in the KG.
        ccscm_id2clus: CCS-CM condition code → cluster ID mapping.
        ccsproc_id2clus: CCS-Proc procedure code → cluster ID mapping.
        atc3_id2clus: ATC3 drug code → cluster ID mapping (required for
            mortality/readmission).

    Returns:
        The same dataset list, with ``ehr_node_set`` added to each record.
    """
    for patient in sample_dataset:
        nodes = []

        for condition in _flatten(patient["conditions"]):
            if condition in ccscm_id2clus:
                ehr_node = int(ccscm_id2clus[condition])
                nodes.append(ehr_node)
                patient["node_set"].append(ehr_node)

        for procedure in _flatten(patient["procedures"]):
            if procedure in ccsproc_id2clus:
                ehr_node = int(ccsproc_id2clus[procedure])
                nodes.append(ehr_node)
                patient["node_set"].append(ehr_node)

        if task in ("mortality", "readmission") and atc3_id2clus is not None:
            for drug in _flatten(patient.get("drugs", [])):
                if drug in atc3_id2clus:
                    ehr_node = int(atc3_id2clus[drug])
                    nodes.append(ehr_node)
                    patient["node_set"].append(ehr_node)

        node_vec = np.zeros(num_nodes)
        if nodes:
            node_vec[nodes] = 1
        patient["ehr_node_set"] = torch.tensor(node_vec)

    return sample_dataset


def get_rel_emb_from_clusters(cluster_rel_map: Dict) -> torch.Tensor:
    """Extract relation embeddings from the cluster relation mapping.

    Args:
        cluster_rel_map: Dict mapping relation cluster ID (str) to
            a dict containing ``"embedding"`` key.

    Returns:
        Tensor of shape ``[num_rels, emb_dim]``.
    """
    rel_emb = []
    for i in range(len(cluster_rel_map)):
        rel_emb.append(cluster_rel_map[str(i)]["embedding"][0])
    return torch.tensor(np.array(rel_emb))


def extract_patient_subgraph(
    G_tg: "Data",
    patient: Dict,
    task: str,
    k_hop: int = 2,
) -> "Data":
    """Extract a patient-specific subgraph from the global KG.

    Uses k-hop subgraph extraction centred on the patient's node set,
    then attaches task labels and visit/EHR node metadata.

    Args:
        G_tg: The global KG as a PyG ``Data`` object.
        patient: A single patient record dict with keys ``node_set``,
            ``visit_padded_node``, ``ehr_node_set``, ``patient_id``,
            and task-specific label fields.
        task: Task name for label extraction.
        k_hop: Number of hops for subgraph extraction.  Default ``2``.

    Returns:
        A PyG ``Data`` object representing the patient's subgraph with
        attached metadata.
    """
    _check_torch_geometric()

    node_set = patient["node_set"]
    if len(node_set) == 0:
        # Return a minimal graph if no nodes
        P = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            y=torch.zeros(0, dtype=torch.long),
            relation=torch.zeros(0, dtype=torch.long),
        )
    else:
        nodes, _, _, edge_mask = k_hop_subgraph(
            torch.tensor(node_set), k_hop, G_tg.edge_index
        )
        mask_idx = torch.where(edge_mask)[0]
        L = G_tg.edge_subgraph(mask_idx)
        P = L.subgraph(torch.tensor(node_set))

    # Attach label
    if task == "drugrec":
        P.label = patient["drugs_ind"]
    elif task == "lenofstay":
        label = np.zeros(10)
        label[patient["label"]] = 1
        P.label = torch.tensor(label)
    else:  # mortality, readmission
        P.label = patient["label"]

    # Attach visit and EHR node info
    P.visit_padded_node = patient["visit_padded_node"]
    P.ehr_nodes = patient["ehr_node_set"]
    P.patient_id = patient["patient_id"]

    return P


# ===========================================================================
# 3. Dataset & DataLoader
# ===========================================================================


class GraphCareDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that lazily extracts patient subgraphs.

    Each ``__getitem__`` call extracts the k-hop subgraph for one patient
    from the global KG, attaches labels and metadata, and returns a
    ``torch_geometric.data.Data`` object.

    Args:
        G_tg: Global KG as a PyG Data object.
        dataset: List of patient record dicts.
        task: Task name.
        k_hop: Number of hops for subgraph extraction.
    """

    def __init__(self, G_tg, dataset: List[Dict], task: str, k_hop: int = 2):
        self.G_tg = G_tg
        self.dataset = dataset
        self.task = task
        self.k_hop = k_hop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patient = self.dataset[idx]
        # Skip patients with empty node sets by falling back
        while len(patient["node_set"]) == 0 and idx > 0:
            idx -= 1
            patient = self.dataset[idx]
        return extract_patient_subgraph(
            self.G_tg, patient, self.task, self.k_hop
        )


# ===========================================================================
# 4. High-level prepare & build functions
# ===========================================================================


def get_task_config(task: str, sample_dataset: List[Dict]) -> Dict[str, Any]:
    """Get task-specific configuration.

    Args:
        task: One of ``"mortality"``, ``"readmission"``, ``"drugrec"``,
            ``"lenofstay"``.

    Returns:
        Dict with ``mode``, ``out_channels``, and ``loss_fn``.
    """
    import torch.nn.functional as F_

    if task in ("mortality", "readmission"):
        return {
            "mode": "binary",
            "out_channels": 1,
            "loss_fn": F_.binary_cross_entropy_with_logits,
        }
    elif task == "drugrec":
        return {
            "mode": "multilabel",
            "out_channels": len(sample_dataset[0]["drugs_ind"]),
            "loss_fn": F_.binary_cross_entropy_with_logits,
        }
    elif task == "lenofstay":
        return {
            "mode": "multiclass",
            "out_channels": 10,
            "loss_fn": F_.cross_entropy,
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def _split_by_patient(
    dataset: List[Dict],
    ratios: List[float],
    seed: int = 528,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split patient records into train/val/test by patient ID.

    Args:
        dataset: List of patient record dicts (must have ``"patient_id"``).
        ratios: Three-element list of train/val/test ratios (must sum to 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) record lists.
    """
    import random as _rng

    patient_ids = sorted(set(p["patient_id"] for p in dataset))
    _rng.seed(seed)
    _rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    train = [p for p in dataset if p["patient_id"] in train_ids]
    val = [p for p in dataset if p["patient_id"] in val_ids]
    test = [p for p in dataset if p["patient_id"] in test_ids]

    return train, val, test


def prepare_graphcare_data(
    artifacts: Dict[str, Any],
    task: str,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 528,
) -> Dict[str, Any]:
    """Prepare all data needed for GraphCare training.

    Takes the raw artifacts from :func:`load_kg_artifacts` and produces:

    * Labelled patient records with ``ehr_node_set``
    * The global KG converted to PyG format
    * Node and relation embedding tensors
    * Task configuration (mode, out_channels, loss_fn)
    * Train/val/test splits

    Args:
        artifacts: Output of :func:`load_kg_artifacts`.
        task: Task name.
        split_ratios: Train/val/test split ratios.
        seed: Random seed for splitting.

    Returns:
        Dict with keys: ``G_tg``, ``node_emb``, ``rel_emb``,
        ``num_nodes``, ``num_rels``, ``max_visit``, ``task_config``,
        ``train_dataset``, ``val_dataset``, ``test_dataset``, ``task``.
    """
    _check_torch_geometric()

    sample_dataset = artifacts["sample_dataset"]
    graph = artifacts["graph"]
    cluster_map = artifacts["cluster_map"]
    cluster_rel_map = artifacts["cluster_rel_map"]

    # Label EHR nodes
    num_cluster_nodes = len(cluster_map)
    sample_dataset = label_ehr_nodes(
        sample_dataset,
        task=task,
        num_nodes=num_cluster_nodes,
        ccscm_id2clus=artifacts["ccscm_id2clus"],
        ccsproc_id2clus=artifacts["ccsproc_id2clus"],
        atc3_id2clus=artifacts["atc3_id2clus"],
    )

    # Convert NetworkX graph to PyG
    G_tg = from_networkx(graph)

    # Embeddings
    rel_emb = get_rel_emb_from_clusters(cluster_rel_map)
    node_emb = G_tg.x

    # Task config
    task_config = get_task_config(task, sample_dataset)

    # Split
    train_dataset, val_dataset, test_dataset = _split_by_patient(
        sample_dataset, list(split_ratios), seed=seed
    )

    max_visit = sample_dataset[0]["visit_padded_node"].shape[0]

    return {
        "G_tg": G_tg,
        "node_emb": node_emb,
        "rel_emb": rel_emb,
        "num_nodes": node_emb.shape[0],
        "num_rels": rel_emb.shape[0],
        "max_visit": max_visit,
        "task_config": task_config,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "task": task,
    }


def build_graphcare_dataloaders(
    prepared: Dict[str, Any],
    batch_size: int = 64,
    k_hop: int = 2,
) -> Tuple["PyGDataLoader", "PyGDataLoader", "PyGDataLoader"]:
    """Build PyG DataLoaders for GraphCare.

    Args:
        prepared: Output of :func:`prepare_graphcare_data`.
        batch_size: Batch size.
        k_hop: Number of hops for subgraph extraction.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    _check_torch_geometric()

    G_tg = prepared["G_tg"]
    task = prepared["task"]

    train_set = GraphCareDataset(G_tg, prepared["train_dataset"], task, k_hop)
    val_set = GraphCareDataset(G_tg, prepared["val_dataset"], task, k_hop)
    test_set = GraphCareDataset(G_tg, prepared["test_dataset"], task, k_hop)

    train_loader = PyGDataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = PyGDataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = PyGDataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_loader, val_loader, test_loader


def reshape_batch_tensors(
    data: "Batch",
    batch_size: int,
    max_visit: int,
    num_nodes: int,
    patient_mode: str = "joint",
) -> Dict[str, torch.Tensor]:
    """Reshape batched tensors from PyG DataLoader for GraphCare forward pass.

    The PyG DataLoader concatenates per-graph tensors.  This function
    reshapes them back into the shapes expected by ``GraphCare.forward()``.

    Args:
        data: A batched ``torch_geometric.data.Batch`` object.
        batch_size: Number of graphs in the batch.
        max_visit: Maximum visits per patient.
        num_nodes: Total KG nodes (for reshaping visit_padded_node).
        patient_mode: Patient mode to determine if ehr_nodes is needed.

    Returns:
        Dict with keys ``node_ids``, ``rel_ids``, ``edge_index``,
        ``batch``, ``visit_node``, ``ehr_nodes``, ``label``.
    """
    result = {
        "node_ids": data.y,
        "rel_ids": data.relation,
        "edge_index": data.edge_index,
        "batch": data.batch,
        "visit_node": data.visit_padded_node.reshape(
            batch_size, max_visit, num_nodes
        ).float(),
    }

    if patient_mode != "graph":
        result["ehr_nodes"] = data.ehr_nodes.reshape(
            batch_size, num_nodes
        ).float()
    else:
        result["ehr_nodes"] = None

    result["label"] = data.label.reshape(
        batch_size, -1
    )

    return result