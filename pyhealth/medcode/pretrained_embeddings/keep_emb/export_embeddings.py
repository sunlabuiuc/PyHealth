"""Export KEEP embeddings to PyHealth's text-vector format.

Converts SNOMED-level KEEP embeddings (numpy array keyed by concept_id)
into the GloVe-style text file that PyHealth's ``init_embedding_with_pretrained``
reads. Each line is: ``token vec[0] vec[1] ... vec[N-1]``.

The primary export is ``keep_snomed.txt`` with SNOMED concept codes as tokens.
This is used with ``code_mapping=("ICD9CM", "SNOMED")`` in PyHealth tasks.

Also supports fallback exports (``keep_icd9.txt``, ``keep_ccs.txt``) for
users who prefer Option B (no SNOMED code_mapping). See
``docs/plans/keep/keep-export-design-decision.md`` for the full discussion.

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def export_to_text(
    embeddings: np.ndarray,
    tokens: List[str],
    output_path: str | Path,
) -> Path:
    """Write embeddings to a GloVe-style text file.

    Produces the format that ``init_embedding_with_pretrained()`` reads:
    one line per token, each line is ``token float float float ...``.

    Args:
        embeddings: np.ndarray of shape ``(N, dim)``.
        tokens: List of N token strings, same order as embedding rows.
        output_path: Path to write the text file.

    Returns:
        Path to the written file.

    Example:
        >>> export_to_text(embeddings, ["84114007", "38341003", ...],
        ...                "keep_snomed.txt")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, token in enumerate(tokens):
            vec_str = " ".join(f"{v:.6f}" for v in embeddings[i])
            f.write(f"{token} {vec_str}\n")

    logger.info(
        "Exported %d embeddings (dim=%d) to %s",
        len(tokens),
        embeddings.shape[1] if embeddings.ndim > 1 else 0,
        output_path,
    )
    return output_path


def export_snomed(
    embeddings: np.ndarray,
    node_ids: List[int],
    graph: nx.DiGraph,
    output_path: str | Path,
) -> Path:
    """Export KEEP embeddings with SNOMED concept codes as tokens.

    This is the primary export format for Option A (SNOMED as code_mapping
    target). Used with ``code_mapping=("ICD9CM", "SNOMED")``.

    Tokens are SNOMED concept code strings (e.g., "84114007" for Heart
    failure), NOT integer concept_ids.

    Args:
        embeddings: KEEP embedding matrix, shape ``(N, dim)``.
        node_ids: List of SNOMED concept_ids (integers), same order as
            embedding rows. From ``train_node2vec`` or ``train_keep``.
        graph: SNOMED hierarchy DiGraph with ``concept_code`` node
            attributes.
        output_path: Path to write ``keep_snomed.txt``.

    Returns:
        Path to the written file.

    Example:
        >>> export_snomed(embeddings, node_ids, graph, "keep_snomed.txt")
        >>> # Use with:
        >>> # model = GRASP(code_mapping=("ICD9CM", "SNOMED"),
        >>> #               pretrained_emb_path="keep_snomed.txt")
    """
    tokens = []
    for nid in node_ids:
        code = str(graph.nodes[nid].get("concept_code", nid))
        tokens.append(code)

    return export_to_text(embeddings, tokens, output_path)


def export_icd(
    embeddings: np.ndarray,
    node_ids: List[int],
    icd_to_snomed: Dict[str, int],
    output_path: str | Path,
) -> Path:
    """Export KEEP embeddings with ICD codes as tokens (Option B fallback).

    Maps each SNOMED concept back to ICD codes and writes the embedding
    under the ICD code token. 1:1 mapping (lossless). For use without
    code_mapping.

    When multiple ICD codes map to the same SNOMED concept, each ICD
    code gets a copy of that concept's embedding.

    Args:
        embeddings: KEEP embedding matrix, shape ``(N, dim)``.
        node_ids: List of SNOMED concept_ids (integers).
        icd_to_snomed: Mapping from ICD code strings to SNOMED concept_ids.
        output_path: Path to write the file.

    Returns:
        Path to the written file.
    """
    # Reverse the mapping: snomed_id -> list of ICD codes
    snomed_to_icd: Dict[int, List[str]] = {}
    for icd_code, snomed_id in icd_to_snomed.items():
        snomed_to_icd.setdefault(snomed_id, []).append(icd_code)

    # Build node_id -> embedding index
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    tokens = []
    vecs = []
    for snomed_id, icd_codes in snomed_to_icd.items():
        if snomed_id in nid_to_idx:
            idx = nid_to_idx[snomed_id]
            for icd_code in icd_codes:
                tokens.append(icd_code)
                vecs.append(embeddings[idx])

    if vecs:
        emb_matrix = np.stack(vecs)
    else:
        emb_matrix = np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    return export_to_text(emb_matrix, tokens, output_path)


def export_all(
    embeddings: np.ndarray,
    node_ids: List[int],
    graph: nx.DiGraph,
    icd9_to_snomed: Dict[str, int],
    output_dir: str | Path,
    icd10_to_snomed: Optional[Dict[str, int]] = None,
) -> Dict[str, Path]:
    """Export KEEP embeddings in all supported formats.

    Produces:
        - ``keep_snomed.txt``: SNOMED concept codes as tokens (primary, Option A)
        - ``keep_icd9.txt``: ICD-9 codes as tokens (fallback, Option B)
        - ``keep_icd10.txt``: ICD-10 codes as tokens (fallback, Option B)

    Args:
        embeddings: KEEP embedding matrix, shape ``(N, dim)``.
        node_ids: List of SNOMED concept_ids.
        graph: SNOMED hierarchy DiGraph.
        icd9_to_snomed: ICD-9 to SNOMED mapping.
        output_dir: Directory to write all files.
        icd10_to_snomed: ICD-10 to SNOMED mapping. Optional.

    Returns:
        Dict mapping format name to output path.

    Example:
        >>> paths = export_all(embeddings, node_ids, graph,
        ...                    icd9_map, "output/", icd10_map)
        >>> paths["snomed"]
        PosixPath('output/keep_snomed.txt')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    paths["snomed"] = export_snomed(
        embeddings, node_ids, graph,
        output_dir / "keep_snomed.txt",
    )

    paths["icd9"] = export_icd(
        embeddings, node_ids, icd9_to_snomed,
        output_dir / "keep_icd9.txt",
    )

    if icd10_to_snomed is not None:
        paths["icd10"] = export_icd(
            embeddings, node_ids, icd10_to_snomed,
            output_dir / "keep_icd10.txt",
        )

    logger.info("Exported all formats to %s: %s", output_dir, list(paths.keys()))
    return paths
