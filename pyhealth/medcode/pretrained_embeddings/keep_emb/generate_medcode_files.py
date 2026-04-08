"""Generate SNOMED vocabulary and cross-map files for PyHealth's medcode system.

This module bridges the KEEP pipeline with PyHealth's medcode infrastructure.
It generates three CSV files from Athena OMOP data and places them in PyHealth's
medcode cache directory so that ``InnerMap.load("SNOMED")`` and
``CrossMap.load("ICD9CM", "SNOMED")`` work automatically.

Generated files:
    - ``SNOMED.csv``: InnerMap vocabulary file with columns
      (code, name, parent_code). Enables ``InnerMap.load("SNOMED")``.
    - ``ICD9CM_to_SNOMED.csv``: CrossMap file with columns
      (ICD9CM, SNOMED). Enables ``CrossMap.load("ICD9CM", "SNOMED")``.
    - ``ICD10CM_to_SNOMED.csv``: CrossMap file with columns
      (ICD10CM, SNOMED). Enables ``CrossMap.load("ICD10CM", "SNOMED")``.

These files are generated from Athena OMOP vocabulary data that the user
has already downloaded. They are placed in ``~/.cache/pyhealth/medcode/``
(the same location PyHealth uses for all medcode data).

Why not host on GCS?
    PyHealth's standard medcode files are hosted on Google Cloud Storage.
    SNOMED files cannot be hosted there because IHTSDO licensing restricts
    redistribution of SNOMED content. Instead, users download Athena
    vocabularies directly from https://athena.ohdsi.org/ (free account,
    select SNOMED + ICD9CM + ICD10CM) and the KEEP pipeline generates
    the medcode CSVs locally.

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Set

import networkx as nx
import pandas as pd

from pyhealth.medcode.utils import MODULE_CACHE_PATH

logger = logging.getLogger(__name__)


def generate_snomed_csv(
    graph: nx.DiGraph,
    output_dir: Optional[str] = None,
) -> Path:
    """Generate SNOMED.csv for PyHealth's InnerMap system.

    Creates a CSV with columns (code, name, parent_code) from the
    SNOMED hierarchy graph built by ``build_omop_graph.build_hierarchy_graph``.

    InnerMap expects:
        - ``code``: the node identifier (SNOMED concept code string)
        - ``name``: human-readable concept name
        - ``parent_code``: code of the parent node (for hierarchy edges)

    For SNOMED, we use the SNOMED ``concept_code`` attribute (e.g., "84114007")
    as the code, since this is what OMOP's CONCEPT.csv stores and what
    the co-occurrence matrix / embeddings are keyed by.

    Note: SNOMED is a polyhierarchy (multiple parents possible). InnerMap
    only supports one parent_code per row. For nodes with multiple parents,
    we use the first parent. The full hierarchy is still available via the
    KEEP graph for Node2Vec walks.

    Args:
        graph: SNOMED hierarchy DiGraph from ``build_hierarchy_graph``.
            Nodes are integer concept_ids with ``concept_name`` and
            ``concept_code`` attributes.
        output_dir: Directory to write the CSV. Defaults to PyHealth's
            medcode cache (``~/.cache/pyhealth/medcode/``).

    Returns:
        Path to the generated SNOMED.csv file.

    Example:
        >>> from pyhealth.medcode.pretrained_embeddings.keep_emb import (
        ...     build_omop_graph, generate_medcode_files,
        ... )
        >>> G = build_omop_graph.build_hierarchy_graph(...)
        >>> path = generate_medcode_files.generate_snomed_csv(G)
        >>> from pyhealth.medcode import InnerMap
        >>> snomed = InnerMap.load("SNOMED", refresh_cache=True)
    """
    if output_dir is None:
        output_dir = MODULE_CACHE_PATH

    rows = []
    for node in graph.nodes():
        attrs = graph.nodes[node]
        concept_code = str(attrs.get("concept_code", node))
        concept_name = attrs.get("concept_name", "")

        # Get first parent (InnerMap only supports single parent).
        # Sort for reproducibility — SNOMED is a polyhierarchy so a
        # node can have multiple parents. Without sorting, the choice
        # depends on edge insertion order which varies across Athena versions.
        parents = sorted(graph.successors(node))
        if parents:
            parent_code = str(
                graph.nodes[parents[0]].get("concept_code", parents[0])
            )
        else:
            parent_code = ""

        rows.append({
            "code": concept_code,
            "name": concept_name,
            "parent_code": parent_code if parent_code else pd.NA,
        })

    df = pd.DataFrame(rows)
    output_path = Path(output_dir) / "SNOMED.csv"
    df.to_csv(output_path, index=False)

    logger.info(
        "Generated SNOMED.csv: %d concepts -> %s",
        len(df),
        output_path,
    )
    return output_path


def generate_crossmap_csv(
    icd_to_snomed: Dict[str, int],
    graph: nx.DiGraph,
    source_vocabulary: str = "ICD9CM",
    output_dir: Optional[str] = None,
) -> Path:
    """Generate a CrossMap CSV (e.g., ICD9CM_to_SNOMED.csv).

    Creates a CSV with columns (source_vocabulary, SNOMED) mapping
    ICD codes to SNOMED concept codes. This enables
    ``CrossMap.load("ICD9CM", "SNOMED")``.

    Args:
        icd_to_snomed: Mapping from ICD code strings to SNOMED
            concept_ids (integers), from ``build_icd_to_snomed``.
        graph: SNOMED hierarchy DiGraph. Used to look up concept_code
            attributes for each SNOMED concept_id.
        source_vocabulary: Source vocabulary name for the CSV column
            header. Default: "ICD9CM".
        output_dir: Directory to write the CSV. Defaults to PyHealth's
            medcode cache.

    Returns:
        Path to the generated CSV file.

    Example:
        >>> path = generate_crossmap_csv(
        ...     icd9_to_snomed, graph, source_vocabulary="ICD9CM",
        ... )
        >>> from pyhealth.medcode import CrossMap
        >>> mapping = CrossMap.load("ICD9CM", "SNOMED", refresh_cache=True)
        >>> mapping.map("428.0")
        ['84114007']
    """
    if output_dir is None:
        output_dir = MODULE_CACHE_PATH

    # Build concept_id -> concept_code lookup from graph
    id_to_code = {}
    for node in graph.nodes():
        id_to_code[node] = str(
            graph.nodes[node].get("concept_code", node)
        )

    rows = []
    for icd_code, snomed_id in icd_to_snomed.items():
        snomed_code = id_to_code.get(snomed_id)
        if snomed_code is not None:
            rows.append({
                source_vocabulary: icd_code,
                "SNOMED": snomed_code,
            })

    df = pd.DataFrame(rows)
    filename = f"{source_vocabulary}_to_SNOMED.csv"
    output_path = Path(output_dir) / filename
    df.to_csv(output_path, index=False)

    logger.info(
        "Generated %s: %d mappings -> %s",
        filename,
        len(df),
        output_path,
    )
    return output_path


def generate_all_medcode_files(
    graph: nx.DiGraph,
    icd9_to_snomed: Dict[str, int],
    icd10_to_snomed: Optional[Dict[str, int]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Generate all medcode files needed for SNOMED code_mapping support.

    This is the main entry point. Call this after building the SNOMED
    graph and ICD-to-SNOMED mappings, and before using
    ``code_mapping=("ICD9CM", "SNOMED")`` in a task.

    Generates:
        - SNOMED.csv (InnerMap vocabulary)
        - ICD9CM_to_SNOMED.csv (CrossMap)
        - ICD10CM_to_SNOMED.csv (CrossMap, if icd10_to_snomed provided)

    After calling this function, clear the pickle cache by passing
    ``refresh_cache=True`` to ``InnerMap.load`` and ``CrossMap.load``
    on first use.

    Args:
        graph: SNOMED hierarchy DiGraph from ``build_hierarchy_graph``.
        icd9_to_snomed: ICD-9 to SNOMED mapping from ``build_icd_to_snomed``.
        icd10_to_snomed: ICD-10 to SNOMED mapping. Optional.
        output_dir: Directory for output files. Defaults to PyHealth's
            medcode cache.

    Example:
        >>> from pyhealth.medcode.pretrained_embeddings.keep_emb import (
        ...     build_omop_graph, generate_medcode_files,
        ... )
        >>> G = build_omop_graph.build_hierarchy_graph(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ... )
        >>> icd9_map, icd10_map = build_omop_graph.build_all_mappings(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ...     snomed_concept_ids=set(G.nodes()),
        ... )
        >>> generate_medcode_files.generate_all_medcode_files(
        ...     G, icd9_map, icd10_map,
        ... )
        >>> # Now these work:
        >>> from pyhealth.medcode import InnerMap, CrossMap
        >>> snomed = InnerMap.load("SNOMED", refresh_cache=True)
        >>> mapping = CrossMap.load("ICD9CM", "SNOMED", refresh_cache=True)
    """
    generate_snomed_csv(graph, output_dir)
    generate_crossmap_csv(
        icd9_to_snomed, graph,
        source_vocabulary="ICD9CM",
        output_dir=output_dir,
    )
    if icd10_to_snomed is not None:
        generate_crossmap_csv(
            icd10_to_snomed, graph,
            source_vocabulary="ICD10CM",
            output_dir=output_dir,
        )

    logger.info("All medcode files generated. Use refresh_cache=True on first load.")
