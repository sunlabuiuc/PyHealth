"""End-to-end KEEP embedding pipeline.

Convenience function that runs all KEEP stages in order:
  1. Build SNOMED hierarchy graph from Athena CSVs
  2. Build ICD-to-SNOMED mappings
  3. Generate medcode files (SNOMED.csv, cross-maps)
  4. Extract patient diagnosis codes from a PyHealth dataset
  5. Roll up codes to ancestor concepts
  6. Build co-occurrence matrix
  7. Train Node2Vec (Stage 1)
  8. Train regularized GloVe (Stage 2)
  9. Export keep_snomed.txt

Usage::

    from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
        run_keep_pipeline,
    )

    keep_emb_path = run_keep_pipeline(
        athena_dir="data/athena",
        dataset=base_dataset,
        output_dir="keep_output",
    )

    model = GRASP(
        dataset=samples,
        pretrained_emb_path=keep_emb_path,
    )

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def detect_mimic_schema(available_columns: List[str]) -> str:
    """Detect MIMIC schema from available dataframe columns.

    Returns "mimic4" if the column list contains ``diagnoses_icd/icd_version``
    (MIMIC-IV's mixed ICD-9/ICD-10 schema), otherwise "mimic3"
    (ICD-9-only schema).

    Args:
        available_columns: List of column names from
            ``dataset.global_event_df.collect().columns`` after filtering to
            diagnoses rows.

    Returns:
        "mimic4" or "mimic3".

    Examples:
        >>> detect_mimic_schema([
        ...     "patient_id",
        ...     "diagnoses_icd/icd_code",
        ...     "diagnoses_icd/icd_version",
        ... ])
        'mimic4'
        >>> detect_mimic_schema(["patient_id", "diagnoses_icd/icd9_code"])
        'mimic3'
    """
    if "diagnoses_icd/icd_version" in available_columns:
        return "mimic4"
    return "mimic3"


def extract_diagnoses_for_schema(
    diag_df,
    icd9_map: Dict[str, List[int]],
    icd10_map: Optional[Dict[str, List[int]]] = None,
    min_occurrences: int = 2,
) -> Dict[str, Set[int]]:
    """Extract per-patient SNOMED code sets from a diagnosis dataframe.

    Dispatches to the correct ICD mapping path based on the dataframe's
    column names. Used by ``run_keep_pipeline`` but exposed as a standalone
    function so callers with custom dataset loaders can use the same
    auto-detection logic.

    Args:
        diag_df: Polars DataFrame with patient_id + diagnosis code columns.
            Must have either:
              - ``patient_id`` + ``diagnoses_icd/icd9_code`` (MIMIC-III), or
              - ``patient_id`` + ``diagnoses_icd/icd_code`` +
                ``diagnoses_icd/icd_version`` (MIMIC-IV).
        icd9_map: ICD-9 → SNOMED concept ID mapping.
        icd10_map: ICD-10 → SNOMED mapping. Required for MIMIC-IV.
        min_occurrences: Minimum times a SNOMED code must appear per patient.

    Returns:
        Dict mapping patient_id to set of SNOMED IDs meeting the
        min_occurrences threshold.

    Raises:
        ValueError: If MIMIC-IV schema is detected but ``icd10_map`` is None.
        KeyError: If required columns are missing.
    """
    from pyhealth.medcode.codes.icd9cm import ICD9CM
    from pyhealth.medcode.codes.icd10cm import ICD10CM
    from .build_cooccurrence import extract_patient_codes_from_df

    schema = detect_mimic_schema(diag_df.columns)

    if schema == "mimic4":
        if icd10_map is None:
            raise ValueError(
                "MIMIC-IV schema detected (diagnoses_icd/icd_version column "
                "present) but icd10_map is None. Pass both icd9_map and "
                "icd10_map for MIMIC-IV datasets."
            )
        return extract_patient_codes_from_df(
            patient_id_col=diag_df["patient_id"].to_list(),
            code_col=diag_df["diagnoses_icd/icd_code"].to_list(),
            version_col=diag_df["diagnoses_icd/icd_version"].to_list(),
            icd_to_snomed=icd9_map,
            icd10_to_snomed=icd10_map,
            standardize_icd9=ICD9CM.standardize,
            standardize_icd10=ICD10CM.standardize,
            min_occurrences=min_occurrences,
        )

    # MIMIC-III path
    return extract_patient_codes_from_df(
        patient_id_col=diag_df["patient_id"].to_list(),
        code_col=diag_df["diagnoses_icd/icd9_code"].to_list(),
        icd_to_snomed=icd9_map,
        standardize_icd9=ICD9CM.standardize,
        min_occurrences=min_occurrences,
    )


def run_keep_pipeline(
    athena_dir: str | Path,
    dataset,
    output_dir: str | Path = "keep_output",
    embedding_dim: int = 100,
    max_depth: int = 5,
    num_walks: int = 750,
    walk_length: int = 30,
    glove_epochs: int = 300,
    lambd: float = 1e-3,
    reg_distance: str = "l2",
    reg_reduction: str = "mean",
    optimizer: str = "adamw",
    min_occurrences: int = 2,
    device: str = "cpu",
    seed: int = 42,
    dev: bool = False,
) -> str:
    """Run the full KEEP embedding pipeline.

    Takes Athena vocabulary files and a PyHealth dataset, produces
    ``keep_snomed.txt`` ready for ``pretrained_emb_path``.

    Also generates SNOMED medcode files so that
    ``code_mapping=("ICD9CM", "SNOMED")`` works in task definitions.

    Default hyperparameters are from the KEEP paper (Appendix A.2-A.3,
    Tables 5-6).

    Args:
        athena_dir: Path to directory containing Athena OMOP vocabulary
            files (CONCEPT.csv, CONCEPT_RELATIONSHIP.csv). Download
            from https://athena.ohdsi.org/ with SNOMED + ICD9CM + ICD10CM.
        dataset: A PyHealth BaseDataset (e.g., MIMIC3Dataset) that has
            been loaded with ``tables=["DIAGNOSES_ICD"]``. The
            ``global_event_df`` is used to extract patient codes.
        output_dir: Directory to write output files. Default: "keep_output".
        embedding_dim: Dimensionality of embeddings.
            Default: 100 (KEEP paper Table 5).
        max_depth: Maximum SNOMED hierarchy depth from root.
            Default: 5 (KEEP paper Section 4.2).
        num_walks: Number of Node2Vec walks per node.
            Default: 750 (KEEP paper Table 5). Use 10 for dev/testing.
        walk_length: Length of each random walk.
            Default: 30 (KEEP paper Table 5).
        glove_epochs: Number of GloVe training epochs.
            Default: 300 (KEEP paper Table 6). Use 10 for dev/testing.
        lambd: GloVe regularization strength (lambda).
            Default: 1e-3 (KEEP paper Table 6).
        min_occurrences: Minimum times a code must appear per patient.
            Default: 2 (KEEP paper Appendix A.4).
        device: Device for GloVe training ("cpu" or "cuda").
            Default: "cpu".
        seed: Random seed for reproducibility. Default: 42.
        dev: If True, use reduced params for fast testing
            (num_walks=10, glove_epochs=10). Default: False.

    Returns:
        str: Path to the exported ``keep_snomed.txt`` file.

    Raises:
        FileNotFoundError: If Athena CSV files are not found.

    Example:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(root="...", tables=["DIAGNOSES_ICD"])
        >>> path = run_keep_pipeline("data/athena", dataset)
        >>> path
        'keep_output/keep_snomed.txt'
    """
    import polars as pl

    from .build_omop_graph import build_hierarchy_graph, build_all_mappings
    from .generate_medcode_files import generate_all_medcode_files
    from .train_node2vec import train_node2vec
    from .build_cooccurrence import (
        rollup_codes,
        build_cooccurrence_matrix,
        apply_count_filter,
    )
    from .train_glove import train_keep
    from .export_embeddings import export_snomed

    athena_dir = Path(athena_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support both .csv and .csv.gz (compressed Athena files)
    concept_csv = athena_dir / "CONCEPT.csv"
    if not concept_csv.exists():
        concept_csv = athena_dir / "CONCEPT.csv.gz"
    relationship_csv = athena_dir / "CONCEPT_RELATIONSHIP.csv"
    if not relationship_csv.exists():
        relationship_csv = athena_dir / "CONCEPT_RELATIONSHIP.csv.gz"
    # CONCEPT_ANCESTOR is optional but enables orphan rescue (paper-faithful)
    ancestor_csv = athena_dir / "CONCEPT_ANCESTOR.csv"
    if not ancestor_csv.exists():
        ancestor_csv_gz = athena_dir / "CONCEPT_ANCESTOR.csv.gz"
        ancestor_csv = ancestor_csv_gz if ancestor_csv_gz.exists() else None

    if not concept_csv.exists():
        raise FileNotFoundError(
            f"CONCEPT.csv(.gz) not found in {athena_dir}. "
            "Download Athena vocabularies from https://athena.ohdsi.org/"
        )
    if not relationship_csv.exists():
        raise FileNotFoundError(
            f"CONCEPT_RELATIONSHIP.csv(.gz) not found in {athena_dir}."
        )
    if ancestor_csv is None:
        logger.warning(
            "CONCEPT_ANCESTOR.csv not found. Skipping orphan rescue. "
            "For paper-faithful reproduction, include CONCEPT_ANCESTOR.csv "
            "in the Athena download."
        )

    # Override params for dev/testing speed
    if dev:
        num_walks = 10
        glove_epochs = 10
        logger.info("DEV MODE: num_walks=10, glove_epochs=10")

    # Step 1: Build SNOMED graph (with orphan rescue if ancestor_csv available)
    print("KEEP [1/7] Building SNOMED hierarchy graph...")
    graph = build_hierarchy_graph(
        concept_csv, relationship_csv,
        max_depth=max_depth,
        ancestor_csv=ancestor_csv,
    )
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 2: Build ICD-to-SNOMED mappings + generate medcode files
    print("KEEP [2/7] Building ICD-to-SNOMED mappings...")
    icd9_map, icd10_map = build_all_mappings(
        concept_csv, relationship_csv,
        snomed_concept_ids=set(graph.nodes()),
    )
    print(f"  ICD-9: {len(icd9_map)} codes, ICD-10: {len(icd10_map)} codes")

    generate_all_medcode_files(graph, icd9_map, icd10_map)

    # Step 3: Extract patient codes from dataset.
    # Auto-detect MIMIC-III (icd9_code column) vs MIMIC-IV (icd_code + icd_version).
    print("KEEP [3/7] Extracting patient codes...")
    diag_df = (
        dataset.global_event_df
        .filter(pl.col("event_type") == "diagnoses_icd")
        .collect()
    )
    schema = detect_mimic_schema(diag_df.columns)
    print(
        f"  Detected {'MIMIC-IV schema (mixed ICD-9/ICD-10)' if schema == 'mimic4' else 'MIMIC-III schema (ICD-9 only)'}"
    )
    patient_codes = extract_diagnoses_for_schema(
        diag_df,
        icd9_map=icd9_map,
        icd10_map=icd10_map,
        min_occurrences=min_occurrences,
    )
    print(f"  Patients with qualifying codes: {len(patient_codes)}")

    # Step 4: Roll up + co-occurrence matrix
    print("KEEP [4/7] Building co-occurrence matrix...")
    patient_codes = rollup_codes(patient_codes, graph)
    matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
        patient_codes, valid_codes=set(graph.nodes()),
    )
    print(f"  Matrix: {matrix.shape[0]} codes, {int((matrix > 0).sum())} non-zero entries")

    # Step 5: Count filter — drop concepts with zero patient observations
    # (paper-faithful, matches G2Lab's "_ct_filter" file suffix)
    print("KEEP [5/7] Applying count filter...")
    graph, matrix, code_to_idx, idx_to_code = apply_count_filter(
        graph, matrix, code_to_idx, idx_to_code,
    )
    print(f"  Filtered graph: {graph.number_of_nodes()} nodes, matrix {matrix.shape}")

    # Step 6: Node2Vec (Stage 1) — on filtered graph
    print("KEEP [6/7] Training Node2Vec (Stage 1)...")
    n2v_embeddings, node_ids = train_node2vec(
        graph,
        embedding_dim=embedding_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        seed=seed,
    )
    print(f"  Node2Vec embeddings: {n2v_embeddings.shape}")

    # Align Node2Vec to co-occurrence matrix ordering
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    init_emb = np.zeros(
        (len(idx_to_code), embedding_dim), dtype=np.float32,
    )
    for i, code_id in enumerate(idx_to_code):
        if code_id in nid_to_idx:
            init_emb[i] = n2v_embeddings[nid_to_idx[code_id]]

    # Step 7: Regularized GloVe (Stage 2)
    print("KEEP [7/7] Training regularized GloVe (Stage 2)...")
    keep_embeddings = train_keep(
        matrix,
        init_embeddings=init_emb,
        embedding_dim=embedding_dim,
        epochs=glove_epochs,
        lambd=lambd,
        reg_distance=reg_distance,
        reg_reduction=reg_reduction,
        optimizer=optimizer,
        device=device,
        seed=seed,
    )
    print(f"  KEEP embeddings: {keep_embeddings.shape}")

    # Export
    keep_emb_path = str(output_dir / "keep_snomed.txt")
    export_snomed(keep_embeddings, idx_to_code, graph, keep_emb_path)
    print(f"  Exported to: {keep_emb_path}")

    # Also save the co-occurrence matrix + index so post-hoc intrinsic
    # evaluation (co-occurrence correlation, paper Table 2) can run
    # without rebuilding the matrix from MIMIC.
    import json
    cooc_path = output_dir / "cooc_matrix.npy"
    index_path = output_dir / "cooc_index.json"
    np.save(cooc_path, matrix)
    with open(index_path, "w") as f:
        json.dump([int(c) for c in idx_to_code], f)
    print(f"  Saved co-occurrence matrix: {cooc_path}")

    return keep_emb_path
