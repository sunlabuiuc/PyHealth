"""KEEP: Knowledge-preserving and Empirically refined Embedding Process.

Two-stage pretrained medical code embeddings:
  Stage 1: Node2Vec on SNOMED "is-a" hierarchy -> structural embeddings
  Stage 2: Regularized GloVe on patient co-occurrence -> refined embeddings

Reference: Elhussein et al., "KEEP: Integrating Medical Ontologies with
Clinical Data for Robust Code Embeddings", CHIL 2025.

Quick start::

    from pyhealth.medcode.pretrained_embeddings.keep_emb import run_keep_pipeline

    keep_emb_path = run_keep_pipeline(
        athena_dir="data/athena",
        dataset=base_dataset,
        output_dir="keep_output",
    )

    model = GRASP(
        dataset=samples,
        code_mapping=("ICD9CM", "SNOMED"),
        pretrained_emb_path=keep_emb_path,
    )
"""

from .build_omop_graph import build_hierarchy_graph, build_all_mappings
from .generate_medcode_files import generate_all_medcode_files
from .train_node2vec import train_node2vec
from .build_cooccurrence import (
    extract_patient_codes_from_df,
    rollup_codes,
    build_cooccurrence_matrix,
)
from .train_glove import train_keep
from .export_embeddings import export_snomed, export_all
