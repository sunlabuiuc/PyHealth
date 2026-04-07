"""GRASP + KEEP embeddings on MIMIC-III mortality prediction.

Demonstrates the full KEEP embedding pipeline (Node2Vec + regularized
GloVe) integrated with GRASP for mortality prediction.

Prerequisites:
    - Athena OMOP vocabularies (SNOMED + ICD9CM + ICD10CM) downloaded
      from https://athena.ohdsi.org/ and unzipped to ATHENA_DIR below
    - pip install pyhealth[keep]

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with
       Clinical Data for Robust Code Embeddings", CHIL 2025.
"""

import tempfile
from pathlib import Path

import numpy as np

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import GRASP
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer

# ── Configuration ─────────────────────────────────────────
USE_KEEP = True                 # False = random embeddings, True = KEEP pipeline
ATHENA_DIR = "data/athena"       # path to Athena OMOP vocabulary download
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: build KEEP embeddings (skip if USE_KEEP=False)
    athena_concept = Path(ATHENA_DIR) / "CONCEPT.csv"
    athena_rel = Path(ATHENA_DIR) / "CONCEPT_RELATIONSHIP.csv"
    keep_emb_path = None

    if USE_KEEP and athena_concept.exists():
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph, build_all_mappings,
        )
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_all_medcode_files,
        )
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_node2vec import (
            train_node2vec,
        )
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df, rollup_codes, build_cooccurrence_matrix,
        )
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import train_keep
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_snomed,
        )
        from pyhealth.medcode.codes.icd9cm import ICD9CM
        import polars as pl

        # 2a. Build SNOMED graph + ICD-to-SNOMED mappings from Athena
        graph = build_hierarchy_graph(athena_concept, athena_rel, max_depth=5)
        icd9_map, icd10_map = build_all_mappings(
            athena_concept, athena_rel, snomed_concept_ids=set(graph.nodes()),
        )

        # 2b. Generate medcode files so CrossMap.load("ICD9CM", "SNOMED") works
        generate_all_medcode_files(graph, icd9_map, icd10_map)

        # 2c. Extract patient diagnosis codes from the loaded dataset
        diag_df = (
            base_dataset.global_event_df
            .filter(pl.col("event_type") == "diagnoses_icd")
            .select(["patient_id", "diagnoses_icd/icd9_code"])
            .collect()
        )
        patient_codes = extract_patient_codes_from_df(
            patient_id_col=diag_df["patient_id"].to_list(),
            code_col=diag_df["diagnoses_icd/icd9_code"].to_list(),
            icd_to_snomed=icd9_map,
            standardize_icd9=ICD9CM.standardize,
            min_occurrences=2,
        )

        # 2d. Roll up to ancestors + build co-occurrence matrix
        patient_codes = rollup_codes(patient_codes, graph)
        matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
            patient_codes, valid_codes=set(graph.nodes()),
        )

        # 2e. Node2Vec (Stage 1) — small num_walks for dev speed
        n2v_embeddings, node_ids = train_node2vec(
            graph, embedding_dim=100, num_walks=10, seed=42,
        )

        # 2f. Regularized GloVe (Stage 2) — few epochs for dev speed
        nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        init_emb = np.zeros((len(idx_to_code), 100), dtype=np.float32)
        for i, code_id in enumerate(idx_to_code):
            if code_id in nid_to_idx:
                init_emb[i] = n2v_embeddings[nid_to_idx[code_id]]

        keep_embeddings = train_keep(
            matrix, init_embeddings=init_emb, embedding_dim=100,
            epochs=10, lambd=1e-3, seed=42,
        )

        # 2g. Export
        output_dir = Path("keep_output")
        output_dir.mkdir(exist_ok=True)
        keep_emb_path = str(output_dir / "keep_snomed.txt")
        export_snomed(keep_embeddings, idx_to_code, graph, keep_emb_path)
        print(f"KEEP embeddings exported: {keep_embeddings.shape} -> {keep_emb_path}")
    elif USE_KEEP:
        print(f"Athena data not found at {ATHENA_DIR}, skipping KEEP pipeline.")
    else:
        print("USE_KEEP=False, using random embeddings.")

    # STEP 3: set task
    if keep_emb_path is not None:
        task = MortalityPredictionMIMIC3(
            code_mapping={
                "conditions": ("ICD9CM", "SNOMED"),
                "procedures": ("ICD9PROC", "CCSPROC"),
                "drugs": ("NDC", "ATC"),
            }
        )
    else:
        task = MortalityPredictionMIMIC3()

    sample_dataset = base_dataset.set_task(task)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 4: define model
    model = GRASP(
        dataset=sample_dataset,
        embedding_dim=100,
        cluster_num=2,
        pretrained_emb_path=keep_emb_path,
    )

    # STEP 5: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="roc_auc",
    )

    # STEP 6: evaluate
    print(trainer.evaluate(test_dataloader))
