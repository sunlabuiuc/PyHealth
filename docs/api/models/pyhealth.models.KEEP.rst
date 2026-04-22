KEEP — Pretrained Medical-Code Embeddings
=========================================

Overview
--------

KEEP (**K**\ nowledge-preserving and **E**\ mpirically refined **E**\ mbedding
**P**\ rocess) produces pretrained medical-code embeddings that any PyHealth
model accepting a ``pretrained_emb_path`` — for example
:class:`~pyhealth.models.GRASP` — can consume directly. The pipeline combines
ontology structure with patient co-occurrence statistics in two stages:

- **Stage 1** — Node2Vec over the SNOMED "Is a" hierarchy yields structural
  embeddings anchored in medical knowledge.
- **Stage 2** — A regularized GloVe objective refines the Stage 1 vectors
  against patient-level code co-occurrence, pulling empirically related
  concepts closer while staying anchored to the ontology.

Reference: Elhussein et al., *"KEEP: Integrating Medical Ontologies with
Clinical Data for Robust Code Embeddings"*, CHIL 2025.

Because SNOMED itself cannot be redistributed under IHTSDO licensing, the
pipeline reads Athena OMOP vocabulary dumps (downloaded locally from
https://athena.ohdsi.org/) and generates the SNOMED graph and ICD↔SNOMED
cross-maps on the user's machine.

Quick start
-----------

.. code-block:: python

    from pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline import (
        run_keep_pipeline,
    )
    from pyhealth.models import GRASP

    # End-to-end: builds SNOMED graph, trains Node2Vec + regularized GloVe,
    # and writes keep_snomed.txt.
    keep_emb_path = run_keep_pipeline(
        athena_dir="data/athena",
        dataset=base_dataset,
        output_dir="output/embeddings/keep",
    )

    model = GRASP(
        dataset=samples,
        code_mapping=("ICD9CM", "SNOMED"),
        pretrained_emb_path=keep_emb_path,
    )

See ``examples/mortality_prediction/mortality_mimic4_grasp_keep.py`` for a
full MIMIC-IV mortality-prediction walkthrough.

API Reference
-------------

End-to-end pipeline
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline.run_keep_pipeline

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline.resolve_device

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.run_pipeline.detect_mimic_schema

Stage 1 — SNOMED hierarchy and Node2Vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.build_hierarchy_graph

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.build_all_mappings

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.generate_all_medcode_files

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.train_node2vec

Stage 2 — Patient co-occurrence and regularized GloVe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.extract_patient_codes_from_df

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.rollup_codes

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence_matrix

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.apply_count_filter

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.train_keep

.. autoclass:: pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove.KeepGloVe
    :members:
    :undoc-members:
    :show-inheritance:

Exporting and evaluating embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.export_snomed

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.export_all

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.load_keep_embeddings

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.evaluate_embeddings

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.resnik_correlation

.. autofunction:: pyhealth.medcode.pretrained_embeddings.keep_emb.cooccurrence_correlation
