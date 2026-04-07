"""KEEP: Knowledge-preserving and Empirically refined Embedding Process.

Two-stage pretrained medical code embeddings:
  Stage 1: Node2Vec on SNOMED "is-a" hierarchy -> structural embeddings
  Stage 2: Regularized GloVe on patient co-occurrence -> refined embeddings

Reference: Elhussein et al., "KEEP: Integrating Medical Ontologies with
Clinical Data for Robust Code Embeddings", CHIL 2025.
"""
