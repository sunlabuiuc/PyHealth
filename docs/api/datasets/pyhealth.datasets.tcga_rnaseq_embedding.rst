pyhealth.datasets.TCGARNASeqEmbeddingDataset
============================================

TCGA RNA-seq embedding dataset that pairs pre-computed
:class:`~pyhealth.models.BulkRNABert` encoder outputs with TCGA cancer-type
labels for the 5-cohort classification benchmark
(:class:`~pyhealth.tasks.TCGACancerClassification5Cohort`). The dataset
reads a ``(n_samples, embed_dim)`` ``.npy`` matrix plus an identifier CSV
(row ``i`` of the matrix corresponds to row ``i`` of the CSV) and joins
each row against the TCGA GDC file-mapping CSV to resolve the cohort label.

.. autoclass:: pyhealth.datasets.TCGARNASeqEmbeddingDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.load_tcga_cancer_classification_5cohort

.. autofunction:: pyhealth.datasets.tcga_rnaseq_embedding.stratified_split_indices
