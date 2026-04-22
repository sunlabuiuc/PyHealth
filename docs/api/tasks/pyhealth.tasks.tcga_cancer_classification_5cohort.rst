pyhealth.tasks.TCGACancerClassification5Cohort
==============================================

5-way TCGA cancer-type classification task over the ``{BLCA, BRCA,
GBM+LGG, LUAD, UCEC}`` cohorts, used with the
:class:`~pyhealth.models.BulkRNABertClassifier` downstream head on
pre-computed :class:`~pyhealth.models.BulkRNABert` embeddings. The task
consumes an ``(embed_dim,)`` float tensor per sample and emits a label in
``{0, 1, 2, 3, 4}`` via the ``LABEL_MAP`` defined in this module.

.. autoclass:: pyhealth.tasks.TCGACancerClassification5Cohort
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: pyhealth.tasks.tcga_cancer_classification_5cohort.LABEL_MAP
    :annotation:

.. autodata:: pyhealth.tasks.tcga_cancer_classification_5cohort.COHORT_NAMES
    :annotation:
