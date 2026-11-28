pyhealth.models.HiCu
--------------------

HiCu (Ren et al., MLHC 2022) predicts ICD codes from clinical notes with a
MultiResCNN encoder and a per-label attention decoder, trained with a
curriculum over the ICD code tree: chapters first, then blocks, 3-digit
codes, and finally the full codes. Each level starts from its parent level's
attention queries (knowledge transfer), corrected by Poincare embeddings of
the tree (``hyperbolic="add"`` adds them to the queries, ``"cat"``
concatenates them), and the asymmetric loss helps the rare codes. Train it by calling
:meth:`~pyhealth.models.HiCu.set_level` with each level in turn.

Paper: `HiCu: Leveraging Hierarchy for Curriculum Learning in Automated ICD
Coding <https://proceedings.mlr.press/v182/ren22a.html>`_.

See ``examples/mimic3_icd_coding_hicu.py`` for training on MIMIC-III and
the ablation study.

.. autoclass:: pyhealth.models.HiCu
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.hicu.icd9_code_path

.. autofunction:: pyhealth.models.hicu.train_poincare_embeddings

.. autoclass:: pyhealth.models.hicu.AsymmetricLoss
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.hicu.MultiResCNNEncoder
    :members:
    :show-inheritance:
