MedCode
===================================

Translating Between Medical Standards
-------------------------------------

Healthcare data comes in many different coding systems and standards. PyHealth's medical code mapping enables:

- Cross-system mapping (e.g., ICD9CM → CCSCM, NDC → ATC)
- Within-system ontology lookup (e.g., ancestors/descendants in ICD, ATC hierarchy)

Quick Examples
^^^^^^^^^^^^^^

.. code-block:: python

   from pyhealth.medcode import CrossMap, InnerMap

   # Cross-system mapping: ICD-9-CM → CCS
   icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
   print(icd9_to_ccs.map("82101"))  # example diagnosis code

   # Drug code mapping: NDC → ATC
   ndc_to_atc = CrossMap.load("NDC", "ATC")
   print(ndc_to_atc.map("00527051210"))

   # Within-system lookup: ICD-9-CM
   icd9cm = InnerMap.load("ICD9CM")
   print(icd9cm.lookup("428.0"))
   print(icd9cm.get_ancestors("428.0"))

   # Translating between ICD versions
   icd9_to_icd10 = CrossMap.load("ICD9CM", "ICD10CM")
   print(icd9_to_icd10.map("428.0"))          # ['I50.9']

   # Grouping ICD-10 into CCSR categories
   icd10_to_ccsr = CrossMap.load("ICD10CM", "CCSR")
   print(icd10_to_ccsr.map("I50.9"))          # ['CIR019']

   # Coarser groupings, useful when a cohort is small
   print(CrossMap.load("ICD9CM", "ICD9CHAPTER").map("428.0"))       # ['7']
   print(CrossMap.load("ICD10CM", "ICD10CHAPTER").map("E11.9"))     # ['E00-E89']
   print(CrossMap.load("ICD10CM", "ICD10BLOCK").map("A41.9"))       # ['A30-A49']

   # Binary chronic-condition flags
   print(CrossMap.load("ICD9CM", "CCI").map("428.0"))               # ['1']
   print(CrossMap.load("ICD9CM", "CCI").map("486"))                 # ['0']

   # Which source served the mapping
   print(icd10_to_ccsr.backend)                                     # 'icdmappings'

We provide medical code mapping tools for (i) ontology mapping within one coding system and 
(ii) mapping the same concept cross different coding systems. 


.. autoclass:: pyhealth.medcode.InnerMap
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CrossMap
    :members:
    :undoc-members:
    :show-inheritance:

Diagnosis codes:
----------------------

.. autoclass:: pyhealth.medcode.ICD9CM
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10CM
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCSCM
    :members:
    :undoc-members:
    :show-inheritance:

Procedure codes:
----------------------

.. autoclass:: pyhealth.medcode.ICD9PROC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10PROC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCSPROC
    :members:
    :undoc-members:
    :show-inheritance:

Medication codes:
-----------------------

.. autoclass:: pyhealth.medcode.NDC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.RxNorm
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ATC
    :members:
    :undoc-members:
    :show-inheritance:

Knowledge graph embeddings
--------------------------

``pyhealth.medcode.pretrained_embeddings.kg_emb`` trains TransE, RotatE,
DistMult and ComplEx on a knowledge graph with the standard streaming
pipeline:

1. :class:`~pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.BaseKGDataset`
   is a :class:`~pyhealth.datasets.BaseDataset` with one ``triples`` table
   (``head``, ``relation``, ``tail``) declared in YAML with
   ``patient_id: null``, so each triple is one record.
   :class:`~pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.UMLSDataset`
   is its bundled UMLS configuration. Entity and relation ids are assigned on
   the full graph, by sorted name.
2. ``set_task(KGLinkPrediction(...), split=PatientSplit(...))`` splits the
   triples and fits the processors on the training part only, returning one
   streaming :class:`~pyhealth.datasets.SampleDataset` per part. Each sample
   holds a ``triple`` and its ``ground_truth_head`` / ``ground_truth_tail``
   lists, which cover the whole graph and filter the ranking at evaluation
   (filtered setting of Bordes et al., 2013). Duplicate triples are removed
   and counted.
3. The ``kg_triple`` processor
   (:class:`~pyhealth.processors.KGTripleProcessor`), fitted on the training
   triples, gives the models the numbers of entities and relations and the
   dicts that keep known training positives out of the training negatives,
   as in the reference implementation of Sun et al. (2019). Validation and
   test triples therefore never shape training. ``set_task`` warns when
   called without ``split``.
4. The models take ``negative_sampling`` as an argument and follow
   ``model.train()`` / ``model.eval()``, as set by
   :class:`~pyhealth.trainer.Trainer`; use the ``hits@n`` and ``mean_rank``
   metrics.

See ``examples/kg_emb_link_prediction.py`` for a self-contained walk-through.

The PyHealth 1.x entry points, ``link_prediction_fn``,
:class:`~pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.SampleKGDataset`
and :func:`~pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.split`,
still work for one release with a ``DeprecationWarning``. They keep their
1.x behaviour, including training negatives filtered with every triple of
the graph, held-out ones included; only the 2.0 path above avoids that. A
model must be built from a sample dataset part, not from the
``BaseKGDataset`` itself. Ids no longer follow the 1.x order of first
appearance: map 1.x embeddings through the ``id2entity`` saved with them.

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.BaseKGDataset
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.UMLSDataset
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.tasks.KGLinkPrediction
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.SampleKGDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.medcode.pretrained_embeddings.kg_emb.datasets.split

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.models.KGEBaseModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.models.TransE
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.models.RotatE
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.models.DistMult
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.pretrained_embeddings.kg_emb.models.ComplEx
    :members:
    :undoc-members:
    :show-inheritance:



    

ICD translation and grouper vocabularies
-----------------------------------------

PyHealth hosts its own mapping tables and always prefers them. Pairs it has no
table for are served by the `icd-mappings <https://pypi.org/project/icd-mappings/>`_
package, whose data ships inside its wheel and therefore needs no network
access. ``CrossMap.backend`` reports which source was used, and the choice can
be forced with ``CrossMap.load(..., backend="pyhealth")`` or
``backend="icdmappings"``.

``ICD9CM->CCSCM`` is the one pair both sources can serve. It stays on PyHealth's
hosted table under ``backend="auto"`` so that existing pipelines are unaffected;
pass ``backend="icdmappings"`` explicitly to get the same mapping offline.

Supported pairs, by source:

================================  ==========================================
Pair                              Notes
================================  ==========================================
``ICD9CM``   -> ``ICD10CM``       NBER General Equivalence Mappings
``ICD10CM``  -> ``ICD9CM``        NBER General Equivalence Mappings
``ICD10CM``  -> ``CCSR``          530 refined categories, ICD-10 only
``ICD9CM``   -> ``CCI``           chronic condition indicator, ``"1"``/``"0"``
``ICD10CM``  -> ``CCIR``          refined chronic condition indicator
``ICD9CM``   -> ``ICD9CHAPTER``   19 chapters
``ICD10CM``  -> ``ICD10CHAPTER``  22 chapters, keyed by code range
``ICD10CM``  -> ``ICD10BLOCK``    226 blocks, keyed by code range
``ICD9CM``/``ICD10CM`` -> ``CCC``     pediatric complex chronic condition
``ICD9CM``/``ICD10CM`` -> ``CCCSUB``  its subcategory
================================  ==========================================

.. warning::

   ICD-9 <-> ICD-10 translation is a **primary-mapping approximation**, not the
   full many-to-many GEM relation. ``map()`` returns at most one target code,
   some codes have no mapping at all, and the translation does not round-trip
   (``428.0`` maps to ``I50.9``, which maps back to ``428.9``). Measured on the
   581 distinct ICD-9 codes in the MIMIC-III demo, 561 mapped and 17 distinct
   ICD-9 codes collapsed onto a shared ICD-10 code. Inspect
   ``CrossMap.unmapped_codes`` after a pass over a dataset to quantify the loss.

   For grouping mixed ICD-9/ICD-10 data into one feature space, prefer mapping
   both versions into a shared grouper (CCS or CCSR) rather than translating
   one version into the other.

.. autoclass:: pyhealth.medcode.FlatMap
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCSR
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCI
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCIR
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD9CHAPTER
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10CHAPTER
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10BLOCK
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCCSUB
    :members:
    :undoc-members:
    :show-inheritance:
