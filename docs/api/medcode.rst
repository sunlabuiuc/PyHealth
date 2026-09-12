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
DistMult and ComplEx on an in-memory list of triples. Since PyHealth 2.0
the sample dataset is a map-style :class:`torch.utils.data.Dataset`. Build
the loader with :class:`torch.utils.data.DataLoader` and
:func:`pyhealth.datasets.collate_fn_dict_with_padding` --
:func:`pyhealth.datasets.get_dataloader` is streaming-only and calls
``set_shuffle()``.

See ``examples/kg_emb_sample_dataset.py`` for a self-contained walk-through.

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



    