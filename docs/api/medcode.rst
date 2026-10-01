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
