"""SNOMED Clinical Terms vocabulary for PyHealth.

SNOMED CT is a clinical ontology with ~350K concepts connected by
"Is a" relationships encoding medical knowledge. Unlike ICD (a billing
taxonomy), SNOMED is a polyhierarchy — a concept can have multiple
parents (e.g., "Hypertensive heart disease" is both a cardiovascular
disease and a hypertensive disorder).

SNOMED concept codes are used as tokens by KEEP embeddings. Adding
SNOMED as a code_mapping target allows any PyHealth model to use
KEEP embeddings via:

    code_mapping=("ICD9CM", "SNOMED")

The SNOMED vocabulary graph is generated from Athena OMOP data by
the KEEP pipeline (``keep_emb.generate_medcode_files``). Unlike other
PyHealth vocabularies hosted on GCS, SNOMED cannot be redistributed
due to IHTSDO licensing restrictions. Instead, users download Athena
vocabularies directly from https://athena.ohdsi.org/ (free account,
instant download) and the KEEP pipeline generates the medcode files
locally.

For KEEP, only three Athena vocabularies are required:
    - **SNOMED** (for the knowledge graph hierarchy)
    - **ICD9CM** (for ICD-9 to SNOMED cross-mapping)
    - **ICD10CM** (for ICD-10 to SNOMED cross-mapping)

Athena offers many more vocabularies (CPT4, LOINC, RxNorm, ATC,
OPCS4, UK Biobank, etc.) but KEEP only needs the three above.

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
"""

from pyhealth.medcode.inner_map import InnerMap


class SNOMED(InnerMap):
    """SNOMED Clinical Terms.

    A clinical ontology with hierarchical "Is a" relationships.
    Concept codes are numeric strings (e.g., "84114007" for Heart failure).

    Note:
        Unlike other PyHealth vocabularies, SNOMED data is not hosted on
        GCS due to IHTSDO licensing restrictions on redistribution. Users
        download Athena vocabularies from https://athena.ohdsi.org/ (free,
        select SNOMED + ICD9CM + ICD10CM) and run the KEEP pipeline's
        ``generate_medcode_files()`` to generate the local medcode CSVs.

    Examples:
        >>> from pyhealth.medcode import InnerMap
        >>> snomed = InnerMap.load("SNOMED")
        >>> snomed.lookup("84114007")
        'Heart failure'
        >>> snomed.get_ancestors("84114007")
        ['49601007', ...]
    """

    def __init__(self, **kwargs):
        super(SNOMED, self).__init__(vocabulary="SNOMED", **kwargs)

    @staticmethod
    def standardize(code: str):
        """Standardizes SNOMED code. SNOMED codes are already standardized."""
        return str(code)
