from nbkit import code, footer, header, md

FILENAME = "06_medical_codes.ipynb"


def cells():
    return [
        *header(
            "06",
            "Medical codes: lookups, mappings and groupers",
            [
                "What the common code systems are (ICD-9, ICD-10, CCS, ATC, NDC) and how they relate",
                "How to look up a code's name, parents and children",
                "How to translate between ICD-9 and ICD-10 and group codes into broader categories",
                "How to use a grouping inside a task to shrink the vocabulary a model must learn",
            ],
            15,
            "Tutorial 02 (processors and vocabularies).",
        ),
        md("""
        ## Why codes need care

        EHR data records diagnoses, procedures and drugs as codes from
        standard systems. Some things to know:

        | System | Codes | Example |
        |---|---|---|
        | ICD-9-CM / ICD-10-CM | diagnoses (US; ICD-10 since 2015) | `428.0` / `I50.9` heart failure |
        | ICD-9-PROC / ICD-10-PCS | procedures | `39.95` hemodialysis |
        | CCS, CCSR | groups of related ICD codes (a few hundred categories) | CCS 108: congestive heart failure |
        | NDC | US drug products (package level) | `00054429731` |
        | ATC | drugs by anatomy and chemistry, 5 levels | `C03CA01` furosemide |

        Fine-grained codes give large, sparse vocabularies: thousands of codes,
        many seen only a handful of times. Grouping them into broader
        categories often helps models, and translating between versions lets
        you combine data from before and after the ICD-10 switch.

        `pyhealth.medcode` has two tools:
        - `InnerMap`: everything *within* one system (names, hierarchy).
        - `CrossMap`: mappings *between* systems.
        """),
        md("""
        ## Looking codes up: `InnerMap`

        The first load of a vocabulary downloads a small table and caches it.
        MIMIC stores ICD-9 codes without the dot (`4280`); lookups accept both
        forms.
        """),
        code("""
        from pyhealth.medcode import InnerMap

        icd9 = InnerMap.load("ICD9CM")
        for c in ["4280", "428.0", "25000", "486"]:
            print(f"{c:6s} {icd9.lookup(c)}")
        """),
        md("""
        Codes form a hierarchy. Ancestors are the broader categories a code
        belongs to; descendants are its more specific codes:
        """),
        code("""
        print("ancestors of 428.0:")
        for a in icd9.get_ancestors("428.0"):
            print(f"  {a:8s} {icd9.lookup(a)}")
        print("\\nmore specific codes under 428.2 (systolic heart failure):")
        for d in icd9.get_descendants("428.2"):
            print(f"  {d:8s} {icd9.lookup(d)}")
        """),
        code("""
        atc = InnerMap.load("ATC")
        for c in ["C03CA01", "C03CA", "C03", "C"]:
            print(f"{c:8s} {atc.lookup(c)}")
        """),
        md("""
        ## Translating and grouping: `CrossMap`

        `CrossMap.load(source, target)` maps codes from one system to another.
        `map` returns a list, because one code can map to several.
        """),
        code("""
        from pyhealth.medcode import CrossMap

        icd9_to_icd10 = CrossMap.load("ICD9CM", "ICD10CM")
        icd10_to_icd9 = CrossMap.load("ICD10CM", "ICD9CM")
        for c in ["428.0", "250.00", "486", "038.9"]:
            print(f"ICD-9 {c:7s} -> ICD-10 {icd9_to_icd10.map(c)}")
        print()
        for c in ["I50.9", "E11.9", "A41.9"]:
            print(f"ICD-10 {c:6s} -> ICD-9 {icd10_to_icd9.map(c)}")
        """),
        md("""
        Notice `038.9` (septicemia) maps to `A41.9`, which maps back to
        `995.91`, not `038.9`. Translation is not a round trip, so translate
        once, in one direction, and say which in your methods.

        Groupers collapse many codes into clinically meaningful categories:
        """),
        code("""
        to_ccs = CrossMap.load("ICD9CM", "CCSCM")
        ccs = InnerMap.load("CCSCM")
        for c in ["428.0", "428.22", "402.91", "250.00", "250.60"]:
            group = to_ccs.map(c)
            print(f"{c:7s} {icd9.lookup(c)[:45]:45s} -> CCS {group} {ccs.lookup(group[0]) if group else ''}")

        print()
        to_chapter = CrossMap.load("ICD10CM", "ICD10CHAPTER")
        to_ccsr = CrossMap.load("ICD10CM", "CCSR")
        for c in ["I50.9", "E11.9", "J18.9"]:
            print(f"{c:6s} chapter {to_chapter.map(c)}  CCSR {to_ccsr.map(c)}")
        """),
        md("""
        Drugs work the same way: NDC product codes map to ATC classes, and
        `target_kwargs={"level": n}` picks how coarse (level 3 is the
        therapeutic subgroup, level 5 the single substance):
        """),
        code("""
        ndc_to_atc = CrossMap.load("NDC", "ATC")
        ndc = "00054429731"
        for level in [3, 4, 5]:
            codes = ndc_to_atc.map(ndc, target_kwargs={"level": level})
            print(f"level {level}: {[(c, atc.lookup(c)) for c in codes]}")
        """),
        md("""
        ## Using a grouping in a task

        Every task accepts `code_mapping={field: (source, target)}`. The
        processors then map each raw code before building the vocabulary, so
        the model sees CCS categories and ATC classes instead of raw codes.
        Let's compare the vocabulary sizes on synthetic MIMIC-III:
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset
        from pyhealth.tasks import MortalityPredictionMIMIC3

        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir="pyhealth_cache",
        )
        raw = dataset.set_task(MortalityPredictionMIMIC3())
        grouped = dataset.set_task(MortalityPredictionMIMIC3(code_mapping={
            "conditions": ("ICD9CM", "CCSCM"),
            "procedures": ("ICD9PROC", "CCSPROC"),
            "drugs": ("NDC", "ATC"),
        }))
        for field in ["conditions", "procedures", "drugs"]:
            n_raw = len(raw.input_processors[field].code_vocab)
            n_grouped = len(grouped.input_processors[field].code_vocab)
            print(f"{field:11s} {n_raw:5d} raw codes -> {n_grouped:4d} groups")
        """),
        md("""
        A model on the grouped version has far fewer embeddings to learn,
        each seen many more times. Whether that helps depends on the question:
        groupings lose detail (all heart-failure subtypes become one category),
        so try both and compare on validation data (Tutorial 03).

        Codes a mapping does not cover fall back to the raw code, so nothing
        is silently dropped. A mapping object records what it could not map:
        """),
        code("""
        translator = CrossMap.load("ICD9CM", "ICD10CM")
        for c in ["428.0", "999.99"]:
            print(c, "->", translator.map(c))
        print("unmapped so far:", translator.unmapped_codes)
        """),
        md("""
        ## Summary

        - `InnerMap.load(system)`: `lookup`, `get_ancestors`, `get_descendants`.
        - `CrossMap.load(source, target).map(code)`: translations and
          groupers; translation is not a round trip.
        - `code_mapping=` on any task groups codes before the vocabulary is
          built, shrinking what the model must learn.
        """),
        *footer("07"),
    ]
