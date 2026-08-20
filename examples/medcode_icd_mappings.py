"""ICD translation and grouper vocabularies in pyhealth.medcode.

Runs entirely offline -- the mappings shown here ship inside the
``icd-mappings`` wheel rather than being downloaded.

Author: John Wu (johnwu3)
"""

from pyhealth.medcode import CrossMap

print("=== ICD-9 <-> ICD-10 translation ===")
icd9_to_icd10 = CrossMap.load("ICD9CM", "ICD10CM")
print("backend:", icd9_to_icd10.backend)
for code in ["428.0", "250.00", "486"]:
    print(f"  ICD-9 {code:7} -> ICD-10 {icd9_to_icd10.map(code)}")

icd10_to_icd9 = CrossMap.load("ICD10CM", "ICD9CM")
print("  ICD-10 I50.9   -> ICD-9 ", icd10_to_icd9.map("I50.9"))
print("  note: 428.0 -> I50.9 -> 428.9, so translation does not round-trip")

print("\n=== grouper vocabularies ===")
for source, target, code in [
    ("ICD10CM", "CCSR", "I50.9"),
    ("ICD9CM", "CCI", "428.0"),
    ("ICD10CM", "CCIR", "I50.9"),
    ("ICD9CM", "ICD9CHAPTER", "428.0"),
    ("ICD10CM", "ICD10CHAPTER", "I50.9"),
    ("ICD10CM", "ICD10BLOCK", "I50.9"),
    ("ICD10CM", "CCC", "I50.9"),
    ("ICD10CM", "CCCSUB", "I50.9"),
]:
    mapping = CrossMap.load(source, target)
    print(f"  {source:8} -> {target:13} {code:7} -> {mapping.map(code)}")

print("\n=== PyHealth's own tables still take precedence ===")
icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
print("  ICD9CM -> CCSCM backend:", icd9_to_ccs.backend)

print("\n=== quantifying mapping loss ===")
translator = CrossMap.load("ICD9CM", "ICD10CM")
for code in ["428.0", "999.99"]:
    print(f"  {code:7} -> {translator.map(code)}")
print("  unmapped so far:", translator.unmapped_codes)
