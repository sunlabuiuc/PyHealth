"""ICD translation and grouper vocabularies in pyhealth.medcode.

Runs entirely offline -- the mappings shown here ship inside the
``icd-mappings`` wheel rather than being downloaded.

Author: John Wu (johnwu3)
"""

from pyhealth.medcode import CrossMap

print("=== ICD-9 <-> ICD-10 translation ===")
icd9_to_icd10 = CrossMap.load("ICD9CM", "ICD10CM")
print("backend:", icd9_to_icd10.backend)
for code in ["428.0", "250.00", "486", "038.9", "584.9", "V30.00"]:
    print(f"  ICD-9 {code:7} -> ICD-10 {icd9_to_icd10.map(code)}")

icd10_to_icd9 = CrossMap.load("ICD10CM", "ICD9CM")
for code in ["I50.9", "E11.9", "J18.9", "N17.9", "A41.9"]:
    print(f"  ICD-10 {code:8} -> ICD-9 {icd10_to_icd9.map(code)}")
print("  note: 038.9 -> A41.9 -> 995.91, so translation does not round-trip")

print("\n=== grouper vocabularies ===")
for source, target, codes in [
    ("ICD10CM", "CCSR", ["I50.9", "E11.9", "J18.9", "N17.9"]),
    ("ICD9CM", "CCI", ["428.0", "250.00", "486"]),
    ("ICD10CM", "CCIR", ["I50.9", "S72.001A"]),
    ("ICD9CM", "ICD9CHAPTER", ["428.0", "038.9", "800.0"]),
    ("ICD10CM", "ICD10CHAPTER", ["I50.9", "E11.9", "A41.9"]),
    ("ICD10CM", "ICD10BLOCK", ["I50.9", "E11.9", "A41.9"]),
    ("ICD10CM", "CCC", ["I50.9", "J18.9"]),
    ("ICD10CM", "CCCSUB", ["I50.9", "J18.9"]),
]:
    mapping = CrossMap.load(source, target)
    rendered = "  ".join(f"{c}->{mapping.map(c)}" for c in codes)
    print(f"  {source:8} -> {target:13} {rendered}")

print("\n=== PyHealth's own tables still take precedence ===")
icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
print("  ICD9CM -> CCSCM backend:", icd9_to_ccs.backend)
print("  same pair, forced offline:",
      CrossMap.load("ICD9CM", "CCSCM", backend="icdmappings").backend)

print("\n=== quantifying mapping loss ===")
translator = CrossMap.load("ICD9CM", "ICD10CM")
for code in ["428.0", "999.99"]:
    print(f"  {code:7} -> {translator.map(code)}")
print("  unmapped so far:", translator.unmapped_codes)
