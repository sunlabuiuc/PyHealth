from pyhealth.medcode import CodeMap
from pyhealth.medcode import NDC, ATC, ICD9CM


def print_dict(d):
    print(d["name"])
    del d["name"]
    for k, v in d.items():
        if len(v) < 50:
            print(f"\t- {k.upper()}: {v}")
        else:
            print(f"\t- {k.upper()}: {v[:50]}...")
    print()


def print_list(l):
    for v in l:
        print(v)
    print()


icd9cm = ICD9CM()
print("Looking up for ICD9CM code 428.0")
print_dict(icd9cm.lookup("428.0"))

ndc = NDC()
print("Looking up for NDC code 00597005801")
print_dict(ndc.lookup("00597005801"))

codemap = CodeMap("NDC", "ATC")
print("Mapping NDC code 00597005801 to ATC")
print_list(codemap.map("00597005801"))
atc = ATC()
print("Looking up for ATC code G04CA02")
print_dict(atc.lookup("G04CA02"))
