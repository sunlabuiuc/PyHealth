from pyhealth.medcode import CrossMap
from pyhealth.medcode import NDC, ATC, ICD9CM

icd9cm = ICD9CM()
print("Looking up for ICD9CM code 428.0")
print(icd9cm.lookup("428.0"))

ndc = NDC()
print("Looking up for NDC code 00597005801")
print(ndc.lookup("00597005801"))

codemap = CrossMap("NDC", "ATC")
print("Mapping NDC code 00597005801 to ATC")
print(codemap.map("00597005801"))
atc = ATC()
print("Looking up for ATC code G04CA02")
print(atc.lookup("G04CA02"))
