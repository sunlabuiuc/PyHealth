from pyhealth.medcode import CrossMap, InnerMap

ndc = InnerMap.load("NDC")
print("Looking up for NDC code 00597005801")
print(ndc.lookup("00597005801"))

codemap = CrossMap.load("NDC", "ATC")
print("Mapping NDC code 00597005801 to ATC")
print(codemap.map("00597005801"))

atc = InnerMap.load("ATC")
print("Looking up for ATC code G04CA02")
print(atc.lookup("G04CA02"))
