from pyhealth.medcode import CrossMap, InnerMap

# The first load downloads the mapping files and caches them under
# ~/.cache/pyhealth/medcode/; later loads read the cache. Downloads are atomic,
# so an interrupted run can simply be retried. Use refresh_cache=True to
# re-download, e.g. InnerMap.load("NDC", refresh_cache=True).
ndc = InnerMap.load("NDC")
print("Looking up for NDC code 00597005801")
print(ndc.lookup("00597005801"))

codemap = CrossMap.load("NDC", "ATC")
print("Mapping NDC code 00597005801 to ATC")
print(codemap.map("00597005801"))

atc = InnerMap.load("ATC")
print("Looking up for ATC code G04CA02")
print(atc.lookup("G04CA02"))
