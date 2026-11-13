"""Look up and map medical codes with pyhealth.medcode.

Looks up an NDC drug code, maps it to ATC with ``CrossMap``, and looks up the
ATC code with ``InnerMap``. Needs internet on the first run only: the NDC and
ATC vocabularies are downloaded (a few MB) and cached.

Usage:
    python examples/medcode.py
"""

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
