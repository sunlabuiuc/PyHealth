import os
from pathlib import Path

CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")

if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)
