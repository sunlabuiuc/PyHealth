import os
from pathlib import Path

BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")

if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)
