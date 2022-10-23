import os
from pathlib import Path

__version__ = "1.0a2"

BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")

if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)
