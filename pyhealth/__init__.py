import logging
import os
from pathlib import Path
import sys

__version__ = "2.0.0"

# package-level cache path
_DEFAULT_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
# Allow overriding cache location (useful for sandboxed environments).
BASE_CACHE_PATH = os.environ.get("PYHEALTH_CACHE_PATH", _DEFAULT_CACHE_PATH)
# BASE_CACHE_PATH = "/srv/local/data/pyhealth-cache"
if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH, exist_ok=True)

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

