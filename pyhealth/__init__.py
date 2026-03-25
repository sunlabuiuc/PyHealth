import logging
import os
from pathlib import Path
import sys

__version__ = "2.0.0"

# package-level cache path
BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
# BASE_CACHE_PATH = "/srv/local/data/pyhealth-cache"
if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

