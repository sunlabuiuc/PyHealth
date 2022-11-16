import logging
import os
from pathlib import Path

__version__ = "1.1.1"

# package-level cache path
BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)
