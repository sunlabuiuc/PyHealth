import logging
import os
from pathlib import Path
import sys

__version__ = "1.1.3"

# package-level cache path
BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
