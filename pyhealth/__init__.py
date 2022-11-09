import logging
import os
import sys
from pathlib import Path

__version__ = "1.0a2"

# package-level cache path
BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)

# logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# streamHandler
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
