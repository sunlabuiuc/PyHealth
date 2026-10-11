import logging
import os
from pathlib import Path
import sys

__version__ = "2.0.2"

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
# PyHealth prints through its own handler above, so don't also pass records
# to the root logger: in Jupyter/Colab (or after logging.basicConfig) the root
# logger has a handler too, and every line would print twice. To route
# PyHealth logs through your own root configuration instead, remove the
# handler and turn propagation back on:
#     logging.getLogger("pyhealth").removeHandler(pyhealth.handler)
#     logging.getLogger("pyhealth").propagate = True
logger.propagate = False

