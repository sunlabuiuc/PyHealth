import os
from urllib.parse import urljoin

import pandas as pd

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import download

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")


def download_and_read_csv(filename: str, refresh_cache: bool = False):
    local_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    online_filepath = urljoin(BASE_URL, filename)
    if (not os.path.exists(local_filepath)) or refresh_cache:
        download(online_filepath, local_filepath)
    return pd.read_csv(local_filepath, dtype=str)
