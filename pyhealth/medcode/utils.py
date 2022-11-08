import os
from urllib.parse import urljoin
from urllib.request import urlretrieve

import pandas as pd

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")
create_directory(MODULE_CACHE_PATH)


def download_and_read_csv(filename: str, refresh_cache: bool = False):
    local_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    online_filepath = urljoin(BASE_URL, filename)
    if (not os.path.exists(local_filepath)) or refresh_cache:
        urlretrieve(online_filepath, local_filepath)
    return pd.read_csv(local_filepath, dtype=str)
