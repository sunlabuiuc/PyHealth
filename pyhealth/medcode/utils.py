import logging
import os
from urllib.parse import urljoin
from urllib.request import urlretrieve

import pandas as pd

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")
create_directory(MODULE_CACHE_PATH)

logger = logging.getLogger(__name__)


def download_and_read_csv(filename: str, refresh_cache: bool = False) -> pd.DataFrame:
    """Reads a csv file from the pyhealth resource folder.

    This function will read the csv file from `MODULE_CACHE_PATH` if it exists.
    Otherwise, it will download the csv file from `BASE_URL` and save it to
    `MODULE_CACHE_PATH`.

    Args:
        filename: The name of the csv file.
        refresh_cache: Whether to refresh the cache. Default is False.

    Returns:
        A pandas DataFrame.
    """
    local_filepath = os.path.join(MODULE_CACHE_PATH, filename)
    online_filepath = urljoin(BASE_URL, filename)
    if (not os.path.exists(local_filepath)) or refresh_cache:
        logger.debug(f"downloading {online_filepath} to {local_filepath}")
        urlretrieve(online_filepath, local_filepath)
    return pd.read_csv(local_filepath, dtype=str)
